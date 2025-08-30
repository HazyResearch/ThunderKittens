#define RED_TEXT "\033[31m"
#define GREEN_TEXT "\033[32m"
#define YELLOW_TEXT "\033[33m"
#define BLUE_TEXT "\033[34m"
#define MAGENTA_TEXT "\033[35m"
#define CYAN_TEXT "\033[36m"
#define WHITE_TEXT "\033[37m"
#define RESET_TEXT "\033[0m"

#include "kittens.cuh"
// #define KVM_DEBUG
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;


using config = default_config;
struct globals
{
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using weights = gl<bf16, 1, -1, -1, 2048, st_bf<16, 512>>; // assumed to be N by 2048 (X@W.T).
    using activations = gl<bf16, 1, 1, 1, 2048, sv_bf<2048>, sv_bf<16>>;
    using barriers = gl<bf16, 1, -1, 6, 32>; // num_layers by 6 ops per layer by up to 32 heads.
    
    instruction_layout instructions;
    timing_layout timings;

    weights UP_PROJ_W;
    weights GATE_PROJ_W;
    activations INP;
    activations O;
    barriers Bar;
    
    // persistent grid structure
    dim3 grid() { return dim3(148); } 
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template <typename config = config, int _OP_IDX = 0>
struct SiLU_MLPOp
{
    static constexpr int opcode = 4;
    static constexpr int OP_IDX = _OP_IDX; // Op index within the layer -- controls which barrier to listen to.
    struct parsed_instruction
    {
        int layer, start_col;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction)
        {
            layer = instruction[1];     // in units of 1
            start_col = instruction[2]; // in units of 1
        }
        __device__ inline parsed_instruction(state<config> &s) : parsed_instruction(s.instruction()) {}
    };
    static __device__ inline parsed_instruction parse_instruction(const globals &g, state<config> &s)
    {
        return parsed_instruction{s.instruction()[1], s.instruction()[2]};
    }

    static constexpr int UP_PAGES    = 4;
    static constexpr int GATE_PAGES  = 4;
    static constexpr int PAGE_INPUT  = UP_PAGES + GATE_PAGES;    // = 8
    static constexpr int PAGE_OUTPUT = PAGE_INPUT + 1;           // = 9
    static constexpr int SEM_COUNT   = PAGE_OUTPUT + 1;          // = 10

    //  semaphores 
  __device__ static inline semaphore &up_arrived   (state<config> &s, int i) { return s.semaphores()[ i            ]; }
  __device__ static inline semaphore &gate_arrived (state<config> &s, int i) { return s.semaphores()[ UP_PAGES + i ]; }
  __device__ static inline semaphore &in_arrived   (state<config> &s)        { return s.semaphores()[ PAGE_INPUT   ]; }
  __device__ static inline semaphore &out_arrived  (state<config> &s)        { return s.semaphores()[ PAGE_OUTPUT  ]; }

    // getters
    __device__ static inline int get_up_page  (state<config> &s, int i) { return s.pid(i); }
    __device__ static inline int get_gate_page(state<config> &s, int i) { return s.pid(UP_PAGES + i); }
    __device__ static inline int get_input_page(state<config> &s) { return s.pid(PAGE_INPUT); }
    __device__ static inline int get_output_page(state<config> &s) { return s.pid(PAGE_OUTPUT); }

    struct controller
    {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query)
        {
            int ret_order[] = {
                6, 7, 8, 9, 10, 11, 12, 
                13,
                0, 1, 2, 3, 4, 5
            };
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s)
        {
            // each weight page and the input page needs exactly 1 “ready” signal
            for (int i = 0; i < UP_PAGES;   i++) init_semaphore(up_arrived(s,i),   1);
            for (int i = 0; i < GATE_PAGES; i++) init_semaphore(gate_arrived(s,i), 1);
            init_semaphore(in_arrived(s),   1);
            // output must wait for all 4 consumer warps
            init_semaphore(out_arrived(s),  16);
            
            return SEM_COUNT;
        }
    };


    struct loader
    {
        static __device__ void run(const globals &g, state<config> &s)
        {
            parsed_instruction inst{s};
            // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
            ((int*)s.scratch())[laneid()] = 0;
            warp::sync(); // done, now we can proceed to other things.

            // 1) UP projections
            if (laneid() < UP_PAGES)
            {
                int pg = get_up_page(s, laneid());
                s.wait_page_ready(pg);
                auto &chunk = reinterpret_cast<st_bf<16,512>&>(s.pages[pg]);
                tma::expect(up_arrived(s,laneid()), chunk);
                tma::load_async(chunk, g.UP_PROJ_W,
                                {inst.layer, inst.start_col/16, laneid()},
                                up_arrived(s,laneid()));
            }

            // 2) GATE projections
            else if (laneid() < UP_PAGES + GATE_PAGES)
            {
                int idx = laneid() - UP_PAGES;
                int pg  = get_gate_page(s, idx);
                s.wait_page_ready(pg);
                auto &chunk = reinterpret_cast<st_bf<16,512>&>(s.pages[pg]);
                tma::expect(gate_arrived(s,idx), chunk);
                tma::load_async(chunk, g.GATE_PROJ_W,
                                {inst.layer, inst.start_col/16, idx},
                                gate_arrived(s,idx));
            }

            // 4) INPUT page
            else if (laneid() == PAGE_INPUT)
            {
                int pg = get_input_page(s);
                s.wait_page_ready(pg);
                // wait on barrier from previous op
                while (*(volatile int*)&g.Bar[{inst.layer, OP_IDX, 0}] == 0)
                    __nanosleep(20);
                auto &buf = reinterpret_cast<sv_bf<2048>&>(s.pages[pg]);
                tma::expect(in_arrived(s), buf);
                tma::load_async(buf, g.INP, {}, in_arrived(s));
            }

            // 5) UNUSED pages: release them immediately so consumer warps can retire
            else if (laneid() >= PAGE_INPUT+1 && laneid() < SEM_COUNT)
            {
                int pg = s.pid(laneid());
                s.wait_page_ready(pg);
                arrive(s.page_finished[pg], config::NUM_CONSUMER_WARPS);
            }
        }
    };


    struct launcher
    { // launches mma's
        // launcher does nothing here, since this doesn't use tensor cores.
        static __device__ void run(const globals &g, state<config> &s)
        {
            s.wait_tensor_ready();
            if (laneid() == 0)
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
        }
    };


    struct consumer
    {
        static __device__ void run(const globals &g, state<config> &s)
        {
            //--------------------------------------------------
            // LOAD INPUT ACTIVATIONS
            //--------------------------------------------------
            rt_bf<16, 128> weights, gate_weights, broadcast_activations, gate_broadcast_activations;
            typename rt_bf<16, 128>::row_vec activations_vec;
            typename rt_bf<16, 128>::col_vec output_col_format, gate_output_col_format;
            rv_bf<16> output, gate_output;
            int group_id = warpgroup::groupid();
            int warp_id = warpgroup::warpid(); // id within the warpgroup

            // Next we need to load the activations
            wait(in_arrived(s), 0);
            // reinterpret the activations page as sv_bf<128>[16]
            int activation_page = get_input_page(s);
            sv_bf<128> (&activations_smem)[16] = reinterpret_cast<sv_bf<128>(&)[16]>(s.pages[activation_page]);
            warp::load(activations_vec, activations_smem[warpid()]);
            warp::sync();
            warp::arrive(s.page_finished[activation_page]); // just 1 is sufficient


            //--------------------------------------------------
            // UP MATVEC
            //--------------------------------------------------
            wait(up_arrived(s, group_id), 0);
            int weight_page = get_up_page(s, group_id);
            st_bf<16, 128> (&weights_smem)[4] = reinterpret_cast<st_bf<16, 128>(&)[4]>(s.pages[weight_page]);
            warp::load(weights, weights_smem[warp_id]);
            warp::sync();
            warp::arrive(s.page_finished[weight_page], config::NUM_CONSUMER_WARPS/4); // this is called by each warp in the warpgroup
            
            // broadcast this into a tile
            warp::broadcast_col(broadcast_activations, activations_vec);
            warp::mul(broadcast_activations, broadcast_activations, weights);
            warp::row_sum(output_col_format, broadcast_activations);
            warp::copy(output, output_col_format);
            warp::sync();


            //--------------------------------------------------
            // GATE MATVEC
            //--------------------------------------------------
            wait(gate_arrived(s, group_id), 0);
            int gate_weight_page = get_gate_page(s, group_id);
            st_bf<16, 128> (&gate_weights_smem)[4] = reinterpret_cast<st_bf<16, 128>(&)[4]>(s.pages[gate_weight_page]);
            warp::load(gate_weights, gate_weights_smem[warp_id]);
            warp::sync();
            warp::arrive(s.page_finished[gate_weight_page], config::NUM_CONSUMER_WARPS/4); // this is called by each warp in the warpgroup
            
            // broadcast this into a tile
            warp::broadcast_col(gate_broadcast_activations, activations_vec);
            warp::mul(gate_broadcast_activations, gate_broadcast_activations, gate_weights);
            warp::row_sum(gate_output_col_format, gate_broadcast_activations);
            warp::copy(gate_output, gate_output_col_format);
            warp::sync();

            float* scratch_f32 = (float*)s.scratch();
            // 1) accumulate partial sums from every consumer warp
            if (laneid() < 16) {
                atomicAdd(&scratch_f32[laneid()     ], float(output      [0][0]));   // up
                atomicAdd(&scratch_f32[laneid() + 16], float(gate_output [0][0]));   // gate
            }
            warp::sync();                             // all adds have landed
            warp::arrive(out_arrived(s));  // let the storer know we’re done 
        }
    };


    struct storer
    {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};

            if (laneid() == 0) {
                wait(out_arrived(s), 0);

                float* scratch_f32 = (float*)s.scratch();
                bf16*  scratch_bf16 = (bf16*)scratch_f32;    // alias
                /* fuse up * SiLU(gate) once, in float, then cast */
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    float up   = scratch_f32[i];
                    float gate = scratch_f32[i + 16];
                    float silu = gate / (1.f + expf(-gate));
                    scratch_bf16[i] = bf16(up * silu);
                }

                sv_bf<16>& vec = *reinterpret_cast<sv_bf<16>*>(scratch_bf16);
                tma::store_async(g.O, vec, {inst.start_col/16});
                tma::store_async_wait();
            }

            warp::sync();
            asm volatile("fence.acq_rel.gpu;");
            if (laneid() == 0) {
                if constexpr (OP_IDX == g.Bar.rows() - 1)
                    atomicAdd(&g.Bar[{inst.layer + 1, 0, 0}], 1);
                else
                    atomicAdd(&g.Bar[{inst.layer, OP_IDX + 1, 0}], 1);
            }
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(silu_mlp, m)
{
    m.doc() = "silu_mlp python module";
    kittens::py::bind_kernel<kvm<config, globals, SiLU_MLPOp<config>>>(
        m, "silu_mlp",
        &globals::instructions,
        &globals::timings,
        &globals::UP_PROJ_W,
        &globals::GATE_PROJ_W,
        &globals::INP,
        &globals::O,
        &globals::Bar
    );
    cudaGetLastError();
}
