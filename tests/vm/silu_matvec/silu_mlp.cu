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
    static constexpr int opcode = 3;
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
            
            if ( laneid() == 0 && warpgroup::warpid() == 0 ) { 
                printf("Inside controller release lid!\n");
            }
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s)
        {

            if ( laneid() == 0 && warpgroup::warpid() == 0 ) { 
                printf("Inside controller init semaphores!\n");
            }

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

            if ( laneid() == 0 && warpgroup::warpid() == 0 ) { 
                printf("Inside loader run!\n");
            }

            parsed_instruction inst{s};
            // clear scratch buffer
            ((int*)s.scratch())[laneid()] = 0;
            warp::sync();

            // 1) UP projections
            if (laneid() < UP_PAGES)
            {
                int pg = get_up_page(s, laneid());
                s.wait_page_ready(pg);
                s.record(16 + laneid());
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
                s.record(16 + laneid());
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
                s.record(24);
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
            // printf("launcher at %d %d\n", laneid(), warpid());
            s.wait_tensor_ready();
            if (laneid() == 0)
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
        }
    };


    struct consumer
    {
        static __device__ void run(const globals &g, state<config> &s)
        {
            int group  = warpgroup::groupid();     // which weight‐page group
            int warpid = warpgroup::warpid();      // which “lane‐block”
            int lid    = laneid();                 // 0–31

            if ( laneid() == 0 && warpid == 0 ) { 
                printf("Inside consumer run!\n");
            }

            //--------------------------------------------------
            // 1) LOAD INPUT ACTIVATIONS
            //--------------------------------------------------
            wait(in_arrived(s), 0);
            int in_pg = get_input_page(s);
            typename rt_bf<16, 128>::row_vec x_vec;
            // copy the 16×128bfslice out of shared pages
            sv_bf<128>(&in_smem)[16] = reinterpret_cast<sv_bf<128>(&)[16]>(s.pages[in_pg]);
            warp::load(x_vec, in_smem[warp::warpid()]);
            warp::sync();
            warp::arrive(s.page_finished[in_pg]); // just 1 is sufficient


            //--------------------------------------------------
            // 2) UP PROJECTION
            //--------------------------------------------------
            wait(up_arrived(s, group), 0);
            int up_pg = get_up_page(s, group);
            st_bf<16,128> up_smem[4];
            memcpy(up_smem, &s.pages[up_pg], sizeof(up_smem));
            rt_bf<16,128> up_reg;
            warp::load(up_reg, up_smem[warpid]);
            warp::sync();
            warp::arrive(s.page_finished[up_pg], config::NUM_CONSUMER_WARPS);

            // broadcast & mul
            rt_bf<16,128> acc;
            warp::broadcast_col(acc, x_vec);
            warp::mul(acc, acc, up_reg);


            //--------------------------------------------------
            // 3) GATE PROJECTION
            //--------------------------------------------------
            wait(gate_arrived(s, group), 0);
            int gate_pg = get_gate_page(s, group);
            st_bf<16,128> gate_smem[4];
            memcpy(gate_smem, &s.pages[gate_pg], sizeof(gate_smem));
            rt_bf<16,128> gate_reg;
            warp::load(gate_reg, gate_smem[warpid]);
            warp::sync();
            warp::arrive(s.page_finished[gate_pg], config::NUM_CONSUMER_WARPS);

            // mul in place
            warp::mul(acc, acc, gate_reg);


            //--------------------------------------------------
            // 4) SiLU FUSION (in‑place on acc)
            //--------------------------------------------------
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    auto & d = acc.tiles[0][i].data[j];
                    float  f0 = __bfloat162float(d.x);
                    float  f1 = __bfloat162float(d.y);
                    float  s0 = f0/(1+expf(-f0));
                    float  s1 = f1/(1+expf(-f1));
                    d.x = __float2bfloat16(f0 * s0);
                    d.y = __float2bfloat16(f1 * s1);
                }
            }
            warp::sync();

            //--------------------------------------------------
            // 5) ROW‐SUM → out_vec (16 lanes)
            //--------------------------------------------------
            rt_bf<16,128>::col_vec col;
            rv_bf<16> out_vec;
            warp::row_sum(col, acc);
            warp::copy(out_vec, col);
            warp::sync();

            // --------------------------------------------------
            // 6) ATOMIC ADD EACH LANE INTO SCRATCH
            // --------------------------------------------------
            // Now the first 16 threads have the output.
            if (laneid() < 16)
            { // this might be a bad idea but yolo, it's probably an okay start
                // and fortunately this is code where ncu will tell us if it's bad..
                // atomicAdd(&((bf16 *)s.scratch())[laneid()], out_vec[0][0]);
            }
            warp::sync();
            warp::arrive(out_arrived(s));
        }
    };


    struct storer
    {
        static __device__ void run(const globals &g, state<config> &s) {
            
            if ( laneid() == 0 && warpgroup::warpid() == 0 ) { 
                printf("Inside storer run!\n");
            }

            parsed_instruction inst{s};

            if (laneid() == 0) {
                // wait for all consumer warps
                wait(out_arrived(s), 0);
                // read back the float sums
                void *scratch = s.scratch();
                // now treat that flat array as an sv_bf<16> tile
                sv_bf<16> &output = *reinterpret_cast<sv_bf<16> *>(scratch);
                tma::store_async(g.O, output, { inst.start_col/16 });
                tma::store_async_wait();
            }

            warp::sync();
            asm volatile("fence.acq_rel.gpu;\n");
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
