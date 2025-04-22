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
    weights DOWN_PROJ_W;
    activations INP;
    activations O;
    // barriers Bar;
    
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
    static constexpr int DOWN_PAGES  = 4;
    static constexpr int PAGE_INPUT  = UP_PAGES + GATE_PAGES + DOWN_PAGES;    // = 12
    static constexpr int PAGE_OUTPUT = PAGE_INPUT + 1;                        // = 13
    static constexpr int SEM_COUNT   = PAGE_OUTPUT + 1;                       // = 14

    //  semaphores 
  __device__ static inline semaphore &up_arrived   (state<config> &s, int i) { return s.semaphores()[ i                  ]; }
  __device__ static inline semaphore &gate_arrived (state<config> &s, int i) { return s.semaphores()[ UP_PAGES + i        ]; }
  __device__ static inline semaphore &down_arrived (state<config> &s, int i) { return s.semaphores()[ UP_PAGES + GATE_PAGES + i ]; }
  __device__ static inline semaphore &in_arrived   (state<config> &s)        { return s.semaphores()[ PAGE_INPUT           ]; }
  __device__ static inline semaphore &out_arrived  (state<config> &s)        { return s.semaphores()[ PAGE_OUTPUT          ]; }

    // getters
    __device__ static inline int get_up_page  (state<config> &s, int i) { return s.pid(i); }
    __device__ static inline int get_gate_page(state<config> &s, int i) { return s.pid(UP_PAGES + i); }
    __device__ static inline int get_down_page(state<config> &s, int i) { return s.pid(UP_PAGES + GATE_PAGES + i); }
    __device__ static inline int get_input_page(state<config> &s) { return s.pid(PAGE_INPUT); }
    __device__ static inline int get_output_page(state<config> &s) { return s.pid(PAGE_OUTPUT); }

    // TODO: fix controller
    struct controller
    {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query)
        {
            int ret_order[] = {6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s)
        {
            // each weight page and the input page needs exactly 1 “ready” signal
            for (int i = 0; i < UP_PAGES;   i++) init_semaphore(up_arrived(s,i),   1);
            for (int i = 0; i < GATE_PAGES; i++) init_semaphore(gate_arrived(s,i), 1);
            for (int i = 0; i < DOWN_PAGES; i++) init_semaphore(down_arrived(s,i), 1);
            init_semaphore(in_arrived(s),   1);
            // output must wait for all 4 consumer warps
            init_semaphore(out_arrived(s),  config::NUM_CONSUMER_WARPS);
            // tell KVM: we set up SEM_COUNT semaphores
            return SEM_COUNT;
        }
    };


    struct loader
    {
        static __device__ void run(const globals &g, state<config> &s)
        {

            parsed_instruction inst{s};
            // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
            ((int *)s.scratch())[laneid()] = 0;
            warp::sync(); // done, now we can proceed to other things.
            
            // lanes 0..UP_PAGES-1 load UP_W tiles
            if(laneid()<UP_PAGES) {
                int page = get_up_page(s, laneid());
                s.wait_page_ready(page);
                tma::expect(up_arrived(s,laneid()), g.UP_PROJ_W);
                tma::load_async(reinterpret_cast<st_bf<16,512>&>(s.pages[page]),
                                g.UP_PROJ_W,
                                {inst.layer, inst.start_col/16, laneid()},
                                up_arrived(s,laneid()));
            }
            // lanes UP_PAGES..UP+GATE load GATE_W
            else if(laneid()<UP_PAGES+GATE_PAGES) {
                int idx = laneid()-UP_PAGES;
                int page = get_gate_page(s, idx);
                s.wait_page_ready(page);
                tma::expect(gate_arrived(s,idx), g.GATE_PROJ_W);
                tma::load_async(reinterpret_cast<st_bf<16,512>&>(s.pages[page]),
                                g.GATE_PROJ_W,
                                {inst.layer, inst.start_col/16, idx},
                                gate_arrived(s,idx));
            }
            // lanes UP+GATE..UP+GATE+DOWN load DOWN_W
            else if(laneid()<UP_PAGES+GATE_PAGES+DOWN_PAGES) {
                int idx = laneid()-(UP_PAGES+GATE_PAGES);
                int page = get_down_page(s, idx);
                s.wait_page_ready(page);
                tma::expect(down_arrived(s,idx), g.DOWN_PROJ_W);
                tma::load_async(reinterpret_cast<st_bf<16,512>&>(s.pages[page]),
                                g.DOWN_PROJ_W,
                                {inst.layer, inst.start_col/16, idx},
                                down_arrived(s,idx));
            }
            // lane for input vector
            else if(laneid()==UP_PAGES+GATE_PAGES+DOWN_PAGES) {
                int page = get_input_page(s);
                s.wait_page_ready(page);
                tma::expect(in_arrived(s), g.INP);
                tma::load_async(reinterpret_cast<sv_bf<2048>&>(s.pages[page]),
                                g.INP,
                                {},
                                in_arrived(s));
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
            int group_id = warpgroup::groupid();
            int warp_id = warpgroup::warpid(); // id within the warpgroup

            // 1) load input into register
            wait(in_arrived(s), 0);
            int in_page = get_input_page(s);
            sv_bf<128>(&in_smem)[16] =
                reinterpret_cast<sv_bf<128>(&)[16]>(s.pages[in_page]);
            typename rt_bf<16,128>::row_vec activations_vec;
            warp::load(activations_vec, in_smem[ warpgroup::warpid() ]);
            warp::sync();
            warp::arrive( s.page_finished[in_page] );


            // 2) UP matvec
            wait(up_arrived(s, group_id), 0);
            st_bf<16,128>(&up_w_smem)[4] =
                reinterpret_cast<st_bf<16,128>(&)[4]>(s.pages[ get_up_page(s,group_id) ]);
            rt_bf<16,128> up_w_reg;
            warp::load(up_w_reg, up_w_smem[ warp_id ]);
            warp::sync();

            rt_bf<16,128> broadcast;
            warp::broadcast_col(broadcast, activations_vec);
            warp::mul(broadcast, broadcast, up_w_reg);

            typename rt_bf<16,128>::col_vec up_col;
            rv_bf<16> up_vec;
            warp::row_sum(up_col, broadcast);
            warp::copy(up_vec, up_col);


            // 3) GATE matvec
            wait(gate_arrived(s, group_id), 0);
            st_bf<16,128>(&gate_w_smem)[4] =
                reinterpret_cast<st_bf<16,128>(&)[4]>(s.pages[ get_gate_page(s,group_id) ]);
            rt_bf<16,128> gate_w_reg;
            warp::load(gate_w_reg, gate_w_smem[ warp_id ]);
            warp::sync();

            // broadcast the same activations
            warp::broadcast_col(broadcast, activations_vec);
            warp::mul(broadcast, broadcast, gate_w_reg);

            typename rt_bf<16,128>::col_vec gate_col;
            rv_bf<16> gate_vec;
            warp::row_sum(gate_col, broadcast);
            warp::copy(gate_vec, gate_col);

           
            // 4) element‑wise SiLU on the gate result and fuse into up_vec
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    // extract the two bf16 lanes from broadcast.tiles[0][i].data[j]
                    __nv_bfloat16 xf = broadcast.tiles[0][i].data[j].x;
                    __nv_bfloat16 yf = broadcast.tiles[0][i].data[j].y;

                    // to float
                    float f = __bfloat162float(xf);
                    float g = __bfloat162float(yf);

                    // sigmoid
                    float sf = f / (1.0f + expf(-f));
                    float sg = g / (1.0f + expf(-g));

                    // write back SiLU = x * sigmoid(x)
                    broadcast.tiles[0][i].data[j].x = __float2bfloat16(f * sf);
                    broadcast.tiles[0][i].data[j].y = __float2bfloat16(g * sg);
                }
            }
            warp::sync();


            // 5) accumulate each lane’s fused up_vec[i] into scratch    
            // Now the first 16 threads have the output.
            if (laneid() < 16)
            { // this might be a bad idea but yolo, it's probably an okay start
                // and fortunately this is code where ncu will tell us if it's bad..
                atomicAdd(&((bf16 *)s.scratch())[laneid()], up_vec[0][0]);
            }
            warp::sync();
            warp::arrive(out_arrived(s));


           
            
        }
    };


    struct storer
    {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s)
        {
            parsed_instruction inst{s};
            if (laneid() == 0)
            {
                wait(out_arrived(s), 0);
                s.record(125);
                void *scratch = s.scratch();
                sv_bf<16> &output = *reinterpret_cast<sv_bf<16> *>(scratch);
                tma::store_async(g.O, output, {inst.start_col / 16});
                tma::store_async_wait(); // not just read wait! full wait! must be visible in global!
                s.record(126);
            }
            warp::sync();
            asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.
            if (laneid() == 0)
            {
                // if constexpr (OP_IDX == g.Bar.rows() - 1)
                //     atomicAdd(&g.Bar[{inst.layer + 1, 0, 0}], 1);
                // else
                //     atomicAdd(&g.Bar[{inst.layer, OP_IDX + 1, 0}], 1);
            }
            if (laneid() == 0)
                s.record(127);
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(silu_mlp, m)
{
    m.doc() = "silu_mlp python module";
    kittens::py::bind_kernel<kvm<config, globals, SiLU_MLPOp<config>>>(m, "silu_mlp",
                                                                     &globals::instructions,
                                                                     &globals::timings,
                                                                     &globals::UP_PROJ_W,
                                                                     &globals::DOWN_PROJ_W,
                                                                     &globals::GATE_PROJ_W,
                                                                     &globals::INP,
                                                                     &globals::O);
}
