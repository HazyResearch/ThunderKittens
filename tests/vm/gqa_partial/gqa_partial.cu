#include <iostream>

#include "kittens.cuh"
#include "vm/vm.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

/*
    Instruction as defined on Python-side:

        class PartialAttention(Instruction):
            layer_idx: int
            kv_head_idx: int
            num_partials: int
            partial_idx: int

    Instruction format on CUDA:

        [0] = opcode (1)
        [1] = layer_idx
        [2] = kv_head_idx
        [3] = num_partials
        [4] = partial_idx
*/

constexpr int NUM_BLOCKS = 148;
constexpr int GQA_PARTIAL_OPCODE = 1;

using config = default_config;
struct globals {
    using q_rt = rt_bf<32, 64>;
    using kv_rt = rt_bf<32, 64>;
    using o_rt = rt_fl<32, 32>;
    using q_st = st_bf<32, 64>; // <= 16KB
    using kv_st = st_bf<32, 64>;
    using o_st = st_bf<32, 32>;
    using q_layout = gl<bf16, 1, -1, -1, -1, q_st>; // assume single batch
    using kv_layout = gl<bf16, 1, -1, -1, -1, kv_st>;
    using o_layout = gl<bf16, 1, -1, -1, -1, o_st>; // assume single batch
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    instruction_layout instructions;
    timing_layout timings;
    q_layout Q;
    kv_layout K_c, V_c;
    o_layout O;
    dim3 grid() { return dim3(NUM_BLOCKS); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct gqa_partial_op {
    static constexpr int opcode = GQA_PARTIAL_OPCODE;
    static constexpr int PIPELINE_STAGES = 3;

    __device__ static inline semaphore &inputs_arrived(state<config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline semaphore &outputs_arrived(state<config> &s) {
        return s.semaphores()[1];
    }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            if (!blockIdx.x && !laneid()) {
                printf("NUM_PAGES: %d\n", config::NUM_PAGES);
            }

            int ret_order[13] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            init_semaphore(inputs_arrived(s), 0, 1);
            init_semaphore(outputs_arrived(s), 0, 1);
            return 2;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            // parsed_instruction inst = parse_instruction(g, s);

            /* Initial of of Qi, Kj, Vj */
            if (laneid() == 0) {
                int Q_page_pid = s.pid(0);
                int K_page_pid = s.pid(1);
                int V_page_pid = s.pid(2);
                s.wait_page_ready(Q_page_pid);
                s.wait_page_ready(K_page_pid);
                s.wait_page_ready(V_page_pid);
                globals::q_st  &Q = *reinterpret_cast<globals::q_st*>(s.pages[Q_page_pid].data); // exactly 1 page (16KB)
                globals::kv_st &K = *reinterpret_cast<globals::kv_st*>(s.pages[K_page_pid].data);
                globals::kv_st &V = *reinterpret_cast<globals::kv_st*>(s.pages[V_page_pid].data);
                tma::expect_bytes(inputs_arrived(s), sizeof(Q) + sizeof(K) + sizeof(V));
                tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(Q, g.Q, {0, 0, 0, 0}, inputs_arrived(s));
                tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(K, g.K_c, {0, 0, 0, 0}, inputs_arrived(s));
                tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(V, g.V_c, {0, 0, 0, 0}, inputs_arrived(s));
            }
            // warp::sync();




            // if(args.iter == 1) group<12>::sync(11); // wait for the consumer to finish its setup, before we do the second load.
            // if(warpgroup::warpid() == 0) {
            //     int pos = args.common.start_pos + NUM_ROWS*args.iter;
            //     int within_page_idx = (pos % PAGE_SIZE) / NUM_ROWS;
            //     int next_page_id = args.globals.Table[coord<>{args.common.q_batch_idx, pos/PAGE_SIZE}];
            //     // next page we need to load?
            //     tma::expect(args.inputs_arrived, args.input.kcache, args.input.vcache);
            //     tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(args.input.kcache, args.globals.K_cache, {0, next_page_id, within_page_idx, 0}, args.inputs_arrived);
            //     tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(args.input.vcache, args.globals.V_cache, {0, next_page_id, within_page_idx, 0}, args.inputs_arrived);
            //     if(laneid() == 0) arrive(args.inputs_arrived, 3);
            // }
            // warpgroup::sync(5);

            // uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

            // int pipeline_stage = 0;
            // for(int i = 0; i < inst.iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
            //     wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
            //     warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), 128*128*4);
            //     #pragma unroll
            //     for(int j = 0; j < 2; j++) {
            //         int Q_page = get_a_page(s, pipeline_stage, j);
            //         if(i < PIPELINE_STAGES) {
            //             s.wait_page_ready(Q_page);
            //         }
            //         st_fp8e4m3<128, 128> &a = s.pages[a_page].template as_st<fp8e4m3>();
            //         warp::tma::load_async(a, g.A, {inst.row+j, i}, inputs_arrived(s, pipeline_stage));
            //     }
            //     #pragma unroll
            //     for(int j = 0; j < 2; j++) {
            //         int b_page = get_b_page(s, pipeline_stage, j);
            //         if(i < PIPELINE_STAGES) {
            //             s.wait_page_ready(b_page);
            //         }
            //         st_fp8e4m3<128, 128> &b = s.pages[b_page].template as_st<fp8e4m3>();
            //         warp::tma::load_async(b, g.B, {inst.col+j, i}, inputs_arrived(s, pipeline_stage));
            //     }
            //     update_phasebit<1>(semaphore_bitfield, pipeline_stage);
            // }
            // warp::sync();
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) { 
            if (!laneid()) {
                const char *msg = "launcher";
                printf("Block %d, Warp ID %d, Lane ID %d, opcode: %d: %s\n", blockIdx.x, warpid(), laneid(), s.instruction()[0], msg);

                col_vec<rt_fl<16, 32>> max_vec, norm_vec;
                printf("%d %d\n", max_vec.outer_dim, max_vec.inner_dim);

            }

        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            if (warpid() == 0) {
                // Wait for the inputs to be ready
                wait(inputs_arrived(s), 0);
    
                // Move from smem to registers
                int Q_page_pid = s.pid(0);
                int K_page_pid = s.pid(1);
                int V_page_pid = s.pid(2);
                int O_page_pid = s.pid(3);
                s.wait_page_ready(O_page_pid);
                globals::q_st  &Q_smem = *reinterpret_cast<globals::q_st*>(s.pages[Q_page_pid].data);
                globals::kv_st &K_smem = *reinterpret_cast<globals::kv_st*>(s.pages[K_page_pid].data);
                globals::kv_st &V_smem = *reinterpret_cast<globals::kv_st*>(s.pages[V_page_pid].data);
                globals::o_st  &O_smem = *reinterpret_cast<globals::o_st*>(s.pages[O_page_pid].data);
                globals::q_rt Q_reg;
                globals::kv_rt K_reg, V_reg;
                globals::o_rt att_block;
                warp::zero(att_block);

                warp::load(Q_reg, Q_smem);
                warp::load(K_reg, K_smem);
                warp::load(V_reg, V_smem);

                warp::mma_ABt(att_block, Q_reg, K_reg, att_block); // Q@K.T
                warp::store(O_smem, att_block);
                warp::sync();
                warp::arrive(outputs_arrived(s));

                warp::arrive(s.page_finished[Q_page_pid], config::NUM_CONSUMER_WARPS);
                warp::arrive(s.page_finished[K_page_pid], config::NUM_CONSUMER_WARPS);
                warp::arrive(s.page_finished[V_page_pid], config::NUM_CONSUMER_WARPS);

                // if (laneid() == 0) {
                //     printf("K batch: %d\n", g.K_c.batch());
                //     printf("K head: %d\n", g.K_c.depth());
                //     printf("K row: %d\n", g.K_c.rows());
                //     printf("K col: %d\n", g.K_c.cols());

                //     float Q_smem_sum = 0;
                //     float Q_g_sum = 0;
                //     float Q_reg_sum = 0;
                //     for (int i = 0; i < 64 * 32; i++) {
                //         Q_g_sum += (float)g.Q.raw_ptr[i];
                //         Q_smem_sum += (float)Q_smem.data[i];
                //     }
                //     for (int i = 0; i < Q_reg.height; ++i) {
                //         for (int j = 0; j < Q_reg.width; ++j) {
                //             for (int k = 0; k < 128; ++k) {
                //                 Q_reg_sum += (float)*reinterpret_cast<bf16*>(&Q_reg.tiles[i][j].data[k]);
                //                 Q_reg_sum += (float)*(reinterpret_cast<bf16*>(&Q_reg.tiles[i][j].data[k]) + 1);
                //             }
                //         }
                //     }
                //     printf("Q_g_sum: %f\n", (float)(Q_g_sum));
                //     printf("Q_smem_sum: %f\n", (float)(Q_smem_sum));
                //     printf("Q_reg_sum: %f\n", (float)(Q_reg_sum));
                    
                //     float K_smem_sum = 0;
                //     float K_g_sum = 0;
                //     for (int i = 0; i < 64 * 32; i++) {
                //         K_g_sum += (float)g.K_c.raw_ptr[i];
                //         K_smem_sum += (float)K_smem.data[i];
                //     }
                //     printf("K_g_sum: %f\n", (float)(K_g_sum));
                //     printf("K_smem_sum: %f\n", (float)(K_smem_sum));
    
                //     float V_smem_sum = 0;
                //     float V_g_sum = 0;
                //     for (int i = 0; i < 64 * 32; i++) {
                //         V_g_sum += (float)g.V_c.raw_ptr[i];
                //         V_smem_sum += (float)V_smem.data[i];
                //     }
                //     printf("V_g_sum: %f\n", (float)(V_g_sum));
                //     printf("V_smem_sum: %f\n", (float)(V_smem_sum));
                // }
            }
        }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            if (laneid() == 0) {
                wait(outputs_arrived(s), 0);
                int O_page_pid = s.pid(3);
                globals::o_st &O_smem = *reinterpret_cast<globals::o_st*>(s.pages[O_page_pid].data);
    
                tma::store_async(g.O, O_smem, {0, 0, 0, 0});
                tma::store_async_read_wait();
                arrive(s.page_finished[O_page_pid], config::NUM_CONSUMER_WARPS);
            }
            warp::sync();
         }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(gqa_partial, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel<config, globals, gqa_partial_op<config>>>(m, "gqa_partial",
        &globals::instructions,
        &globals::timings,
        &globals::Q,
        &globals::K_c,
        &globals::V_c,
        &globals::O
    );
}
