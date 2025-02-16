#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

static constexpr int QK_D = 576, VO_D = 512, VO_Dd2 = VO_D/2, NUM_ROWS = 32, PAGE_SIZE = 256, ITERS_PER_PAGE = PAGE_SIZE / NUM_ROWS;
using q_tile              = st_bf<64, QK_D>;
using q_global            = kittens::gl<bf16, -1, -1, -1, QK_D>; // B * R * H * D_QK
using cache_tile          = st_bf<NUM_ROWS, QK_D>; using v_tile = st_bf<NUM_ROWS, VO_Dd2>; // we need the v_tile for later
using cache_global        = kittens::gl<bf16, 1, -1, PAGE_SIZE, QK_D, cache_tile>; // 1 * #page * pagesize * QK_D
using instructions_global = kittens::gl<int, 1, -1, -1, 8>;
using table_global        = kittens::gl<int, 1, 1, -1, -1>; // B * (max # pages)
using o_tile_d2           = st_bf<64, VO_Dd2>;
using o_tile_fl           = st_fl<16, VO_D>;
using o_global            = kittens::gl<bf16, -1, -1, -1, VO_D, o_tile_d2>; // B * R * H * D_VO
using o_scratch_global    = kittens::gl<float, 1, -1, 64, VO_D, o_tile_fl>; // For partial O's
using lvec_scratch_global = kittens::gl<float, 1, 1, -1, 64, sv_fl<16>>; // For partial O's
using semaphore_global    = kittens::gl<int, 1, 1, 1, -1>; // 1 * 1 * 1 * uid

struct config {
    struct globals {
        instructions_global instructions;
        q_global Q;
        cache_global Cache;
        table_global Table;
        o_global O;
        o_scratch_global O_scratch;
        lvec_scratch_global Lvec_scratch;
        semaphore_global semaphore;
        const float Softmax_scale;
        int dynamic_shared_memory() { return 226000; }
        dim3 grid()  { return dim3(132); } //dim3(Q.batch * ((Q.depth + 3) / 4)); }
        dim3 block() { return dim3((8+4)*WARP_THREADS); }
    };
};

struct location {
    int batch_idx; // batch_idx >=0
    int seq_idx;
};
struct partial_layout {
    using globals = config::globals;
    struct input_block { cache_tile c; };
    struct scratch_block { q_tile q; st_bf<64, cache_tile::rows> att_block; sv_fl<64> vec[2]; };
    struct common_state {
        int uid;
        location dst;
        int q_batch_idx;
        int q_seq_idx;
        int length;
    };
    struct consumer_state {
        col_vec<rt_fl<16, cache_tile::rows>> max_vec, norm_vec;
        col_vec<rt_fl<16, cache_tile::rows>> max_vec_last_scaled, max_vec_scaled;
        rt_fl<16, VO_Dd2> o;
    };
};
struct reduction_layout {
    using globals = config::globals;
    struct input_block { o_tile_fl o[2]; sv_fl<16> lvec[2]; sv_fl<16> padding[14]; };
    struct common_state {
        location dst;
        int src_uid[2];
    };
    struct consumer_state {
        col_vec<rt_fl<16, cache_tile::rows>> lvec[2];
        rt_fl<16, VO_D / 8> o[2];
    };
};
struct partial_template {
    using config = config;
    using layout = partial_layout;
    static constexpr int opcode = 1;
    static constexpr int INPUT_PIPE_STAGES = 2;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        args.common.uid = args.globals.instructions[kittens::coord<>{0, blockIdx.x, args.task_iter, 1}];
        args.common.dst = {args.globals.instructions[kittens::coord<>{0, blockIdx.x, args.task_iter, 2}], args.globals.instructions[kittens::coord<>{0, blockIdx.x, args.task_iter, 3}]};
        args.common.q_batch_idx = args.globals.instructions[kittens::coord<>{0, blockIdx.x, args.task_iter, 4}];
        args.common.q_seq_idx = args.globals.instructions[kittens::coord<>{0, blockIdx.x, args.task_iter, 5}];
        args.common.length = args.globals.instructions[kittens::coord<>{0, blockIdx.x, args.task_iter, 6}];
        args.num_iters = (args.common.length + NUM_ROWS - 1) / NUM_ROWS;
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                int global_load_idx = args.iter % ITERS_PER_PAGE;
                int next_page_id = args.globals.Table[coord<>{args.common.uid, args.iter / ITERS_PER_PAGE}];
                // next page we need to load?
                tma::expect(args.inputs_arrived, args.input.c);
                // cache shape is 1, # pages, page size, QK_D
                tma::load_async(args.input.c, args.globals.Cache, {0, next_page_id, global_load_idx, 0}, args.inputs_arrived);
            }
            else if(laneid() == 0) arrive(args.inputs_arrived);
            warpgroup::sync(5);
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            if(group<8>::warpid() == 0) {
                // auto q_st = subtile_inplace<16, QK_D>(args.scratch.q, {warpgroup::warpid(), 0});
                // int num_valid_Q_tokens = int(args.globals.Q.depth) * int(args.globals.Q.rows);
                // if ((4*args.common.q_seq_idx + warpgroup::warpid()) * q_st.rows < num_valid_Q_tokens) {
                //     // TODO: need to put in the partial load and store
                //     load(q_st, args.globals.Q, {args.common.q_batch_idx, 4*args.common.q_seq_idx + warpgroup::warpid(), 0});
                // }
                // else {
                //     zero(q_st);
                // }
                // TODO: the / 4 needs to change for different num heads
                load(args.scratch.q, args.globals.Q, {args.common.q_batch_idx, args.common.q_seq_idx, 0, 0});
                zero(args.scratch.vec[0]);
            }
            zero(args.state.norm_vec);
            neg_infty(args.state.max_vec);
            zero(args.state.o);
            group<8>::sync(10);
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            // 1.44269504089f is from exp2
            const float SOFTMAX_TEMPERATURE = args.globals.Softmax_scale * 1.44269504089f;
            int total_q_tokens = int(args.globals.Q.rows) / 16;
            int valid_q_tokens = total_q_tokens - args.common.q_seq_idx;
            if(warpgroup::groupid() == 0) {
                // A = Q @ K.T
                rt_fl<16, cache_tile::rows> att_block_fp32;
                warpgroup::mm_ABt(att_block_fp32, args.scratch.q, args.input.c);
                mul(args.state.max_vec_last_scaled, args.state.max_vec, SOFTMAX_TEMPERATURE);
                warpgroup::mma_async_wait();
                // // softmax
                // if(args.iter == args.num_iters-1) { // need to mask out a bunch of entries in the last page
                //     right_fill(att_block_fp32, att_block_fp32, args.common.length - args.iter*NUM_ROWS, base_types::constants<float>::neg_infty());
                // }
                // if(args.iter >= args.num_iters-2) { // Need to (possibly) do a causal mask.
                //     warpgroup::tril(att_block_fp32, att_block_fp32, args.common.length - args.iter*NUM_ROWS - args.globals.Q.depth, base_types::constants<float>::neg_infty());
                // }
                row_max(args.state.max_vec, att_block_fp32, args.state.max_vec); // accumulate onto the max_vec
                mul(args.state.max_vec_scaled, args.state.max_vec, SOFTMAX_TEMPERATURE);
                mul(att_block_fp32, att_block_fp32, SOFTMAX_TEMPERATURE);
                sub_row(att_block_fp32, att_block_fp32, args.state.max_vec_scaled);
                exp2(att_block_fp32, att_block_fp32);
                sub(args.state.max_vec_last_scaled, args.state.max_vec_last_scaled, args.state.max_vec_scaled);
                exp2(args.state.max_vec_last_scaled, args.state.max_vec_last_scaled);
                warpgroup::store(args.scratch.vec[0], args.state.max_vec_last_scaled);
                mul(args.state.norm_vec, args.state.norm_vec, args.state.max_vec_last_scaled);
                row_sum(args.state.norm_vec, att_block_fp32, args.state.norm_vec); // accumulate onto the norm_vec
                warpgroup::store(args.scratch.att_block, att_block_fp32);
            }
            group<8>::sync(10);
            warpgroup::load(args.state.max_vec_last_scaled, args.scratch.vec[0]);

            mul_row(args.state.o, args.state.o, args.state.max_vec_last_scaled); // normalize o_reg before mma
            // O += A @ V
            auto (&v_smem)[2] = reinterpret_cast<v_tile(&)[2]>(args.input.c);
            warpgroup::mma_AB(args.state.o, args.scratch.att_block, v_smem[warpgroup::groupid()]);
            warpgroup::mma_async_wait();
            if(warpgroup::laneid() == 0) arrive(args.inputs_finished, WARPGROUP_WARPS); // done!
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if(warpgroup::groupid() == 0) {
                warpgroup::store(args.scratch.vec[1], args.state.norm_vec);
            }
            group<8>::sync(10);
            warpgroup::load(args.state.norm_vec, args.scratch.vec[1]);
            div_row(args.state.o, args.state.o, args.state.norm_vec);
            if(args.common.dst.batch_idx >= 0) { // batch is meaningful
                auto (&o_smem)[2] = reinterpret_cast<o_tile_d2(&)[2]>(args.scratch.q);
                warpgroup::store(o_smem[warpgroup::groupid()], args.state.o);
                warpgroup::sync(warpgroup::groupid());
                // auto o_st = subtile_inplace<16, VO_Dd2>(o_smem[warpgroup::groupid()], {warpgroup::warpid(), 0});
                if (warpgroup::warpid() == 0) {
                    store(args.globals.O, o_smem[warpgroup::groupid()], {args.common.dst.batch_idx, args.common.dst.seq_idx, 0, warpgroup::groupid()});
                }
                // if ((4*args.common.dst.seq_idx + warpgroup::warpid() % 4) * o_st.rows < num_valid_O_tokens) {
                //     // todo: need the partial store
                //     store(args.globals.O, o_st, {args.common.dst.batch_idx, 4*args.common.dst.seq_idx + warpgroup::warpid() % 4, 0, warpgroup::groupid()});
                // }
            }
            else { // write out directly to O scratch, without going through smem
                warpgroup::store(args.globals.O_scratch, args.state.o, {args.common.dst.seq_idx, 0, 0});
            }
            warpgroup::sync(warpgroup::groupid());
            if(args.common.dst.batch_idx < 0) {
                if(group<8>::laneid() == 0) {
                    args.globals.semaphore[{args.common.dst.seq_idx}] = 1;
                }
            }
            if(warpgroup::laneid() == 0) arrive(args.finish_finished, WARPGROUP_WARPS); // done!
        }
    };
};
struct reduction_template {
    using config = config;
    using layout = reduction_layout;
    static constexpr int opcode = 2;
    static constexpr int INPUT_PIPE_STAGES = 2;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        args.common.dst = {args.globals.instructions[kittens::coord<>{0, blockIdx.x, args.task_iter, 1}], args.globals.instructions[kittens::coord<>{0, blockIdx.x, args.task_iter, 2}]};
        args.common.src_uid[0] = args.globals.instructions[kittens::coord<>{0, blockIdx.x, args.task_iter, 3}];
        args.common.src_uid[1] = args.globals.instructions[kittens::coord<>{0, blockIdx.x, args.task_iter, 4}];
        args.num_iters = 4;
        // If we are doing a reduction, we need to spinloop until we have confirmation that all the partial results have been written out.
        if(threadIdx.x == 0) { // easier to have a single thread spin
            while(*(volatile int*)&args.globals.semaphore[{args.common.src_uid[0]}] == 0) {} // note volatile, L1 is not guaranteed to be coherent.
            while(*(volatile int*)&args.globals.semaphore[{args.common.src_uid[1]}] == 0) {}
        }
        group<12>::sync(11); // all warps must sync here.
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                // next page we need to load?
                tma::expect(args.inputs_arrived, args.input.o[0], args.input.o[1], args.input.lvec[0], args.input.lvec[1]);
                tma::load_async(args.input.o[0], args.globals.O_scratch, {args.common.src_uid[0], args.iter, 0}, args.inputs_arrived);
                tma::load_async(args.input.o[1], args.globals.O_scratch, {args.common.src_uid[1], args.iter, 0}, args.inputs_arrived);
                tma::load_async(args.input.lvec[0], args.globals.Lvec_scratch, {args.common.src_uid[0], args.iter}, args.inputs_arrived);
                tma::load_async(args.input.lvec[1], args.globals.Lvec_scratch, {args.common.src_uid[1], args.iter}, args.inputs_arrived);
            }
            else if(laneid() == 0) arrive(args.inputs_arrived);
            warpgroup::sync(5);
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            // 1.44269504089f is from exp2
            const float SOFTMAX_TEMPERATURE = args.globals.Softmax_scale * 1.44269504089f;
            auto o1_st = subtile_inplace<16, VO_D/8>(args.input.o[0], {0, group<8>::warpid()});
            auto o2_st = subtile_inplace<16, VO_D/8>(args.input.o[1], {0, group<8>::warpid()});
            load(args.state.o[0], o1_st);
            load(args.state.o[1], o2_st);
            load(args.state.lvec[0], args.input.lvec[0]);
            load(args.state.lvec[1], args.input.lvec[1]);
            if(warpgroup::laneid() == 0) arrive(args.inputs_finished, WARPGROUP_WARPS); // done!
            col_vec<rt_fl<16, cache_tile::rows>> max_lvec, sum_lvec;
            max(max_lvec, args.state.lvec[0], args.state.lvec[1]);
            sub(args.state.lvec[0], args.state.lvec[0], max_lvec);
            sub(args.state.lvec[1], args.state.lvec[1], max_lvec);
            exp2(args.state.lvec[0], args.state.lvec[0]);
            exp2(args.state.lvec[1], args.state.lvec[1]);
            add(sum_lvec, args.state.lvec[0], args.state.lvec[1]);
            div(args.state.lvec[0], args.state.lvec[0], sum_lvec);
            div(args.state.lvec[1], args.state.lvec[1], sum_lvec);
            mul_row(args.state.o[0], args.state.o[0], args.state.lvec[0]);
            mul_row(args.state.o[1], args.state.o[1], args.state.lvec[1]);
            add(args.state.o[0], args.state.o[0], args.state.o[1]);
            log2(sum_lvec, sum_lvec);
            add(sum_lvec, sum_lvec, max_lvec);
            if(args.common.dst.batch_idx >= 0) {
                store(args.globals.O, args.state.o[0],
                    {args.common.dst.batch_idx, 4*args.common.dst.seq_idx + args.iter, 0, group<8>::warpid()});
            }
            else {
                // if (threadIdx.x == 0) {
                //     printf("store to O_scratch, 254: %d, %d\n", args.common.dst.seq_idx, group<8>::warpid());
                // }
                store(args.globals.O_scratch, args.state.o[0], {args.common.dst.seq_idx, 0, group<8>::warpid()});
                if(warpgroup::warpid() == 0) {
                    store(args.globals.Lvec_scratch, sum_lvec, {args.common.dst.seq_idx, 0});
                }
            }
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            // Increment the semaphore for the next stage, if this is not the last one.
            warpgroup::sync(warpgroup::groupid()); // Make sure memory has been flushed to global memory.
            if(args.common.dst.batch_idx < 0) {
                if(group<8>::laneid() == 0) {
                    args.globals.semaphore[{args.common.dst.seq_idx}] = 1;
                }
                __syncwarp();
            }
            if(warpgroup::laneid() == 0) arrive(args.finish_finished, WARPGROUP_WARPS); // done!
        }
    };
};

PYBIND11_MODULE(mla_decode, m) {
    m.doc() = "mla_decode python module";
    py::bind_kernel<interpreter::kernel<config, partial_template>>(m, "mla_decode", //, reduction_template
        &config::globals::instructions,
        &config::globals::Q,
        &config::globals::Cache,
        &config::globals::Table,
        &config::globals::O,
        &config::globals::O_scratch,
        &config::globals::Lvec_scratch,
        &config::globals::semaphore,
        &config::globals::Softmax_scale
    );
}