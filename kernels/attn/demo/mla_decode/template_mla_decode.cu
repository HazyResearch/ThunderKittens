#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

static constexpr int QK_D = 576, VO_D = 512, VO_Dd2 = VO_D/2, NUM_ROWS = 32, PAGE_SIZE = 256, ITERS_PER_PAGE = PAGE_SIZE / NUM_ROWS;
using q_tile              = st_bf<64, QK_D>;
using q_global            = kittens::gl<bf16, -1, -1, -1, QK_D, q_tile>; // B * R * H * D_QK
using cache_tile          = st_bf<NUM_ROWS, QK_D>; using v_tile = st_bf<NUM_ROWS, VO_Dd2>; // we need the v_tile for later
using cache_global        = kittens::gl<bf16, 1, -1, PAGE_SIZE, QK_D, cache_tile>; // 1 * #page * pagesize * QK_D
using instructions_global = kittens::gl<int, 1, -1, -1, 8>;
using table_global        = kittens::gl<int, 1, 1, -1, -1>; // B * (max # pages)
using o_tile              = st_bf<64, VO_D>;
using o_tile_d2           = st_bf<64, VO_Dd2>;
using o_tile_fl           = st_fl<16, VO_D>;
using o_global            = kittens::gl<bf16, -1, -1, -1, VO_D, o_tile_d2, st_bf<16, VO_D>>; // B * R * H * D_VO

// using o_scratch_global    = kittens::gl<float, 1, -1, 64, VO_D, o_tile_fl>; // For partial O's
// using lvec_scratch_global = kittens::gl<float, 1,  1, -1,   64, sv_fl<16>>; // For partial O's
using o_scratch_global    = kittens::gl<float, -1, -1, 16, VO_D, o_tile_fl>; // For partial O's
using lvec_scratch_global = kittens::gl<float,  1, -1, -1, 16, sv_fl<16>>; // For partial O's

using semaphore_global    = kittens::gl<int,   1,  1,  1, -1>;              // 1 * 1 * 1 * uid

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
    int batch_idx; // batch_idx >=0, otherwise it's the negative index into scratch
    int seq_idx;
};
struct partial_layout {
    using globals = config::globals;
    struct input_block { cache_tile c; };
    struct scratch_block { q_tile q; st_bf<64, cache_tile::rows> att_block; sv_fl<64> max_vec, norm_vec; };
    struct common_state {
        int uid;
        location dst;
        int q_batch_idx;
        int q_seq_idx;
        int length;
    };
    struct consumer_state {
        col_vec<rt_fl<16, cache_tile::rows>> max_vec, norm_vec;
        rt_fl<16, VO_Dd2> o;
    };
};
struct partial_template {
    using config = config;
    using layout = partial_layout;
    static constexpr int opcode = 1;
    static constexpr int INPUT_PIPE_STAGES = 2;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        args.common.uid         =  args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 1}];
        args.common.dst         = {args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 2}],
                                   args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 3}]};
        args.common.q_batch_idx =  args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 4}];
        args.common.q_seq_idx   =  args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 5}];
        args.common.length      =  args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 6}];
        args.num_iters          = (args.common.length + NUM_ROWS - 1) / NUM_ROWS;
        args.common.length -= (args.globals.Q.depth - (args.common.q_seq_idx + warpgroup::warpid()) - 1); // adjust for the causal mask
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
            zero(args.state.norm_vec);
            neg_infty(args.state.max_vec);
            zero(args.state.o);
            auto q_st = subtile_inplace<16, QK_D/2>(args.scratch.q, {warpgroup::warpid(), warpgroup::groupid()});
            load_async(q_st, args.globals.Q, {args.common.q_batch_idx, args.common.q_seq_idx + warpgroup::warpid(), 0, warpgroup::groupid()});
            load_async_wait();
            group<8>::sync(10);
        }
        template<bool do_right_fill> __device__ static inline void internal_compute(consumer_compute_args<layout> args) {
            // 1.44269504089f is from exp2
            const float SOFTMAX_TEMPERATURE = args.globals.Softmax_scale * 1.44269504089f;

            col_vec<rt_fl<16, cache_tile::rows>> local_max_vec, local_norm_vec;
            col_vec<rt_fl<16, cache_tile::rows>> max_vec_last_scaled, max_vec_scaled;

            copy(local_max_vec,  args.state.max_vec);
            copy(local_norm_vec, args.state.norm_vec);

            if(warpgroup::groupid() == 0) {
                // A = Q @ K.T
                rt_fl<16, cache_tile::rows> att_block_fp32;
                warpgroup::mm_ABt(att_block_fp32, args.scratch.q, args.input.c);

                mul(max_vec_last_scaled, local_max_vec, SOFTMAX_TEMPERATURE);

                warpgroup::mma_async_wait();
                // softmax
                if constexpr (do_right_fill) { // need to mask out a bunch of entries in the last page
                    const int length = args.common.length - args.iter*NUM_ROWS;
                    right_fill(att_block_fp32, att_block_fp32, length, base_types::constants<float>::neg_infty());
                }

                row_max(local_max_vec, att_block_fp32, local_max_vec);
                mul(max_vec_scaled, local_max_vec, SOFTMAX_TEMPERATURE);

                mul(att_block_fp32, att_block_fp32, SOFTMAX_TEMPERATURE);
                sub_row(att_block_fp32, att_block_fp32, max_vec_scaled);
                
                exp2(att_block_fp32, att_block_fp32);
                
                sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
                exp2(max_vec_last_scaled, max_vec_last_scaled);
                warpgroup::store(args.scratch.max_vec, max_vec_last_scaled);
                
                mul(local_norm_vec, local_norm_vec, max_vec_last_scaled);
                row_sum(local_norm_vec, att_block_fp32, local_norm_vec);
                warpgroup::store(args.scratch.att_block, att_block_fp32);
            }
            group<8>::sync(10);

            warpgroup::load(max_vec_last_scaled, args.scratch.max_vec);
            mul_row(args.state.o, args.state.o, max_vec_last_scaled); // normalize o_reg before mma

            // O += A @ V
            auto (&v_smem)[2] = reinterpret_cast<v_tile(&)[2]>(args.input.c);
            warpgroup::mma_AB(args.state.o, args.scratch.att_block, v_smem[warpgroup::groupid()]);

            copy(args.state.max_vec, local_max_vec);
            copy(args.state.norm_vec, local_norm_vec);

            warpgroup::mma_async_wait();
            if(warpgroup::laneid() == 0) arrive(args.inputs_finished, WARPGROUP_WARPS); // done!
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            if(args.iter >= args.num_iters-2) internal_compute<true>(args);
            else internal_compute<false>(args);
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            col_vec<rt_fl<16, cache_tile::rows>> local_max_vec, local_norm_vec;

            copy(local_norm_vec, args.state.norm_vec);
            copy(local_max_vec, args.state.max_vec);

            if (warpgroup::groupid() == 0) warpgroup::store(args.scratch.norm_vec, local_norm_vec);
            group<8>::sync(10);
            if(warpgroup::groupid() == 1) warpgroup::load(local_norm_vec, args.scratch.norm_vec);
            div_row(args.state.o, args.state.o, local_norm_vec);

            if(args.common.dst.batch_idx >= 0) { // batch is meaningful
                auto &o_smem = reinterpret_cast<o_tile&>(args.scratch.q);
                auto o_st = subtile_inplace<16, VO_Dd2>(o_smem, {warpgroup::warpid(), warpgroup::groupid()});
                store(o_st, args.state.o);
                __syncwarp();
                store(args.globals.O, o_st, {args.common.dst.batch_idx, args.common.dst.seq_idx+warpgroup::warpid(), 0, warpgroup::groupid()});
            }
            else { // write out directly to O scratch, without going through smem
                if(warpgroup::groupid() == 0) {
                    mul(local_max_vec, local_max_vec, args.globals.Softmax_scale * 1.44269504089f);
                    log2(local_norm_vec, local_norm_vec);
                    add(local_norm_vec, local_norm_vec, local_max_vec); // l_vec = log2(norm_vec) + max_vec
                    store(args.globals.Lvec_scratch, local_norm_vec, {args.common.dst.seq_idx, warpgroup::warpid(), 0});
                }

                store(args.globals.O_scratch, args.state.o, {args.common.dst.seq_idx, warpgroup::warpid(), 0, warpgroup::groupid()});
            }
            group<8>::sync(10);
            if(args.common.dst.batch_idx < 0) {
                asm volatile("fence.sc.sys;");
                if(group<8>::laneid() == 0) {
                    args.globals.semaphore[{args.common.dst.seq_idx}] = 1;
                }
            }
            if(warpgroup::laneid() == 0) arrive(args.finish_finished, WARPGROUP_WARPS); // done!
        }
    };
};
struct reduction_layout {
    using globals = config::globals;
    struct input_block  { o_tile_fl o[2]; sv_fl<16> lvec[2]; sv_fl<16> padding[14]; };
    struct output_block { o_tile_fl o; sv_fl<16> lvec; sv_fl<16> padding[15]; };
    struct common_state {
        int uid;
        location dst;
        int src_uid[2];
        int src_batch[2];
    };
    struct consumer_state {};
};
struct reduction_template {
    using config = config;
    using layout = reduction_layout;
    static constexpr int opcode = 2;
    static constexpr int INPUT_PIPE_STAGES = 2;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        args.common.uid           = args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 1}];
        args.common.dst           = {args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 2}], args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 3}]};
        args.common.src_uid[0] = args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 4}];
        args.common.src_uid[1] = args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 5}];
        args.common.src_batch[0] = args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 6}];
        args.common.src_batch[1] = args.globals.instructions[kittens::coord<>{0, (int)(blockIdx.x), args.task_iter, 7}];
        args.num_iters = 1;
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
            if(warpgroup::warpid() == args.iter%4) {
                // next page we need to load?
                tma::expect(args.inputs_arrived, args.input.o[0], args.input.o[1], args.input.lvec[0], args.input.lvec[1]);
                tma::load_async(args.input.o[0], args.globals.O_scratch, {args.common.src_batch[0], args.common.dst.seq_idx, 0, 0}, args.inputs_arrived);
                tma::load_async(args.input.o[1], args.globals.O_scratch, {args.common.src_batch[1], args.common.dst.seq_idx, 0, 0}, args.inputs_arrived);
                tma::load_async(args.input.lvec[0], args.globals.Lvec_scratch, {args.common.src_batch[0], args.common.dst.seq_idx, 0}, args.inputs_arrived);
                tma::load_async(args.input.lvec[1], args.globals.Lvec_scratch, {args.common.src_batch[1], args.common.dst.seq_idx, 0}, args.inputs_arrived);
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
            }
        }
        __device__ static inline void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                if(args.common.dst.batch_idx >= 0) {
                    tma::store_async(args.globals.O, reinterpret_cast<st_bf<16, VO_D>&>(args.output.o),
                        {args.common.dst.batch_idx, args.common.dst.seq_idx, 0, group<8>::warpid()});
                }
                else {
                    tma::store_async(args.globals.O_scratch, args.output.o, {-args.common.dst.batch_idx, args.common.dst.seq_idx, 0});
                    tma::store_async(args.globals.Lvec_scratch, args.output.lvec, {-args.common.dst.batch_idx, args.common.dst.seq_idx});
                }
                tma::store_async_wait();
                __syncwarp();
                if(laneid() == 0) arrive(args.outputs_finished, 4);
            }
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {}
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            col_vec<rt_fl<16, cache_tile::rows>> lvec[2], max_lvec, sum_lvec;
            auto o1_st = subtile_inplace<16, VO_D/8>(args.input.o[0], {0, group<8>::warpid()});
            auto o2_st = subtile_inplace<16, VO_D/8>(args.input.o[1], {0, group<8>::warpid()});
            rt_fl<16, VO_D / 8> o[2];
            load(o[0], o1_st);
            load(o[1], o2_st);
            load(lvec[0], args.input.lvec[0]);
            load(lvec[1], args.input.lvec[1]);
            __syncwarp();
            if(laneid() == 0) arrive(args.inputs_finished); // done!
            max(max_lvec, lvec[0], lvec[1]);
            sub(lvec[0], lvec[0], max_lvec);
            sub(lvec[1], lvec[1], max_lvec);
            exp2(lvec[0], lvec[0]);
            exp2(lvec[1], lvec[1]);
            add(sum_lvec, lvec[0], lvec[1]);
            div(lvec[0], lvec[0], sum_lvec);
            div(lvec[1], lvec[1], sum_lvec);
            mul_row(o[0], o[0], lvec[0]);
            mul_row(o[1], o[1], lvec[1]);
            add(o[0], o[0], o[1]);
            log2(sum_lvec, sum_lvec);
            add(sum_lvec, sum_lvec, max_lvec);
            if(args.common.dst.batch_idx >= 0) {
                auto &o_smem = reinterpret_cast<st_bf<16, VO_D>&>(args.output.o);
                auto o_st = subtile_inplace<16, VO_D/8>(o_smem, {0, group<8>::warpid()});
                store(o_st, o[0]);
            }
            else {
                auto o_st = subtile_inplace<16, VO_D/8>(args.output.o, {0, group<8>::warpid()});
                store(o_st, o[0]);
                if(group<8>::warpid() == 0) {
                    store(args.output.lvec, sum_lvec);
                }
            }
            __syncwarp();
            if(laneid() == 0) arrive(args.outputs_arrived); // done!
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            // Increment the semaphore for the next stage, if this is not the last one.
            warpgroup::sync(warpgroup::groupid()); // Make sure memory has been flushed to global memory.
            if(args.common.dst.batch_idx < 0) {
                asm volatile("fence.sc.sys;");
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
    py::bind_kernel<interpreter::kernel<config, partial_template, reduction_template>>(m, "mla_decode",
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