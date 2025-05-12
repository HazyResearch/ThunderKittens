#define KITTENS_TIMINGS

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

static constexpr int QKRot_D = 64, QVO_D = 512, QVO_Dd2 = QVO_D/2, NUM_ROWS = 32, PAGE_SIZE = 256;
using qrot_tile           = st_bf<64, QKRot_D>;
using qvo_tile            = st_bf<64, QVO_D>;
using q_global            = kittens::gl<bf16, -1, -1, -1, QKRot_D, qrot_tile>; // B * R * H * D_QKRot_D
using qv_global           = kittens::gl<bf16, -1, -1, -1, QVO_D, qvo_tile>; // B * R * H * D_QVO_D
using kcache_tile         = st_bf<NUM_ROWS, QKRot_D>;
using vcache_tile         = st_bf<NUM_ROWS, QVO_D>; // we need the v_tile for later
using vcache_tile2        = st_bf<NUM_ROWS, QVO_Dd2>; // we need the v_tile for later
using kcache_global       = kittens::gl<bf16, 1, -1, PAGE_SIZE, QKRot_D, kcache_tile>; // 1 * #page * pagesize * QKRot_D
using vcache_global       = kittens::gl<bf16, 1, -1, PAGE_SIZE, QVO_D, vcache_tile>; // 1 * #page * pagesize * QVO_D
using instructions_global = kittens::gl<int, 1, -1, -1, 32>;
using table_global        = kittens::gl<int, 1, 1, -1, -1>; // B * (max # pages)
using o_tile              = st_bf<64, QVO_D>;
using o_tile_fl           = st_fl<16, QVO_D>;
using o_global            = kittens::gl<bf16, -1, -1, -1, QVO_D, st_bf<16, QVO_Dd2>, st_bf<16, QVO_D/8>>; // B * NEWTOKENS * H * D_VO

template<int Q_HEADS=16>
using o_scratch_global    = kittens::gl<float, -1, -1, Q_HEADS, QVO_D, st_fl<16, QVO_D/8>, st_fl<16,256>>; // For partial O's

template<int Q_HEADS=16>
using lvec_scratch_global = kittens::gl<float,  1, -1, -1, Q_HEADS, sv_fl<16>>; // For partial O's
using semaphore_global    = kittens::gl<int,    1,  1,  -1, -1>;            // 1 * 1 * uid * NEWTOKENS

template<int Q_HEADS=16>
struct config {
    struct globals {
        using instructions_global = instructions_global;
        instructions_global instructions;
        q_global Q;
        qv_global QV;
        kcache_global K_cache;
        vcache_global V_cache;
        table_global Table;
        o_global O;
        o_scratch_global<Q_HEADS> O_scratch;
        lvec_scratch_global<Q_HEADS> Lvec_scratch;
        semaphore_global semaphore;
        const float Softmax_scale;
        int tic;
#ifdef KITTENS_TIMINGS
        gl<int, 1, -1, -1, 64> timings;
#endif
        int dynamic_shared_memory() { return 226000; }
        dim3 grid()  { return dim3(132); } //dim3(Q.batch * ((Q.depth + 3) / 4)); }
        dim3 block() { return dim3((8+4)*WARP_THREADS); }
    };
};
template<int Q_HEADS=16>
struct config_partial {
    struct globals {
        using instructions_global = instructions_global;
        instructions_global instructions;
        q_global Q;
        qv_global QV;
        kcache_global K_cache;
        vcache_global V_cache;
        table_global Table;
        o_global O;
        o_scratch_global<Q_HEADS> O_scratch;
        lvec_scratch_global<Q_HEADS> Lvec_scratch;
        semaphore_global semaphore;
        const float Softmax_scale;
        int tic;
#ifdef KITTENS_TIMINGS
        gl<int, 1, -1, -1, 64> timings;
#endif
        int dynamic_shared_memory() { return 226000; }
        dim3 grid()  { return dim3(132); } //dim3(Q.batch * ((Q.depth + 3) / 4)); }
        dim3 block() { return dim3((8+4)*WARP_THREADS); }
    };
};
template<int Q_HEADS=16>
struct config_reduction {
    struct globals {
        using instructions_global = instructions_global;
        instructions_global instructions;
        o_global O;
        o_scratch_global<Q_HEADS> O_scratch;
        lvec_scratch_global<Q_HEADS> Lvec_scratch;
        semaphore_global semaphore;
        const float Softmax_scale;
        int tic;
#ifdef KITTENS_TIMINGS
        gl<int, 1, -1, -1, 64> timings;
#endif
        int dynamic_shared_memory() { return 226000; }
        dim3 grid()  { return dim3(132); } //dim3(Q.batch * ((Q.depth + 3) / 4)); }
        dim3 block() { return dim3((8+4)*WARP_THREADS); }
    };
};

__device__ static inline uint64_t my_clock() {
    uint64_t time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}

struct location {
    int batch_idx; // batch_idx >=0, otherwise it's the negative index, minus one, into scratch
    int seq_idx;
};
template<int Q_HEADS=16>
struct partial_layout {
    using globals = config<Q_HEADS>::globals;
    struct input_block { kcache_tile kcache; vcache_tile vcache; };
    struct scratch_block { qrot_tile qrot; qvo_tile qvo; st_bf<64, kcache_tile::rows> att_block; sv_fl<64> max_vec, norm_vec; };
    struct finish_block { st_fl<16, QVO_Dd2> o[4][2]; sv_fl<16> lvec[4]; };
    struct common_state {
        int uid;
        location dst;
        int q_batch_idx;
        int q_seq_idx;
        int start_pos; // MUST BE A MULTIPLE OF PAGE_SIZE
        int end_pos; // One past the last position to load
        int length; // the length of the overall sequence in question
    };
    struct consumer_state {
        col_vec<rt_fl<16, kcache_tile::rows>> max_vec, norm_vec;
        rt_fl<16, QVO_Dd2> o;
    };
};
template<int Q_HEADS=16>
struct partial_template {
    using config = config<Q_HEADS>;
    using layout = partial_layout<Q_HEADS>;
    static constexpr int opcode = 1;
    static constexpr int INPUT_PIPE_STAGES = 3;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
#ifdef KITTENS_TIMINGS
        if(group<12>::laneid() == 0) args.timings[0] = my_clock();
#endif
        args.common.uid         =  args.instruction[1];
        args.common.dst         = {args.instruction[2],
                                   args.instruction[3]};
        args.common.q_batch_idx =  args.instruction[4];
        args.common.q_seq_idx   =  args.instruction[5];
        args.common.start_pos   =  args.instruction[6];
        args.common.end_pos     =  args.instruction[7];
        args.common.length      =  args.instruction[8];
        args.num_iters          = (args.common.end_pos - args.common.start_pos + NUM_ROWS - 1) / NUM_ROWS;
        args.common.length    -= (args.globals.Q.depth() - (args.common.q_seq_idx + warpgroup::warpid()) - 1); // adjust for the causal mask
        
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static inline void load(producer_load_args<layout> args) {
            if(args.iter == 1) group<12>::sync(11); // wait for the consumer to finish its setup, before we do the second load.
            if(warpgroup::warpid() == 0) {
                int pos = args.common.start_pos + NUM_ROWS*args.iter;
                int within_page_idx = (pos % PAGE_SIZE) / NUM_ROWS;
                int next_page_id = args.globals.Table[coord<>{args.common.q_batch_idx, pos/PAGE_SIZE}];
                // next page we need to load?
                warp::tma::expect(args.inputs_arrived, args.input.kcache, args.input.vcache);
                warp::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(args.input.kcache, args.globals.K_cache, {0, next_page_id, within_page_idx, 0}, args.inputs_arrived);
                warp::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(args.input.vcache, args.globals.V_cache, {0, next_page_id, within_page_idx, 0}, args.inputs_arrived);
                warp::arrive(args.inputs_arrived, 3);
            }
#ifdef KITTENS_TIMINGS
            else if(warpgroup::laneid() == 32 && args.iter < 24) args.timings[32+args.iter] = my_clock();
#endif
            warpgroup::sync(5);
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[1] = my_clock();
#endif
            auto qrot_st = args.scratch.qrot.template subtile<16, QKRot_D/2>({warpgroup::warpid(), warpgroup::groupid()});
            warp::load_async(qrot_st, args.globals.Q, {args.common.q_batch_idx, args.common.q_seq_idx + warpgroup::warpid(), 0, warpgroup::groupid()});
            auto qvo_st = args.scratch.qvo.template subtile<16, QVO_Dd2>({warpgroup::warpid(), warpgroup::groupid()});
            warp::load_async(qvo_st, args.globals.QV, {args.common.q_batch_idx, args.common.q_seq_idx + warpgroup::warpid(), 0, warpgroup::groupid()});
            warp::zero(args.state.norm_vec);
            if(args.num_iters > 0) warp::neg_infty(args.state.max_vec);
            else { warp::one(args.state.max_vec); warp::mul(args.state.max_vec, args.state.max_vec, -999999.f); }
            warp::zero(args.state.o);
            warp::load_async_wait();
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[2] = my_clock();
#endif
        }
        template<bool do_right_fill> __device__ static inline void internal_compute(consumer_compute_args<layout> args) {
            // 1.44269504089f is from exp2
            if(args.iter == 0 && args.num_iters > 1) {
                group<12>::arrive(11); // this <12> will allow us to prevent the second producer load from happening before this point.
            }
            const float SOFTMAX_TEMPERATURE = args.globals.Softmax_scale * 1.44269504089f;
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0 && args.iter < 24) args.timings[8+args.iter] = my_clock();
#endif

            col_vec<rt_fl<16, kcache_tile::rows>> local_max_vec, local_norm_vec;
            col_vec<rt_fl<16, kcache_tile::rows>> max_vec_last_scaled, max_vec_scaled;

            kittens::barrier<8> cons_barrier(10);

            if(warpgroup::groupid() == 0) {
                // A = Q @ K.T
                rt_fl<16, kcache_tile::rows> att_block_fp32;
                warpgroup::mm_ABt(att_block_fp32, args.scratch.qrot, args.input.kcache);
                warpgroup::mma_ABt(att_block_fp32, args.scratch.qvo, args.input.vcache);

                warp::copy(local_max_vec,  args.state.max_vec);
                warp::copy(local_norm_vec, args.state.norm_vec);

                warp::mul(max_vec_last_scaled, local_max_vec, SOFTMAX_TEMPERATURE);

                warpgroup::mma_async_wait();
                // softmax
                if constexpr (do_right_fill) { // need to mask out a bunch of entries in the last page
                    const int length = args.common.length - args.common.start_pos - args.iter*NUM_ROWS;
                    // right_fill(att_block_fp32, att_block_fp32, length, -9999999999.f);
                    warp::apply(att_block_fp32, att_block_fp32, [length]__device__(int row, int col, const float &val) {
                        return col < length ? val : -9999999999.f;
                    });
                }

                warp::row_max(local_max_vec, att_block_fp32, local_max_vec);
                warp::mul(max_vec_scaled, local_max_vec, SOFTMAX_TEMPERATURE);

                warp::mul(att_block_fp32, att_block_fp32, SOFTMAX_TEMPERATURE);
                warp::sub_row(att_block_fp32, att_block_fp32, max_vec_scaled);
                
                warp::exp2(att_block_fp32, att_block_fp32);
                
                warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
                warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
                warpgroup::store(args.scratch.max_vec, max_vec_last_scaled);
                
                warp::mul(local_norm_vec, local_norm_vec, max_vec_last_scaled);
                warp::row_sum(local_norm_vec, att_block_fp32, local_norm_vec);
                warpgroup::store(args.scratch.att_block, att_block_fp32);
                arrive(cons_barrier);
            }
            else {
                arrive_and_wait(cons_barrier);
                warpgroup::load(max_vec_last_scaled, args.scratch.max_vec);
            }

            warp::mul_row(args.state.o, args.state.o, max_vec_last_scaled); // normalize o_reg before mma

            // O += A @ V
            auto (&v_smem)[2] = reinterpret_cast<vcache_tile2(&)[2]>(args.input.vcache);
            warpgroup::mma_AB(args.state.o, args.scratch.att_block, v_smem[warpgroup::groupid()]);

            warp::copy(args.state.max_vec, local_max_vec);
            warp::copy(args.state.norm_vec, local_norm_vec);

            warpgroup::mma_async_wait();
            if(warpgroup::laneid() == 0) arrive(args.inputs_finished, WARPGROUP_WARPS); // done!
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            if(args.iter >= args.num_iters-2) internal_compute<true>(args);
            else internal_compute<false>(args);
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            col_vec<rt_fl<16, kcache_tile::rows>> local_max_vec, local_norm_vec;

            warp::copy(local_norm_vec, args.state.norm_vec);
            warp::copy(local_max_vec, args.state.max_vec);

#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[62] = my_clock(); // Start of store out.
#endif

            if (warpgroup::groupid() == 0) warpgroup::store(args.scratch.norm_vec, local_norm_vec);
            group<8>::sync(10);
            if(warpgroup::groupid() == 1) warpgroup::load(local_norm_vec, args.scratch.norm_vec);
            warp::div_row(args.state.o, args.state.o, local_norm_vec);

            if(args.common.dst.batch_idx >= 0) { // batch is meaningful
                auto &o_smem = reinterpret_cast<st_bf<16, QVO_Dd2>&>(args.finish.o[warpgroup::warpid()][warpgroup::groupid()]);
                warp::store(o_smem, args.state.o);
                __syncwarp();
                warp::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(args.globals.O, o_smem, {args.common.dst.batch_idx, args.common.dst.seq_idx+warpgroup::warpid(), 0, warpgroup::groupid()});
            }
            else { // write out directly to O scratch, without going through smem
                if(warpgroup::groupid() == 0) {
                    warp::mul(local_max_vec, local_max_vec, args.globals.Softmax_scale * 1.44269504089f);
                    warp::log2(local_norm_vec, local_norm_vec);
                    warp::add(local_norm_vec, local_norm_vec, local_max_vec); // l_vec = log2(norm_vec) + max_vec
                    warp::store(args.finish.lvec[warpgroup::warpid()], local_norm_vec);
                    __syncwarp();
                    warp::tma::store_async<cache_policy::EVICT_LAST>(args.globals.Lvec_scratch, args.finish.lvec[warpgroup::warpid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx+warpgroup::warpid(), 0});
                }
                warp::store(args.finish.o[warpgroup::warpid()][warpgroup::groupid()], args.state.o);
                __syncwarp();
                warp::tma::store_async<dim::ROW, cache_policy::EVICT_LAST>(args.globals.O_scratch, args.finish.o[warpgroup::warpid()][warpgroup::groupid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx+warpgroup::warpid(), 0, warpgroup::groupid()});
            }
            warpgroup::arrive(args.finish_finished, WARPGROUP_WARPS); // done!
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 32) args.timings[63] = my_clock();
#endif
        }
    };
};

template<int Q_HEADS=16, typename _config=config<Q_HEADS>>
struct reduction_layout {
    using config = _config;
    using globals = config::globals;
    struct input_block   { st_fl<16, QVO_D/8> o[8]; sv_fl<16> lvec; sv_fl<16> padding[15]; };
    struct scratch_block { st_fl<16, QVO_D/8> o[8]; sv_fl<16> lvec; semaphore producer_block; }; // used both for setup load and finish store
    struct common_state {
        int uid;
        // int num_iters; // same as the number of active load_uid's, marked here for instruction clarity but we just use args.num_iters instead.
        location dst; // again, negative batch means we're writing to O scratch, seq_idx is consistent
        int src_uid;
    };
    struct consumer_state {
        rt_fl<16, QVO_D/8> o;
        col_vec<rt_fl<16, kcache_tile::rows>> lvec;
    };
};

template<int Q_HEADS=16, typename _config=config<Q_HEADS>>
struct reduction_template {
    using config = _config;
    using layout = reduction_layout<Q_HEADS, config>;
    static constexpr int opcode = 2;
    static constexpr int INPUT_PIPE_STAGES = 4;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
#ifdef KITTENS_TIMINGS
        if(group<12>::laneid() == 0) args.timings[0] = my_clock();
#endif
        args.common.uid     =  args.instruction[1];
        args.num_iters      =  args.instruction[2];
        args.common.dst     = {args.instruction[3],
                               args.instruction[4]};
        args.common.src_uid =  args.instruction[5];
        group<12>::sync(7);
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static inline void load(producer_load_args<layout> args) {
            // if(args.iter == 1) group<12>::sync(8);
            if(warpgroup::warpid() == args.iter%4) {
                // spinloop until we're ready
                int load_uid = args.instruction[6+args.iter];
                if(laneid() == 0) while(*(volatile int*)&args.globals.semaphore[{load_uid, args.common.dst.seq_idx}] != args.globals.tic) {}
                __syncwarp();
#ifdef KITTENS_TIMINGS
                if(laneid() == 0 && args.iter < 24) args.timings[32+args.iter] = my_clock();
#endif
                // next page we need to load?
                warp::tma::expect(args.inputs_arrived, args.input.o, args.input.lvec);
                #pragma unroll
                for(int i = 0; i < 8; i++) {
                    warp::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(args.input.o[i], args.globals.O_scratch, {load_uid, args.common.dst.seq_idx, 0, i}, args.inputs_arrived);
                }
                warp::tma::load_async<cache_policy::EVICT_FIRST>(args.input.lvec, args.globals.Lvec_scratch, {load_uid, args.common.dst.seq_idx, 0}, args.inputs_arrived);
                warp::arrive(args.inputs_arrived, 3);
            }
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            // If we are doing a reduction, we need to spinloop until we have confirmation that all the partial results have been written out.
            if(threadIdx.x == 0) { // easier to have a single thread spin
                while(*(volatile int*)&args.globals.semaphore[{args.common.src_uid, args.common.dst.seq_idx}] != args.globals.tic) {} // note volatile, L1 is not guaranteed to be coherent.
            }
            group<8>::sync(11); // all warps must sync here.
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[1] = my_clock();
#endif
            warp::load_async(args.scratch.o[group<8>::warpid()], args.globals.O_scratch, {args.common.src_uid, args.common.dst.seq_idx, 0, group<8>::warpid()});
            if(warpid() == 0) {
                warp::load_async(args.scratch.lvec, args.globals.Lvec_scratch, {args.common.src_uid, args.common.dst.seq_idx, 0});
            }
            warp::load_async_wait();
            __syncwarp();
            warp::load(args.state.o, args.scratch.o[group<8>::warpid()]);
            group<8>::sync(11); // we use this to also stall the producer until the consumer is ready.
            // group<12>::sync(9);
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[2] = my_clock();
#endif
            warp::load(args.state.lvec, args.scratch.lvec);
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0 && args.iter < 24) args.timings[8+args.iter] = my_clock();
#endif
            col_vec<rt_fl<16, kcache_tile::rows>> lvec, max_lvec, sum_lvec;
            rt_fl<16, QVO_D / 8> o;
            warp::load(o, args.input.o[group<8>::warpid()]);
            warp::load(lvec, args.input.lvec);
            __syncwarp();
            warp::arrive(args.inputs_finished); // done!
            warp::max(max_lvec, args.state.lvec, lvec);
            warp::sub(args.state.lvec, args.state.lvec, max_lvec);
            warp::sub(lvec, lvec, max_lvec);
            warp::exp2(args.state.lvec, args.state.lvec);
            warp::exp2(lvec, lvec);
            warp::add(sum_lvec, args.state.lvec, lvec);
            warp::div(args.state.lvec, args.state.lvec, sum_lvec);
            warp::div(lvec, lvec, sum_lvec);
            warp::mul_row(args.state.o, args.state.o, args.state.lvec);
            warp::mul_row(o, o, lvec);
            warp::add(args.state.o, args.state.o, o);
            warp::log2(sum_lvec, sum_lvec);
            warp::add(args.state.lvec, sum_lvec, max_lvec);
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[62] = my_clock();
#endif
            if(args.common.dst.batch_idx >= 0) {
                auto &o_smem = reinterpret_cast<st_bf<16, QVO_D/8>&>(args.scratch.o[group<8>::warpid()]);
                warp::store(o_smem, args.state.o);
                __syncwarp();
                warp::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(args.globals.O, o_smem, {args.common.dst.batch_idx, args.common.dst.seq_idx, 0, group<8>::warpid()});
            }
            else {
                warp::store(args.scratch.o[group<8>::warpid()], args.state.o);
                if(group<8>::warpid() == 0) warp::store(args.scratch.lvec, args.state.lvec);
                __syncwarp();
                warp::tma::store_async<dim::ROW, cache_policy::EVICT_LAST>(args.globals.O_scratch, args.scratch.o[group<8>::warpid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx, 0, group<8>::warpid()});
                if(group<8>::warpid() == 0) warp::tma::store_async<cache_policy::EVICT_LAST>(args.globals.Lvec_scratch, args.scratch.lvec, {-args.common.dst.batch_idx-1, args.common.dst.seq_idx, 0});
            }
            tma::store_async_wait();
            asm volatile("fence.sc.cta;\n");
            group<8>::sync(11);
            // Increment the semaphore for the next stage, if this is not the last one.
            if(args.common.dst.batch_idx < 0) {
                if(group<8>::laneid() == 0) {
                    args.globals.semaphore[{-args.common.dst.batch_idx-1, args.common.dst.seq_idx}] = args.globals.tic;
                }
            }
            warpgroup::arrive(args.finish_finished, WARPGROUP_WARPS); // done!
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 32) args.timings[63] = my_clock();
#endif
        }
    };
};

#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Timing constants (in microseconds)
constexpr float PARTIAL_STARTUP_TIME = 3.0f;         // Startup time for partial operations
constexpr float PARTIAL_WRITEOUT_TIME = 4.5f;        // Writeout time for partial operations
constexpr float PARTIAL_COST_PER_STEP = 1.49f;       // Cost per step (per 32 tokens) for partial operations
constexpr float PARTIAL_OVERHEAD = PARTIAL_STARTUP_TIME + PARTIAL_WRITEOUT_TIME; // Total overhead for a partial operation.

constexpr float REDUCTION_STARTUP_TIME = 2.0f;       // Startup time for reduction operations
// constexpr float REDUCTION_WRITEOUT_TIME = 1.0f;   // Writeout time for reduction operations, not used so commented to disable warnings.
constexpr float REDUCTION_PRODUCER_LATENCY = 1.0f;   // Latency between a producer load and when the consumer can access it.
constexpr float REDUCTION_COST_PER_STEP = 0.4f;      // Cost per reduction step

// constexpr float SYNCHRONIZATION_COST = 0.5f;      // Synchronization cost between dependent operations, not used so commented to disable warnings.

float get_quality(const std::vector<float>& next_times_input, int num_processors, int num_tokens, int seq_length) {
    int num_partial_steps = (seq_length + 31) / 32;

    if (next_times_input.size() > num_processors) {
        // This particular scheduler is just not set up to deal with these situations.
        return -999999999.0f;
    }
    
    // Copy the input times so we can modify them
    std::vector<float> next_times = next_times_input;
    std::sort(next_times.begin(), next_times.end(), std::greater<float>());

    std::vector<float> partial_times;
    for (int i = 0; i < num_processors; i++) {
        next_times[i%next_times.size()] -= REDUCTION_COST_PER_STEP;
        partial_times.push_back(next_times[i%next_times.size()]);
    }

    // We also have to account for the fact that some of these partials are going to be forced to start earlier than the coscheduled reduction.
    // The number of these is equal to the number of reduction ops * the number of tokens, since those are each handled independently.
    std::sort(partial_times.begin(), partial_times.end()); // Thankfully we can pick the worst ones to be forced to start earlier.
    
    for (size_t j = 0; j < next_times.size(); j++) {
        float actual_start_time = next_times[j] + REDUCTION_PRODUCER_LATENCY - REDUCTION_STARTUP_TIME; // When does this instruction actually start?
        for (int k = 0; k < num_tokens; k++) {
            if (num_tokens * j + k < partial_times.size()) {
                partial_times[num_tokens * j + k] = actual_start_time;
            }
        }
    }
    
    // Now that we know when the partials are going to start, we can start to assign the steps of the work.
    std::sort(partial_times.begin(), partial_times.end(), std::greater<float>()); // Largest to smallest.
    
    float min_value = partial_times.back();
    for(int i = 0; i < partial_times.size(); i++) {
        if(num_partial_steps > 0) {
            int num_steps_alloc = std::min(num_partial_steps, (int)(round((partial_times[i]-min_value) / PARTIAL_COST_PER_STEP)));
            num_partial_steps -= num_steps_alloc;
            partial_times[i] -= num_steps_alloc * PARTIAL_COST_PER_STEP;
            if(num_steps_alloc > 0) partial_times[i] -= PARTIAL_OVERHEAD;
        }
    }

    int full_passes = num_partial_steps / partial_times.size();
    int remainder = num_partial_steps - (full_passes * partial_times.size());

    std::sort(partial_times.begin(), partial_times.end(), std::greater<float>());
    min_value = 9999999999.0f;
    for(int i = 0; i < remainder; i++){
        float f = partial_times[i] - PARTIAL_COST_PER_STEP * (full_passes+1);
        if(f < min_value) min_value = f;
    }
    for(int i = remainder; i < partial_times.size(); i++) {
        float f = partial_times[i] - PARTIAL_COST_PER_STEP * full_passes;
        if(f < min_value) min_value = f;
    }

    return min_value;
}

PYBIND11_MODULE(mla_decode, m) {
    m.doc() = "mla_decode python module";
    kittens::py::bind_kernel<interpreter::kernel<config<16>, partial_template<16>, reduction_template<16, config<16>>>>(m, "mla_decode",
        &config<16>::globals::instructions,
        &config<16>::globals::Q,
        &config<16>::globals::QV,
        &config<16>::globals::K_cache,
        &config<16>::globals::V_cache,
        &config<16>::globals::Table,
        &config<16>::globals::O,
        &config<16>::globals::O_scratch,
        &config<16>::globals::Lvec_scratch,
        &config<16>::globals::semaphore,
        &config<16>::globals::Softmax_scale,
        &config<16>::globals::tic
#ifdef KITTENS_TIMINGS
        , &config<16>::globals::timings
#endif
    );
    kittens::py::bind_kernel<interpreter::kernel<config_reduction<16>, reduction_template<16, config_reduction<16>>>>(m, "mla_decode_reduction",
        &config_reduction<16>::globals::instructions,
        &config_reduction<16>::globals::O,
        &config_reduction<16>::globals::O_scratch,
        &config_reduction<16>::globals::Lvec_scratch,
        &config_reduction<16>::globals::semaphore,
        &config_reduction<16>::globals::Softmax_scale,
        &config_reduction<16>::globals::tic
#ifdef KITTENS_TIMINGS
        , &config_reduction<16>::globals::timings
#endif
    );
    
    kittens::py::bind_kernel<interpreter::kernel<config<8>, partial_template<8>, reduction_template<8, config<8>>>>(m, "mla_decode_8_heads",
        &config<8>::globals::instructions,
        &config<8>::globals::Q,
        &config<8>::globals::QV,
        &config<8>::globals::K_cache,
        &config<8>::globals::V_cache,
        &config<8>::globals::Table,
        &config<8>::globals::O,
        &config<8>::globals::O_scratch,
        &config<8>::globals::Lvec_scratch,
        &config<8>::globals::semaphore,
        &config<8>::globals::Softmax_scale,
        &config<8>::globals::tic
#ifdef KITTENS_TIMINGS
        , &config<8>::globals::timings
#endif
    );
    m.def("__get_quality__", &get_quality, 
        "An internal utility function for generating efficient schedules.",
        pybind11::arg("next_times"), 
        pybind11::arg("num_processors"), 
        pybind11::arg("num_tokens"), 
        pybind11::arg("seq_length")
    );
}