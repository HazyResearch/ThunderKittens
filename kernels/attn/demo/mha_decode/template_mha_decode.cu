#define KITTENS_TIMINGS

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

static constexpr int DIM       = 128; 
static constexpr int NUM_ROWS  = 64; 
static constexpr int PAGE_SIZE = 256; 

using q_tile   = st_bf<64, DIM>; 
using q_global = kittens::gl<bf16, -1, -1, -1, DIM, tma::descriptor<q_tile, 1>>; // B * R * H * DIM

using kcache_tile = st_bf<NUM_ROWS, DIM>; 
using vcache_tile = st_bf<NUM_ROWS, DIM>; 
using kcache_global = kittens::gl<bf16, -1, PAGE_SIZE, -1, DIM, tma::descriptor<kcache_tile, 1>>; // #page * pagesize * H * DIM
using vcache_global = kittens::gl<bf16, -1, PAGE_SIZE, -1, DIM, tma::descriptor<vcache_tile, 1>>; // #page * pagesize * H * DIM

using instructions_global = kittens::gl<int,  1, -1, -1, 32>;
using table_global        = kittens::gl<int,  1,  1, -1, -1>; // B * (max # pages)

using o_tile              = st_bf<16, DIM>;
using o_tile_fl           = st_fl<16, DIM>;

using o_global            = kittens::gl<bf16,  -1, -1, -1, DIM, tma::descriptor<o_tile, 1>, tma::descriptor<st_bf<16,32>, 1>>; // B * R * H * DIM

// using o_scratch_global    = kittens::gl<float, -1, -1, 16, DIM, st_fl<16, QVO_D/8>, st_fl<16,256>>; // For partial O's
// using lvec_scratch_global = kittens::gl<float,  1, -1, -1, 16,  sv_fl<16>>; // For partial O's
using o_scratch_global    = kittens::gl<bf16, -1, -1, -1, DIM, tma::descriptor<st_fl<16,128>, 1>, tma::descriptor<st_fl<16,32>, 1>>; // For partial O's SCRATCH_DIM * NEWTOKENS * H * DIM
using lvec_scratch_global = kittens::gl<float, 1, -1, -1, -1,  sv_fl<16>>;     // For partial O's SCRATCH_DIM * NEWTOKENS * H

using semaphore_global    = kittens::gl<int,    1,  1, -1, -1>;             // 1 * 1 * uid * NEWTOKENS

struct config {
    struct globals {
        using instructions_global = instructions_global;
        instructions_global instructions;
        
        q_global  Q;
        kcache_global K_cache;
        vcache_global V_cache;
        table_global  Table;
        
        o_global O;
        o_scratch_global O_scratch;
        
        lvec_scratch_global Lvec_scratch;
        semaphore_global    semaphore;
        
        const float Softmax_scale;
        int tic;
        gl<int, 1, -1, -1, 64> timings;
        
        int dynamic_shared_memory() { return 226000; }
        dim3 grid()                 { return dim3(132); }
        dim3 block()                { return dim3((8+4)*WARP_THREADS); }
    };
};

struct location {
    int batch_idx; // batch_idx >=0, otherwise it's the negative index, minus one, into scratch
    int seq_idx;
};
struct partial_layout {
    using globals = config::globals;
    struct input_block { kcache_tile kcache; vcache_tile vcache; };
    struct scratch_block { q_tile q; st_bf<64, kcache_tile::rows> att_block; sv_fl<64> max_vec, norm_vec; };
    struct finish_block { st_fl<16, DIM> o[4]; sv_fl<16> lvec[4]; };
    struct common_state {
        int uid;
        location dst;
        int q_batch_idx;
        int q_seq_idx; 
        int start_pos; // MUST BE A MULTIPLE OF PAGE_SIZE
        int end_pos;   // One past the last position to load
        int length;    // the length of the overall sequence in question
        int head; 
    };
    struct consumer_state {
        col_vec<rt_fl<16, kcache_tile::rows>> max_vec, norm_vec;
        rt_fl<16, DIM> o;
    };
};
struct partial_template {
    using config = config;
    using layout = partial_layout;
    static constexpr int opcode = 1;
    static constexpr int INPUT_PIPE_STAGES = 3;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if(group<12>::laneid() == 0) args.timings[0] = clock64();
        args.common.uid         =  args.instruction[1];
        args.common.dst         = {args.instruction[2],
                                   args.instruction[3]};
        args.common.q_batch_idx =  args.instruction[4];
        args.common.q_seq_idx   =  args.instruction[5];
        args.common.start_pos   =  args.instruction[6];
        args.common.end_pos     =  args.instruction[7];
        args.common.length      =  args.instruction[8];
        args.common.head        =  args.instruction[9];
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
                tma::expect(args.inputs_arrived, args.input.kcache, args.input.vcache);
                // tma::expect(args.inputs_arrived, args.input.vcache);
                // cache shape is #page * pagesize * H * DIM
                tma::load_async<1, cache_policy::EVICT_FIRST>(args.input.kcache, args.globals.K_cache, {next_page_id, within_page_idx, args.common.head, 0}, args.inputs_arrived);
                tma::load_async<1, cache_policy::EVICT_FIRST>(args.input.vcache, args.globals.V_cache, {next_page_id, within_page_idx, args.common.head, 0}, args.inputs_arrived);
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
            }
            else if(warpgroup::laneid() == 32 && args.iter < 24) args.timings[32+args.iter] = clock64();
            warpgroup::sync(5);
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            if(group<8>::laneid() == 0) args.timings[1] = clock64();
            
            // auto qrot_st = subtile_inplace<16, QKRot_D/2>(args.scratch.qrot, {warpgroup::warpid(), warpgroup::groupid()});
            // load_async(qrot_st, args.globals.Q, {args.common.q_batch_idx, args.common.q_seq_idx + warpgroup::warpid(), 0, warpgroup::groupid()});
            // auto qvo_st = subtile_inplace<16, QVO_Dd2>(args.scratch.qvo, {warpgroup::warpid(), warpgroup::groupid()});
            // load_async(qvo_st, args.globals.QV, {args.common.q_batch_idx, args.common.q_seq_idx + warpgroup::warpid(), 0, warpgroup::groupid()});
            group<8>::load_async<1, false>(args.scratch.q, args.globals.Q, {args.common.q_batch_idx, args.common.q_seq_idx, args.common.head, 0});
            
            zero(args.state.norm_vec);
            neg_infty(args.state.max_vec);
            zero(args.state.o);
            load_async_wait();
            barrier<12> producer_barrier(11);
            arrive(producer_barrier); // this <12> will allow us to prevent the second producer load from happening before this point.
            if(group<8>::laneid() == 0) args.timings[2] = clock64();
        }
        template<bool do_right_fill> __device__ static inline void internal_compute(consumer_compute_args<layout> args) {
            // 1.44269504089f is from exp2
            const float SOFTMAX_TEMPERATURE = args.globals.Softmax_scale * 1.44269504089f;
            if(group<8>::laneid() == 0 && args.iter < 24) args.timings[8+args.iter] = clock64();

            col_vec<rt_fl<16, kcache_tile::rows>> local_max_vec, local_norm_vec;
            col_vec<rt_fl<16, kcache_tile::rows>> max_vec_last_scaled, max_vec_scaled;

            kittens::barrier<8> cons_barrier(10);

            if(warpgroup::groupid() == 0) {
                // A = Q @ K.T
                rt_fl<16, kcache_tile::rows> att_block_fp32;
                // warpgroup::mm_ABt(att_block_fp32, args.scratch.qrot, args.input.kcache);
                // warpgroup::mma_ABt(att_block_fp32, args.scratch.qvo, args.input.vcache);
                warpgroup::mm_ABt(att_block_fp32, args.scratch.q, args.input.kcache);

                copy(local_max_vec,  args.state.max_vec);
                copy(local_norm_vec, args.state.norm_vec);

                mul(max_vec_last_scaled, local_max_vec, SOFTMAX_TEMPERATURE);

                warpgroup::mma_async_wait();
                // softmax
                if constexpr (do_right_fill) { // need to mask out a bunch of entries in the last page
                    const int length = args.common.length - args.common.start_pos - args.iter*NUM_ROWS;
                    right_fill(att_block_fp32, att_block_fp32, length, -9999999999.f);
                }

                row_max(local_max_vec, att_block_fp32, local_max_vec);
                mul(max_vec_scaled, local_max_vec, SOFTMAX_TEMPERATURE);

                mul(att_block_fp32, att_block_fp32, SOFTMAX_TEMPERATURE);
                sub_row(att_block_fp32, att_block_fp32, max_vec_scaled);
                
                exp2(att_block_fp32, att_block_fp32);
                
                sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
                exp2(max_vec_last_scaled, max_vec_last_scaled);
                // warpgroup::store(args.scratch.max_vec, max_vec_last_scaled);
                
                mul(local_norm_vec, local_norm_vec, max_vec_last_scaled);
                row_sum(local_norm_vec, att_block_fp32, local_norm_vec);
                // warpgroup::store(args.scratch.att_block, att_block_fp32);
                // arrive(cons_barrier);

                mul_row(args.state.o, args.state.o, max_vec_last_scaled); // normalize o_reg before mma

                warpgroup::mma_AB(args.state.o, args.scratch.att_block, args.input.vcache);

                copy(args.state.max_vec, local_max_vec);
                copy(args.state.norm_vec, local_norm_vec);

                warpgroup::mma_async_wait();

                arrive_and_wait(cons_barrier);
                if(warpgroup::groupid() == 0) { arrive(args.inputs_finished, WARPGROUP_WARPS*2); }
            }
            else {
                arrive_and_wait(cons_barrier);
            }
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            if(args.iter >= args.num_iters-2) internal_compute<true>(args);
            else internal_compute<false>(args);
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            col_vec<rt_fl<16, kcache_tile::rows>> local_max_vec, local_norm_vec;

            copy(local_norm_vec, args.state.norm_vec);
            copy(local_max_vec,  args.state.max_vec);

            if(group<8>::laneid() == 0) args.timings[62] = clock64(); // Start of store out.

            // if (warpgroup::groupid() == 0) warpgroup::store(args.scratch.norm_vec, local_norm_vec);
            // group<8>::sync(10);
            // if (warpgroup::groupid() == 1) warpgroup::load(local_norm_vec, args.scratch.norm_vec);
            if(warpgroup::groupid() == 0) {
                div_row(args.state.o, args.state.o, local_norm_vec);
                if(args.common.dst.batch_idx >= 0) { // batch is meaningful
                    auto &o_smem = reinterpret_cast<st_bf<16, 128>&>(args.finish.o[warpgroup::warpid()]);
                    store(o_smem, args.state.o);
                    __syncwarp();
                    tma::store_async<1, cache_policy::EVICT_FIRST>(args.globals.O, o_smem, {args.common.dst.batch_idx, args.common.dst.seq_idx+warpgroup::warpid(), args.common.head, 0});
                }
                else { // write out directly to O scratch, without going through smem
                    if(warpgroup::groupid() == 0) {
                        mul(local_max_vec, local_max_vec, args.globals.Softmax_scale * 1.44269504089f);
                        log2(local_norm_vec, local_norm_vec);
                        add(local_norm_vec, local_norm_vec, local_max_vec); // l_vec = log2(norm_vec) + max_vec
                        store(args.finish.lvec[warpgroup::warpid()], local_norm_vec);
                        __syncwarp();
                        tma::store_async<cache_policy::EVICT_LAST>(args.globals.Lvec_scratch, args.finish.lvec[warpgroup::warpid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx+warpgroup::warpid(), args.common.head, 0});
                    }

                    store(args.finish.o[warpgroup::warpid()], args.state.o);
                    __syncwarp();
                    tma::store_async<1, cache_policy::EVICT_LAST>(args.globals.O_scratch, args.finish.o[warpgroup::warpid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx+warpgroup::warpid(), args.common.head, 0});
                }
                tma::store_async_wait(); // not just read wait
                group<8>::sync(10);
                if(args.common.dst.batch_idx < 0) {
                    if(group<8>::laneid() < 4 && args.common.dst.seq_idx + group<8>::laneid() < args.globals.O_scratch.depth()) {
                        // Todo: this can probably replaced by a st.async, which may prevent an expensive wait on the final finish barrier.
                        args.globals.semaphore[{-args.common.dst.batch_idx-1, args.common.dst.seq_idx + group<8>::laneid()}] = args.globals.tic;
                    }
                }
                if(warpgroup::laneid() == 0) arrive(args.finish_finished, WARPGROUP_WARPS); // done!
                else if(group<8>::laneid() == 32) args.timings[63] = clock64();
            }
        }
    };
};
struct reduction_layout {
    using globals = config::globals;
    struct input_block   { st_fl<16, 32> o[4]; sv_fl<16> lvec; sv_fl<16> padding[15]; };
    struct scratch_block { st_fl<16, 32> o[4]; sv_fl<16> lvec; semaphore producer_block; }; // used both for setup load and finish store
    struct common_state {
        int uid;
        // int num_iters; // same as the number of active load_uid's, marked here for instruction clarity but we just use args.num_iters instead.
        location dst; // again, negative batch means we're writing to O scratch, seq_idx is consistent
        int head;
        int src_uid;
    };
    struct consumer_state {
        rt_fl<16, 32> o;
        col_vec<rt_fl<16, kcache_tile::rows>> lvec;
    };
};
struct reduction_template {
    using config = config;
    using layout = reduction_layout;
    static constexpr int opcode = 2;
    static constexpr int INPUT_PIPE_STAGES = 4;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if(group<12>::laneid() == 0) args.timings[0] = clock64();
        args.common.uid     =  args.instruction[1];
        args.num_iters      =  args.instruction[2];
        args.common.dst     = {args.instruction[3],
                               args.instruction[4]};
        args.common.head    =  args.instruction[5];
        args.common.src_uid =  args.instruction[6];
        if(warpid() == 0) init_semaphore(args.scratch.producer_block, 0, 1);
        group<12>::sync(11);
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                // spinloop until we're ready
                int load_uid = args.instruction[7+args.iter];
                if(laneid() == 0) while(*(volatile int*)&args.globals.semaphore[{load_uid, args.common.dst.seq_idx}] != args.globals.tic) {}
                __syncwarp();
                if(args.iter > 0) wait(args.scratch.producer_block, 0);
                if(laneid() == 0 && args.iter < 24) args.timings[32+args.iter] = clock64();
                // next page we need to load?
                tma::expect(args.inputs_arrived, args.input.o, args.input.lvec);
                #pragma unroll
                for(int i = 0; i < 8; i++) {
                    tma::load_async<1, cache_policy::EVICT_FIRST>(args.input.o[i], args.globals.O_scratch, {load_uid, args.common.dst.seq_idx, args.common.head, i}, args.inputs_arrived);
                }
                tma::load_async(args.input.lvec, args.globals.Lvec_scratch, {load_uid, args.common.dst.seq_idx, args.common.head}, args.inputs_arrived);
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
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
            if(group<8>::laneid() == 0) args.timings[1] = clock64();
            load_async<1, false>(args.scratch.o[group<8>::warpid()], args.globals.O_scratch, {args.common.src_uid, args.common.dst.seq_idx, args.common.head, group<8>::warpid()});
            if(warpid() == 0) {
                load_async(args.scratch.lvec, args.globals.Lvec_scratch, {args.common.src_uid, args.common.dst.seq_idx, args.common.head});
            }
            load_async_wait();
            __syncwarp();
            load(args.state.o, args.scratch.o[group<8>::warpid()]);
            group<8>::sync(11); // we use this to also stall the producer until the consumer is ready.
            if(group<8>::laneid() == 0) arrive(args.scratch.producer_block);
            if(group<8>::laneid() == 0) args.timings[2] = clock64();
            load(args.state.lvec, args.scratch.lvec);
        }

        __device__ static inline void compute(consumer_compute_args<layout> args) {
            if(group<8>::laneid() == 0 && args.iter < 24) args.timings[8+args.iter] = clock64();
            if(warpgroup::groupid() == 0) {
                col_vec<rt_fl<16, kcache_tile::rows>> lvec, max_lvec, sum_lvec;
                rt_fl<16, 32> o;
                load(o, args.input.o[group<8>::warpid()]);
                load(lvec, args.input.lvec);
                __syncwarp();
                if(laneid() == 0) arrive(args.inputs_finished, 2); // done!
                max(max_lvec, args.state.lvec, lvec);
                sub(args.state.lvec, args.state.lvec, max_lvec);
                sub(lvec, lvec, max_lvec);
                exp2(args.state.lvec, args.state.lvec);
                exp2(lvec, lvec);
                add(sum_lvec, args.state.lvec, lvec);
                div(args.state.lvec, args.state.lvec, sum_lvec);
                div(lvec, lvec, sum_lvec);
                mul_row(args.state.o, args.state.o, args.state.lvec);
                mul_row(o, o, lvec);
                add(args.state.o, args.state.o, o);
                log2(sum_lvec, sum_lvec);
                add(args.state.lvec, sum_lvec, max_lvec);
            }
        }

        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if(group<8>::laneid() == 0) args.timings[62] = clock64();
            if(args.common.dst.batch_idx >= 0) {
                auto &o_smem = reinterpret_cast<st_bf<16, 32>&>(args.scratch.o[group<8>::warpid()]);
                store(o_smem, args.state.o);
                __syncwarp();
                tma::store_async<1, cache_policy::EVICT_FIRST>(args.globals.O, o_smem, {args.common.dst.batch_idx, args.common.dst.seq_idx, args.common.head, group<8>::warpid()});
            }
            else {
                store(args.scratch.o[group<8>::warpid()], args.state.o);
                if(group<8>::warpid() == 0) store(args.scratch.lvec, args.state.lvec);
                __syncwarp();
                tma::store_async<1, cache_policy::EVICT_LAST>(args.globals.O_scratch, args.scratch.o[group<8>::warpid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx, args.common.head, group<8>::warpid()});
                if(group<8>::warpid() == 0) tma::store_async<cache_policy::EVICT_LAST>(args.globals.Lvec_scratch, args.scratch.lvec, {-args.common.dst.batch_idx-1, args.common.dst.seq_idx, 0});
            }
            tma::store_async_wait();
            if(warpid() == 0) invalidate_semaphore(args.scratch.producer_block);
            group<8>::sync(11);
            // Increment the semaphore for the next stage, if this is not the last one.
            if(args.common.dst.batch_idx < 0) {
                if(group<8>::laneid() == 0) {
                    args.globals.semaphore[{-args.common.dst.batch_idx-1, args.common.dst.seq_idx}] = args.globals.tic;
                }
            }
            if(warpgroup::laneid() == 0) arrive(args.finish_finished, WARPGROUP_WARPS); // done!
            else if(group<8>::laneid() == 32) args.timings[63] = clock64();
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

constexpr float REDUCTION_STARTUP_TIME     = 2.0f;   // Startup time for reduction operations
constexpr float REDUCTION_WRITEOUT_TIME    = 1.0f;   // Writeout time for reduction operations
constexpr float REDUCTION_PRODUCER_LATENCY = 1.0f;   // Latency between a producer load and when the consumer can access it.
constexpr float REDUCTION_COST_PER_STEP = 0.4f;      // Cost per reduction step

constexpr float SYNCHRONIZATION_COST = 0.5f;         // Synchronization cost between dependent operations

float get_quality(const std::vector<float>& next_times_input, int num_processors, int num_tokens, int seq_length) {
    int num_partial_steps = (seq_length + 31) / 32;
    
    if (next_times_input.size() * num_tokens > num_processors) {
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

PYBIND11_MODULE(mha_decode, m) {
    m.doc() = "mha_decode python module";
    kittens::py::bind_kernel<interpreter::kernel<config, partial_template, reduction_template>>(m, "mha_decode",
        &config::globals::instructions,
        &config::globals::Q,
        &config::globals::K_cache,
        &config::globals::V_cache,
        &config::globals::Table,
        &config::globals::O,
        &config::globals::O_scratch,
        &config::globals::Lvec_scratch,
        &config::globals::semaphore,
        &config::globals::Softmax_scale,
        &config::globals::tic,
        &config::globals::timings
    );
    m.def("__get_quality__", &get_quality, 
          "An internal utility function for generating efficient schedules.",
          pybind11::arg("next_times"), 
          pybind11::arg("num_processors"), 
          pybind11::arg("num_tokens"), 
          pybind11::arg("seq_length"));
}