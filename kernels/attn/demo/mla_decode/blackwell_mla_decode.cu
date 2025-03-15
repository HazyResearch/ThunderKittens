#define KITTENS_TIMINGS

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

static constexpr int QKRot_D = 64, QVO_D = 512, NUM_ROWS = 128, PAGE_SIZE = 256;
using q_tile              = st_bf<128, 64>;
using k_tile              = st_bf<128, 64>;
using v_tile              = st_bf<128, 128>; // w i d e
using att_tile            = st_bf<128, 128>;
using q_rot_global        = kittens::gl<bf16, -1, -1, -1, QKRot_D, st_bf<16, 64>>; // B * R * H * D_QKRot_D
using q_global            = kittens::gl<bf16, -1, -1, -1, QVO_D, st_bf<16, 64>>; // B * R * H * D_QVO_D
using k_rot_cache_global  = kittens::gl<bf16, 1, -1, PAGE_SIZE, QKRot_D, k_tile>; // 1 * #page * pagesize * QKRot_D
using kv_cache_global     = kittens::gl<bf16, 1, -1, PAGE_SIZE, QVO_D, k_tile, v_tile>; // 1 * #page * pagesize * QVO_D
using instructions_global = kittens::gl<int, 1, -1, -1, 32>;
using table_global        = kittens::gl<int, 1, 1, -1, -1>; // B * (max # pages)
using o_tile              = st_bf<128, QVO_D>;
using o_tile_fl           = st_fl<16, QVO_D>;
using o_global            = kittens::gl<bf16, -1, -1, -1, QVO_D, st_bf<16, 128>, st_bf<16, QVO_D/8>>; // B * NEWTOKENS * H * D_VO

template<int Q_HEADS=16>
using o_scratch_global    = kittens::gl<float, -1, -1, Q_HEADS, QVO_D, st_fl<16,128>, st_fl<16, QVO_D/8>>; // For partial O's

template<int Q_HEADS=16>
using lvec_scratch_global = kittens::gl<float,  1, -1, -1, Q_HEADS, sv_fl<16>>; // For partial O's
using semaphore_global    = kittens::gl<int,    1,  1,  -1, -1>;            // 1 * 1 * uid * NEWTOKENS

template<int Q_HEADS=16>
struct config {
    struct globals {
        using instructions_global = instructions_global;
        instructions_global instructions;
        q_rot_global Q_rot;
        q_global Q;
        k_rot_cache_global K_rot_cache;
        kv_cache_global KV_cache;
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
        dim3 grid()  { return dim3(instructions.depth()); } // but should probably be 148 unless debugging!
        dim3 block() { return dim3((8+4)*WARP_THREADS); }
    };
};

struct location {
    int batch_idx; // batch_idx >=0, otherwise it's the negative index, minus one, into scratch
    int seq_idx;
};
template<int Q_HEADS=16>
struct partial_layout {
    using globals = config<Q_HEADS>::globals;
    struct input_block {
        q_tile q;
        k_tile k;
    };
    struct v_input_block { v_tile v; };
    static_assert(sizeof(input_block) == sizeof(v_input_block), "input_block and v_input_block must be the same size");
    struct scratch_block { att_tile att_block; sv_fl<128> max_vec, norm_vec; semaphore mma_sem; };
    struct finish_block { st_fl<16, 128> o[8][2]; sv_fl<16> lvec[8][2][2]; }; // Last 2 on Lvec is padding for 128-byte alignments
    struct common_state {
        int uid;
        location dst;
        int q_batch_idx;
        int q_seq_idx;
        int start_pos; // MUST BE A MULTIPLE OF PAGE_SIZE
        int end_pos; // One past the last position to load
        int length; // the length of the overall sequence in question
    };
    // struct producer_state {
    //     int iter_idx, intra_idx; // / 13, % 13
    // };
    using o_tt_t              = tt<float, 128, 128>;
    using att_block_tt_t      = tt<float, 128, k_tile::rows>;
    struct consumer_state {
        rt_fl<16, 128>                        o_chunk;
        att_block_tt_t                        att_block_tt;
        col_vec<rt_fl<16, k_tile::rows>> max_vec, norm_vec;
    };
};
template<int Q_HEADS=16>
struct partial_template {
    using config = config<Q_HEADS>;
    using layout = partial_layout<Q_HEADS>;
    static constexpr int opcode = 1;
    static constexpr int INPUT_PIPE_STAGES = 5;
    using consumer_group = group<8>;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        #ifdef KITTENS_TIMINGS
        if(group<12>::laneid() == 0) args.timings[0] = clock64();
    #endif
        args.common.uid         =  args.instruction[1];
        args.common.dst         = {args.instruction[2],
                                   args.instruction[3]};
        args.common.q_batch_idx =  args.instruction[4];
        args.common.q_seq_idx   =  args.instruction[5];
        args.common.start_pos   =  args.instruction[6];
        args.common.end_pos     =  args.instruction[7];
        args.common.length      =  args.instruction[8];
        args.num_iters          = ((args.common.end_pos - args.common.start_pos + NUM_ROWS - 1) / NUM_ROWS) * 13; // 9 iters x 64 headdim for K, 4 iters x 128 headdim for V
        args.common.length    -= (args.globals.Q.depth() - (args.common.q_seq_idx + consumer_group::warpid()) - 1); // adjust for the causal mask
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static inline void load(producer_load_args<layout> args) {
            int iter_idx = args.iter / 13;
            int intra_idx = args.iter % 13;
        #ifdef KITTENS_TIMINGS
            if(warpgroup::laneid() == 0) args.timings[32+args.iter] = clock64();
        #endif
            int pos = args.common.start_pos + NUM_ROWS*iter_idx;
            int within_page_idx = (pos % PAGE_SIZE) / NUM_ROWS;
            int next_page_id = args.globals.Table[coord<>{args.common.q_batch_idx, pos/PAGE_SIZE}];
            // next page we need to load?
            if(warpgroup::warpid() == 0) {
                if(intra_idx < 9) {
                    tma::expect(args.inputs_arrived, args.input);
                    if(intra_idx < 8) {
                        tma::load_async<dim::ROW, cache_policy::NORMAL>(args.input.k, args.globals.KV_cache, {next_page_id, within_page_idx, intra_idx}, args.inputs_arrived);
                    }
                    else {
                        tma::load_async<dim::ROW, cache_policy::NORMAL>(args.input.k, args.globals.K_rot_cache, {next_page_id, within_page_idx, 0}, args.inputs_arrived);
                    }
                    // Since we are only load 64 rows of Q at a time, TMA is actually fair game here.
                    st_bf<16, 64> (&q_arr)[8] = reinterpret_cast<st_bf<16, 64>(&)[8]>(args.input.q);
                    if(intra_idx < 8) {
                        #pragma unroll
                        for(int i = 0; i < 8; i++) {
                            tma::load_async(q_arr[i], args.globals.Q, {args.common.q_batch_idx, args.common.q_seq_idx + i, 0, intra_idx}, args.inputs_arrived);
                        }
                    }
                    else {
                        #pragma unroll
                        for(int i = 0; i < 8; i++) {
                            tma::load_async(q_arr[i], args.globals.Q_rot, {args.common.q_batch_idx, args.common.q_seq_idx + i, 0, 0}, args.inputs_arrived);
                        }
                    }
                }
                else {
                    typename layout::v_input_block &v_input = reinterpret_cast<typename layout::v_input_block&>(args.input);
                    tma::expect(args.inputs_arrived, v_input);
                    tma::load_async<dim::ROW, cache_policy::NORMAL>(v_input.v, args.globals.KV_cache, {next_page_id, within_page_idx, intra_idx-9}, args.inputs_arrived);
                }
            }
            warpgroup::sync(5);
            if(warpgroup::laneid() == 0) arrive(args.inputs_arrived, 3);
        }
    };
    struct consumer {
        template<typename T> __device__ static inline typename layout::o_tt_t get_o(T &allocator, int col_idx) {
            return allocator.template allocate<typename layout::o_tt_t>(128*col_idx);
        }
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            #ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[1] = clock64();
            #endif
            zero(args.state.norm_vec);
            if(args.num_iters > 0) neg_infty(args.state.max_vec);
            else { one(args.state.max_vec); mul(args.state.max_vec, args.state.max_vec, -999999.f); }

            if(warpid() == 0) {
                init_semaphore(args.scratch.mma_sem, 1);
            }

            // Allocate tensor memory
            args.state.att_block_tt = args.tensor_alloc.template allocate<typename layout::att_block_tt_t>(0);

            rt_fl<16, 128> z;
            z = 0.f;
            typename layout::o_tt_t o0 = get_o(args.tensor_alloc, 0);
            typename layout::o_tt_t o1 = get_o(args.tensor_alloc, 1);
            typename layout::o_tt_t o2 = get_o(args.tensor_alloc, 2);
            typename layout::o_tt_t o3 = get_o(args.tensor_alloc, 3);
            consumer_group::store_async(o0, z);
            consumer_group::store_async(o1, z);
            consumer_group::store_async(o2, z);
            consumer_group::store_async(o3, z);
            tm_store_wait();
            consumer_group::sync(10);
            #ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[2] = clock64();
            #endif
        }
        __device__ static inline void initial_qk(consumer_compute_args<layout> args) { // intra_idx == 0
            if(warpid() == 0) {
                mm_ABt(args.state.att_block_tt, args.input.q, args.input.k, args.inputs_finished);
            }
            else if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static inline void main_qk(consumer_compute_args<layout> args) { // intra_idx in 1...7
            if(warpid() == 0) {
                mma_ABt(args.state.att_block_tt, args.input.q, args.input.k, args.inputs_finished);
            }
            else if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static inline void final_qk_softmax(consumer_compute_args<layout> args) { // intra_idx == 8
            const float SOFTMAX_TEMPERATURE = args.globals.Softmax_scale * 1.44269504089f;

            col_vec<rt_fl<16, k_tile::rows>> local_max_vec, local_norm_vec;
            col_vec<rt_fl<16, k_tile::rows>> max_vec_last_scaled, max_vec_scaled;

            if(warpid() == 0) {
                mma_ABt(args.state.att_block_tt, args.input.q, args.input.k, args.scratch.mma_sem);
            }
            consumer_group::sync(10);

            copy(local_max_vec,  args.state.max_vec);
            copy(local_norm_vec, args.state.norm_vec);

            mul(max_vec_last_scaled, local_max_vec, SOFTMAX_TEMPERATURE);

            wait(args.scratch.mma_sem, 0);
            if(consumer_group::laneid() == 0) arrive(args.inputs_finished, 8);

            // softmax

            rt_fl<16, k_tile::rows> att_block_fp32;
            consumer_group::load_async(att_block_fp32, args.state.att_block_tt);
            const int length = args.common.length - args.common.start_pos - (args.iter/13)*NUM_ROWS;
            right_fill(att_block_fp32, att_block_fp32, length, -9999999999.f);

            row_max(local_max_vec, att_block_fp32, local_max_vec);
            mul(max_vec_scaled, local_max_vec, SOFTMAX_TEMPERATURE);

            mul(att_block_fp32, att_block_fp32, SOFTMAX_TEMPERATURE);
            sub_row(att_block_fp32, att_block_fp32, max_vec_scaled);
            
            exp2(att_block_fp32, att_block_fp32);
            
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last_scaled, max_vec_last_scaled);

            consumer_group::store(args.scratch.att_block, att_block_fp32); // store to shared memory
                
            mul(local_norm_vec, local_norm_vec, max_vec_last_scaled);
            row_sum(local_norm_vec, att_block_fp32, local_norm_vec);

            // Now we need to normalize O
            typename layout::o_tt_t o0 = get_o(args.tensor_alloc, 0);
            mul_row(args.state.o_chunk, args.state.o_chunk, max_vec_last_scaled);
            consumer_group::store_async(o0, args.state.o_chunk);
            typename layout::o_tt_t o1 = get_o(args.tensor_alloc, 1);
            consumer_group::load_async(args.state.o_chunk, o1);
            mul_row(args.state.o_chunk, args.state.o_chunk, max_vec_last_scaled);
            consumer_group::store_async(o1, args.state.o_chunk);
            typename layout::o_tt_t o2 = get_o(args.tensor_alloc, 2);
            consumer_group::load_async(args.state.o_chunk, o2);
            mul_row(args.state.o_chunk, args.state.o_chunk, max_vec_last_scaled);
            consumer_group::store_async(o2, args.state.o_chunk);
            typename layout::o_tt_t o3 = get_o(args.tensor_alloc, 3);
            consumer_group::load_async(args.state.o_chunk, o3);
            mul_row(args.state.o_chunk, args.state.o_chunk, max_vec_last_scaled);
            consumer_group::store_async(o3, args.state.o_chunk);


            // decltype(args.state.o_chunk) o_chunk_2;
            // consumer_group::load_async(o_chunk_2, o1); // >>> 1
            // mul_row(args.state.o_chunk, args.state.o_chunk, max_vec_last_scaled);
            // consumer_group::store_async(o0, args.state.o_chunk); // <<< 0
            // consumer_group::load_async(args.state.o_chunk, o2); // >>> 2
            // mul_row(o_chunk_2, o_chunk_2, max_vec_last_scaled);
            // consumer_group::store_async(o1, o_chunk_2); // <<< 1
            // consumer_group::load_async(o_chunk_2, o3); // >>> 3
            // mul_row(args.state.o_chunk, args.state.o_chunk, max_vec_last_scaled);
            // consumer_group::store_async(o2, args.state.o_chunk); // <<< 2
            // // no load for get_o(0) until after the matmul.
            // mul_row(o_chunk_2, o_chunk_2, max_vec_last_scaled);
            // consumer_group::store_async(o3, o_chunk_2); // <<< 3

            copy(args.state.max_vec, local_max_vec);
            copy(args.state.norm_vec, local_norm_vec);

            tm_store_wait();
            consumer_group::sync(10); // tensor memory ready, shared memory ready, we can now rip av matmuls!

            // if(consumer_group::laneid() == 0) {
            //     for(int i = 0; i < 128*128; i++) {
            //         printf("%f ", __bfloat162float(args.scratch.att_block.data[i]));
            //     }
            //     printf("\n\n\n\n\n");
            // }
        }
        __device__ static inline void av(consumer_compute_args<layout> args) { // intra_idx in 9...12
            int intra_idx = args.iter % 13;
            typename layout::v_input_block &v_input = reinterpret_cast<typename layout::v_input_block&>(args.input);
            auto ot = get_o(args.tensor_alloc, intra_idx-9);
            if(intra_idx < 12) {
                if(warpid() == 0) {
                    mma_AB(ot, args.scratch.att_block, v_input.v, args.inputs_finished);
                }
                else if(laneid() == 0) arrive(args.inputs_finished);
            }
            else {
                if(warpid() == 0) {
                    mma_AB(ot, args.scratch.att_block, v_input.v, args.scratch.mma_sem);
                }
                consumer_group::sync(10);
                wait(args.scratch.mma_sem, 1);
                if(consumer_group::laneid() == 0) arrive(args.inputs_finished, 8);
                consumer_group::load_async(args.state.o_chunk, get_o(args.tensor_alloc, 0));
                tm_load_wait();
                consumer_group::sync(10);
            }
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0 && args.iter < 24) args.timings[8+args.iter] = clock64();
#endif
            int intra_idx = args.iter % 13;
            if(intra_idx == 0) initial_qk(args);
            else if(intra_idx > 0 && intra_idx < 8) main_qk(args);
            else if(intra_idx == 8) final_qk_softmax(args);
            else av(args);
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
             // if(consumer_group::laneid() == 0) printf("starting finish\n");
            col_vec<rt_fl<16, k_tile::rows>> local_max_vec, local_norm_vec;

            copy(local_norm_vec, args.state.norm_vec);
            copy(local_max_vec, args.state.max_vec);

#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[62] = clock64(); // Start of store out.
#endif

            #pragma unroll
            for(int i = 0; i < 4; i++) {
                consumer_group::load_async(args.state.o_chunk, get_o(args.tensor_alloc, i));
                div_row(args.state.o_chunk, args.state.o_chunk, local_norm_vec);

                // for(int j = 0; j < args.state.o_chunk.height; j++) {
                //     for(int k = 0; k < args.state.o_chunk.width; k++) {
                //         for(int l = 0; l < 4; l++) {
                //             printf("%f ", __bfloat162float(args.state.o_chunk.tiles[j][k].data[l].x));
                //             printf("%f ", __bfloat162float(args.state.o_chunk.tiles[j][k].data[l].y));
                //         }
                //     }
                // }

                if(args.common.dst.batch_idx >= 0) { // batch is meaningful
                    auto &o_smem = reinterpret_cast<st_bf<16, 128>&>(args.finish.o[warpid()][i%2]);
                    store(o_smem, args.state.o_chunk);
                    __syncwarp();
                    tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(args.globals.O, o_smem, {args.common.dst.batch_idx, args.common.dst.seq_idx+warpid(), 0, i});
                }
                else { // write out directly to O scratch, without going through smem
                    mul(local_max_vec, local_max_vec, args.globals.Softmax_scale * 1.44269504089f);
                    log2(local_norm_vec, local_norm_vec);
                    add(local_norm_vec, local_norm_vec, local_max_vec); // l_vec = log2(norm_vec) + max_vec
                    store(args.finish.lvec[warpid()][i%2][0], local_norm_vec);
                    __syncwarp();
                    tma::store_async<cache_policy::EVICT_LAST>(args.globals.Lvec_scratch, args.finish.lvec[warpid()][i%2][0], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx+warpid(), 0});
                    store(args.finish.o[warpid()][i%2], args.state.o_chunk);
                    __syncwarp();
                    tma::store_async<dim::ROW, cache_policy::EVICT_LAST>(args.globals.O_scratch, args.finish.o[warpid()][i%2], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx+warpid(), 0, i});
                }
                tma::store_async_read_wait<1>();
            }
             // if(consumer_group::laneid() == 0) printf("pre invalidate semaphore\n");
            if(warpid() == 0) invalidate_semaphore(args.scratch.mma_sem); // invalidate the semaphore for the next iteration.
             // if(consumer_group::laneid() == 0) printf("post invalidate semaphore\n");
            tma::store_async_wait(); // not just read wait
             // if(consumer_group::laneid() == 0) printf("post store_async_wait\n");
            asm volatile("fence.sc.cta;\n"); // Can't reorder across this boundary
            group<8>::sync(10);
            if(args.common.dst.batch_idx < 0) {
                if(group<8>::laneid() < 4 && args.common.dst.seq_idx + group<8>::laneid() < args.globals.O_scratch.depth()) {
                    // Todo: this can probably replaced by a st.async, which may prevent an expensive wait on the final finish barrier.
                    args.globals.semaphore[{-args.common.dst.batch_idx-1, args.common.dst.seq_idx + group<8>::laneid()}] = args.globals.tic;
                    // For blackwell
                    // asm volatile(
                    //     "st.async.global.b32 [%0], %1;\n"
                    // ::  "l"(&args.globals.semaphore[{-args.common.dst.batch_idx-1, args.common.dst.seq_idx + group<8>::laneid()}]), "r"(args.globals.tic)
                    // :   "memory"
                    // );
                }
            }
            if(consumer_group::laneid() == 0) arrive(args.finish_finished, 8); // done!
#ifdef KITTENS_TIMINGS
            else if(group<8>::laneid() == 32) args.timings[63] = clock64();
#endif
        }
    };
};

template<int Q_HEADS=16>
struct reduction_layout {
    using globals = config<Q_HEADS>::globals;
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
        col_vec<rt_fl<16, k_tile::rows>> lvec;
    };
};

template<int Q_HEADS=16>
struct reduction_template {
    using config = config<Q_HEADS>;
    using layout = reduction_layout<Q_HEADS>;
    static constexpr int opcode = 2;
    static constexpr int INPUT_PIPE_STAGES = 4;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
#ifdef KITTENS_TIMINGS
        if(group<12>::laneid() == 0) args.timings[0] = clock64();
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
                if(laneid() == 0 && args.iter < 24) args.timings[32+args.iter] = clock64();
#endif
                // next page we need to load?
                tma::expect(args.inputs_arrived, args.input.o, args.input.lvec);
                #pragma unroll
                for(int i = 0; i < 8; i++) {
                    tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(args.input.o[i], args.globals.O_scratch, {load_uid, args.common.dst.seq_idx, 0, i}, args.inputs_arrived);
                }
                tma::load_async<cache_policy::EVICT_FIRST>(args.input.lvec, args.globals.Lvec_scratch, {load_uid, args.common.dst.seq_idx, 0}, args.inputs_arrived);
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
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[1] = clock64();
#endif
            load_async(args.scratch.o[group<8>::warpid()], args.globals.O_scratch, {args.common.src_uid, args.common.dst.seq_idx, 0, group<8>::warpid()});
            if(warpid() == 0) {
                load_async(args.scratch.lvec, args.globals.Lvec_scratch, {args.common.src_uid, args.common.dst.seq_idx, 0});
            }
            load_async_wait();
            __syncwarp();
            load(args.state.o, args.scratch.o[group<8>::warpid()]);
            group<8>::sync(11); // we use this to also stall the producer until the consumer is ready.
            // group<12>::sync(9);
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[2] = clock64();
#endif
            load(args.state.lvec, args.scratch.lvec);
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0 && args.iter < 24) args.timings[8+args.iter] = clock64();
#endif
            col_vec<rt_fl<16, k_tile::rows>> lvec, max_lvec, sum_lvec;
            rt_fl<16, QVO_D / 8> o;
            load(o, args.input.o[group<8>::warpid()]);
            load(lvec, args.input.lvec);
            __syncwarp();
            if(laneid() == 0) arrive(args.inputs_finished); // done!
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
        __device__ static inline void finish(consumer_finish_args<layout> args) {
#ifdef KITTENS_TIMINGS
            if(group<8>::laneid() == 0) args.timings[62] = clock64();
#endif
            if(args.common.dst.batch_idx >= 0) {
                auto &o_smem = reinterpret_cast<st_bf<16, QVO_D/8>&>(args.scratch.o[group<8>::warpid()]);
                store(o_smem, args.state.o);
                __syncwarp();
                tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(args.globals.O, o_smem, {args.common.dst.batch_idx, args.common.dst.seq_idx, 0, group<8>::warpid()});
            }
            else {
                store(args.scratch.o[group<8>::warpid()], args.state.o);
                if(group<8>::warpid() == 0) store(args.scratch.lvec, args.state.lvec);
                __syncwarp();
                tma::store_async<dim::ROW, cache_policy::EVICT_LAST>(args.globals.O_scratch, args.scratch.o[group<8>::warpid()], {-args.common.dst.batch_idx-1, args.common.dst.seq_idx, 0, group<8>::warpid()});
                if(group<8>::warpid() == 0) tma::store_async<cache_policy::EVICT_LAST>(args.globals.Lvec_scratch, args.scratch.lvec, {-args.common.dst.batch_idx-1, args.common.dst.seq_idx, 0});
            }
            tma::store_async_wait();
            group<8>::sync(11);
            // Increment the semaphore for the next stage, if this is not the last one.
            if(args.common.dst.batch_idx < 0) {
                if(group<8>::laneid() == 0) {
                    args.globals.semaphore[{-args.common.dst.batch_idx-1, args.common.dst.seq_idx}] = args.globals.tic;
                }
            }
            if(warpgroup::laneid() == 0) arrive(args.finish_finished, WARPGROUP_WARPS); // done!
#ifdef KITTENS_TIMINGS
            else if(group<8>::laneid() == 32) args.timings[63] = clock64();
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
    int num_partial_steps = (seq_length + 127) / 128;

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
    kittens::py::bind_kernel<interpreter::kernel<config<16>, partial_template<16>, reduction_template<16>>>(m, "mla_decode",
        &config<16>::globals::instructions,
        &config<16>::globals::Q_rot,
        &config<16>::globals::Q,
        &config<16>::globals::K_rot_cache,
        &config<16>::globals::KV_cache,
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
//     kittens::py::bind_kernel<interpreter::kernel<config<8>, partial_template<8>, reduction_template<8>>>(m, "mla_decode_8_heads",
//         &config<8>::globals::instructions,
//         &config<8>::globals::Q_rot,
//         &config<8>::globals::Q,
//         &config<8>::globals::K_rot_cache,
//         &config<8>::globals::KV_cache,
//         &config<8>::globals::Table,
//         &config<8>::globals::O,
//         &config<8>::globals::O_scratch,
//         &config<8>::globals::Lvec_scratch,
//         &config<8>::globals::semaphore,
//         &config<8>::globals::Softmax_scale,
//         &config<8>::globals::tic
// #ifdef KITTENS_TIMINGS
//         , &config<8>::globals::timings
// #endif
//     );
    m.def("__get_quality__", &get_quality, 
        "An internal utility function for generating efficient schedules.",
        pybind11::arg("next_times"), 
        pybind11::arg("num_processors"), 
        pybind11::arg("num_tokens"), 
        pybind11::arg("seq_length")
    );
}