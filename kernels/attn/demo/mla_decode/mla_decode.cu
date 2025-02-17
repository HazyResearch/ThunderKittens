#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
union location {
    struct {
        int type; // -1 means writeout, -2 means no writeout
        int uid;
    } uniform;
    struct {
        int batch_idx; // batch_idx >=0
        int seq_idx;
    } tensor;
};
struct mla_decode_layout {
    static constexpr int QK_D = 576, VO_D = 512, VO_Dd2 = VO_D/2, NUM_ROWS = 32, PAGE_SIZE = 256, ITERS_PER_PAGE = PAGE_SIZE / NUM_ROWS;
    using q_tile              = st_bf<64, QK_D>;
    using q_global            = kittens::gl<bf16, 1, -1, -1, QK_D>; // 1 * B * (R * H) * D_QK
    using cache_tile          = st_bf<NUM_ROWS, QK_D>; using v_tile = st_bf<NUM_ROWS, VO_Dd2>; // we need the v_tile for later
    using cache_global        = kittens::gl<bf16, 1, -1, PAGE_SIZE, QK_D, cache_tile>; // 1 * #page * pagesize * QK_D
    using ops_global          = kittens::gl<bf16, 1, -1, -1, 8>;
    using table_global        = kittens::gl<int, 1, 1, -1, -1>; // B * (max # pages)
    using o_tile_d2           = st_bf<64, VO_Dd2>;
    using o_tile_fl           = st_fl<16, VO_D>;
    using o_global            = kittens::gl<bf16, 1, -1, -1,  VO_D, o_tile_d2>; // B * (R * H) * D_VO
    using o_scratch_global    = kittens::gl<float, 1, -1, 64, VO_D, o_tile_fl>; // For partial O's
    using lvec_scratch_global = kittens::gl<float, 1, 1, -1, 64, sv_fl<16>>; // For partial O's
    using semaphore_global    = kittens::gl<int, 1, 1, 1, -1>; // 1 * 1 * 1 * uid
    struct globals {
        ops_global Ops;
        q_global Q;
        cache_global Cache;
        table_global Table;
        o_global O;
        o_scratch_global O_scratch;
        lvec_scratch_global Lvec_scratch;
        semaphore_global Semaphore;
        const float softmax_scale;
        int dynamic_shared_memory() { return 224000; }
        dim3 grid()  { return dim3(148); } //dim3(Q.batch * ((Q.depth + 3) / 4)); }
        dim3 block() { return dim3((8+4)*WARP_THREADS); }
    };
    union input_block {
        struct partial {
            cache_tile c;
        } partial;
        struct reduction {
            o_tile_fl o[2];
            sv_fl<16> lvec[2]; // TODO: fix me to fix address alignment issues.
            sv_fl<16> padding[14];
        } reduction;
    };
    struct scratch_block  { 
        q_tile q;
        st_bf<64, cache_tile::rows> att_block;
        sv_fl<64> vec[2];
    };
    union common_state {
        enum opcode {
            op_stop = 0,
            op_partial = 1,
            op_reduction = 2,
        };
        struct { // Base layout
            uint32_t op, data[7];
        } raw;
        struct {
            opcode op;
            int uid;
            location dst;
            int q_batch_idx;
            int q_seq_idx;
            int length;
        } partial;
        struct {
            opcode op;
            location dst;
            int src_uid[2];
        } reduction;
    };
    union consumer_state {
        struct {
            col_vec<rt_fl<16, cache_tile::rows>> max_vec, norm_vec;
            col_vec<rt_fl<16, cache_tile::rows>> max_vec_last_scaled, max_vec_scaled;
            rt_fl<16, VO_Dd2> o;
        } partial;
        struct {
            col_vec<rt_fl<16, cache_tile::rows>> lvec[2];
            rt_fl<16, VO_D / 8> o[2];
        } reduction;
        __device__ inline consumer_state() : partial{} {} // Default constructor initializing partial member
    };
};
struct mla_decode_template {
    static constexpr int NUM_CONSUMER_WARPS = 8, NUM_WORKERS = NUM_CONSUMER_WARPS/4, INPUT_PIPE_STAGES = 2, PRODUCER_BARRIER_ARRIVALS = 1, CONSUMER_BARRIER_ARRIVALS = 2;
    using layout = mla_decode_layout;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if(args.task_iter == args.globals.Ops.rows) { // no remaining instructions to process
            args.num_iters = -1;
            return;
        }
        // if(threadIdx.x == 0) printf("block %d, task_iter %d, op: %d\n", blockIdx.x, args.task_iter, args.globals.Ops[kittens::coord<>{0, blockIdx.x, args.task_iter, 0}]);
        args.common.raw.op = args.globals.Ops[kittens::coord<>{0, blockIdx.x, args.task_iter, 0}];
        #pragma unroll
        for(int i = 0; i < 7; i++) {
            args.common.raw.data[i] = args.globals.Ops[kittens::coord<>{0, blockIdx.x, args.task_iter, i+1}];
        }
        // if(threadIdx.x == 0) printf("block %d, task_iter %d, AFTERLOAD op: %d\n", blockIdx.x, args.task_iter, args.common.raw.op);
        switch(args.common.raw.op) {
            case layout::common_state::opcode::op_stop:
                args.num_iters = -1;
                break;
            case layout::common_state::opcode::op_partial:
                // here, partial is the input cache tile
                args.num_iters = (args.common.partial.length + layout::NUM_ROWS - 1) / layout::NUM_ROWS;
                break;
            case layout::common_state::opcode::op_reduction:
                args.num_iters = 4; // must be split across 4 iters, due to shared memory limitations
                break;
            default:
                *(int*)0 = 0; // Unreachable
                break;
        }
        // If we are doing a reduction, we need to spinloop until we have confirmation that all the partial results have been written out.
        if(args.common.raw.op == layout::common_state::opcode::op_reduction) {
            if(threadIdx.x == 0) { // easier to have a single thread spin
                while(*(volatile int*)&args.globals.Semaphore[{args.common.reduction.src_uid[0]}] == 0) {} // note volatile, L1 is not guaranteed to be coherent.
                while(*(volatile int*)&args.globals.Semaphore[{args.common.reduction.src_uid[1]}] == 0) {}
            }
            group<12>::sync(11); // all warps must sync here.
        }
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            if(args.common.raw.op == layout::common_state::opcode::op_partial) {
                if(warpgroup::warpid() == 0) {
                    int global_load_idx = args.iter % layout::ITERS_PER_PAGE;
                    int next_page_id = args.globals.Table[coord<>{args.common.partial.uid, args.iter / layout::ITERS_PER_PAGE}];
                    // next page we need to load?
                    tma::expect(args.inputs_arrived, args.input.partial.c);
                    // cache shape is 1, # pages, page size, QK_D
                    tma::load_async(args.input.partial.c, args.globals.Cache, {0, next_page_id, global_load_idx, 0}, args.inputs_arrived);
                }
                warpgroup::sync(5);
                // int global_load_idx = args.iter % layout::ITERS_PER_PAGE;
                // int next_page_id = args.globals.Table[coord<>{args.common.partial.uid, args.iter / layout::ITERS_PER_PAGE}];
                // if (warpgroup::laneid() == 0) {
                //     printf("load %d, %d, %d\n", args.iter, next_page_id, global_load_idx);
                //     printf("cache shape: %d, %d\n", int(args.input.partial.c.rows), int(args.input.partial.c.cols));
                // }
            }
            else if(args.common.raw.op == layout::common_state::opcode::op_reduction) {
                if(warpgroup::warpid() == 0) {
                    // next page we need to load?
                    tma::expect(args.inputs_arrived, args.input.reduction.o[0], args.input.reduction.o[1], args.input.reduction.lvec[0], args.input.reduction.lvec[1]);
                    tma::load_async(args.input.reduction.o[0], args.globals.O_scratch, {args.common.reduction.src_uid[0], args.iter, 0}, args.inputs_arrived);
                    tma::load_async(args.input.reduction.o[1], args.globals.O_scratch, {args.common.reduction.src_uid[1], args.iter, 0}, args.inputs_arrived);
                    tma::load_async(args.input.reduction.lvec[0], args.globals.Lvec_scratch, {args.common.reduction.src_uid[0], args.iter}, args.inputs_arrived);
                    tma::load_async(args.input.reduction.lvec[1], args.globals.Lvec_scratch, {args.common.reduction.src_uid[1], args.iter}, args.inputs_arrived);
                }
                warpgroup::sync(5);
            }
            else { *(int*)0 = 0; } // Unreachable
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_WORKERS>();
            if(args.common.raw.op == layout::common_state::opcode::op_partial) {
                if(warpgroup::warpid() == 0) {
                    // auto q_st = subtile_inplace<16, layout::QK_D>(args.scratch.q, {warpgroup::warpid(), 0});
                    // int num_valid_Q_tokens = int(args.globals.Q.depth) * int(args.globals.Q.rows);
                    // if ((4*args.common.partial.q_seq_idx + warpgroup::warpid()) * q_st.rows < num_valid_Q_tokens) {
                    //     // TODO: need to put in the partial load and store
                    //     load(q_st, args.globals.Q, {args.common.partial.q_batch_idx, 4*args.common.partial.q_seq_idx + warpgroup::warpid(), 0});
                    // }
                    // else {
                    //     zero(q_st);
                    // }
                    // TODO: the / 4 needs to change for different num heads
                    load(args.scratch.q, args.globals.Q, {args.common.partial.q_batch_idx, args.common.partial.q_seq_idx / 4, 0});
                    zero(args.scratch.vec[0]);

                }
                zero(args.state.partial.norm_vec);
                neg_infty(args.state.partial.max_vec);
                zero(args.state.partial.o);
                group<8>::sync(10);
            }
            else if(args.common.raw.op == layout::common_state::opcode::op_reduction) {
                // Actually, nothing to do here!
            }
            else { *(int*)0 = 0; } // Unreachable
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            // 1.44269504089f is from exp2
            const float SOFTMAX_TEMPERATURE = args.globals.softmax_scale * 1.44269504089f;
            if (args.common.raw.op == layout::common_state::opcode::op_partial) {
                int total_q_tokens = int(args.globals.Q.rows) / 16;
                int valid_q_tokens = total_q_tokens - args.common.partial.q_seq_idx;
                if(warpgroup::groupid() == 0) {
                    // A = Q @ K.T
                    rt_fl<16, layout::cache_tile::rows> att_block_fp32;
                    warpgroup::mm_ABt(att_block_fp32, args.scratch.q, args.input.partial.c);
                    mul(args.state.partial.max_vec_last_scaled, args.state.partial.max_vec, SOFTMAX_TEMPERATURE);
                    warpgroup::mma_async_wait();
                    // // softmax
                    // if(args.iter == args.num_iters-1) { // need to mask out a bunch of entries in the last page
                    //     right_fill(att_block_fp32, att_block_fp32, args.common.partial.length - args.iter*layout::NUM_ROWS, base_types::constants<float>::neg_infty());
                    // }
                    // if(args.iter >= args.num_iters-2) { // Need to (possibly) do a causal mask.
                    //     warpgroup::tril(att_block_fp32, att_block_fp32, args.common.partial.length - args.iter*layout::NUM_ROWS - args.globals.Q.depth, base_types::constants<float>::neg_infty());
                    // }
                    row_max(args.state.partial.max_vec, att_block_fp32, args.state.partial.max_vec); // accumulate onto the max_vec
                    mul(args.state.partial.max_vec_scaled, args.state.partial.max_vec, SOFTMAX_TEMPERATURE);
                    mul(att_block_fp32, att_block_fp32, SOFTMAX_TEMPERATURE);
                    sub_row(att_block_fp32, att_block_fp32, args.state.partial.max_vec_scaled);
                    exp2(att_block_fp32, att_block_fp32);
                    sub(args.state.partial.max_vec_last_scaled, args.state.partial.max_vec_last_scaled, args.state.partial.max_vec_scaled);
                    exp2(args.state.partial.max_vec_last_scaled, args.state.partial.max_vec_last_scaled);
                    warpgroup::store(args.scratch.vec[0], args.state.partial.max_vec_last_scaled);
                    mul(args.state.partial.norm_vec, args.state.partial.norm_vec, args.state.partial.max_vec_last_scaled);
                    row_sum(args.state.partial.norm_vec, att_block_fp32, args.state.partial.norm_vec); // accumulate onto the norm_vec
                    warpgroup::store(args.scratch.att_block, att_block_fp32);
                }
                group<8>::sync(10);
                warpgroup::load(args.state.partial.max_vec_last_scaled, args.scratch.vec[0]);

                if (warpgroup::warpid() < valid_q_tokens) {
                    mul_row(args.state.partial.o, args.state.partial.o, args.state.partial.max_vec_last_scaled); // normalize o_reg before mma
                } else {
                    zero(args.state.partial.o);
                }
                // O += A @ V
                auto (&v_tile)[2] = reinterpret_cast<typename layout::v_tile(&)[2]>(args.input.partial.c);
                warpgroup::mma_AB(args.state.partial.o, args.scratch.att_block, v_tile[warpgroup::groupid()]);
                warpgroup::mma_async_wait();
                if(warpgroup::laneid() == 0) arrive(args.inputs_finished); // done!
                warpgroup::sync(warpgroup::groupid());
            }
            else if(args.common.raw.op == layout::common_state::opcode::op_reduction) {
                // Actually, nothing to do here!
                auto o1_st = subtile_inplace<16, layout::VO_D/8>(args.input.reduction.o[0], {0, group<8>::warpid()});
                auto o2_st = subtile_inplace<16, layout::VO_D/8>(args.input.reduction.o[1], {0, group<8>::warpid()});
                load(args.state.reduction.o[0], o1_st);
                load(args.state.reduction.o[1], o2_st);
                load(args.state.reduction.lvec[0], args.input.reduction.lvec[0]);
                load(args.state.reduction.lvec[1], args.input.reduction.lvec[1]);
                if(warpgroup::laneid() == 0) arrive(args.inputs_finished); // done!
                col_vec<rt_fl<16, layout::cache_tile::rows>> max_lvec, sum_lvec;
                max(max_lvec, args.state.reduction.lvec[0], args.state.reduction.lvec[1]);
                sub(args.state.reduction.lvec[0], args.state.reduction.lvec[0], max_lvec);
                sub(args.state.reduction.lvec[1], args.state.reduction.lvec[1], max_lvec);
                exp2(args.state.reduction.lvec[0], args.state.reduction.lvec[0]);
                exp2(args.state.reduction.lvec[1], args.state.reduction.lvec[1]);
                add(sum_lvec, args.state.reduction.lvec[0], args.state.reduction.lvec[1]);
                div(args.state.reduction.lvec[0], args.state.reduction.lvec[0], sum_lvec);
                div(args.state.reduction.lvec[1], args.state.reduction.lvec[1], sum_lvec);
                mul_row(args.state.reduction.o[0], args.state.reduction.o[0], args.state.reduction.lvec[0]);
                mul_row(args.state.reduction.o[1], args.state.reduction.o[1], args.state.reduction.lvec[1]);
                add(args.state.reduction.o[0], args.state.reduction.o[0], args.state.reduction.o[1]);
                log2(sum_lvec, sum_lvec);
                add(sum_lvec, sum_lvec, max_lvec);
                if(args.common.reduction.dst.uniform.type >= 0) {
                    store(args.globals.O, args.state.reduction.o[0],
                        {args.common.reduction.dst.tensor.batch_idx, 4*args.common.reduction.dst.tensor.seq_idx + args.iter, 0, group<8>::warpid()});
                }
                else {
                    // if (threadIdx.x == 0) {
                    //     printf("store to O_scratch, 254: %d, %d\n", args.common.reduction.dst.uniform.uid, group<8>::warpid());
                    // }
                    store(args.globals.O_scratch, args.state.reduction.o[0], {args.common.reduction.dst.uniform.uid, 0, group<8>::warpid()});
                    if(warpgroup::warpid() == 0) {
                        store(args.globals.Lvec_scratch, sum_lvec, {args.common.reduction.dst.uniform.uid, 0});
                    }
                }
            }
            else { *(int*)0 = 0; } // Unreachable
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if(args.common.raw.op == layout::common_state::opcode::op_partial) {
                if(warpgroup::groupid() == 0) {
                    warpgroup::store(args.scratch.vec[1], args.state.partial.norm_vec);
                }
                group<8>::sync(10);
                warpgroup::load(args.state.partial.norm_vec, args.scratch.vec[1]);
                div_row(args.state.partial.o, args.state.partial.o, args.state.partial.norm_vec);
                // if (threadIdx.x == 0) {
                //     printf("partial dst: %d, %d\n", args.common.partial.dst.uniform.type, args.common.partial.dst.uniform.uid);
                // }
                if(args.common.partial.dst.uniform.type >= 0) { // batch is meaningful
                    auto (&o_smem)[2] = reinterpret_cast<typename layout::o_tile_d2(&)[2]>(args.scratch.q);
                    warpgroup::store(o_smem[warpgroup::groupid()], args.state.partial.o);
                    warpgroup::sync(warpgroup::groupid());
                    // auto o_st = subtile_inplace<16, layout::VO_Dd2>(o_smem[warpgroup::groupid()], {warpgroup::warpid(), 0});
                    // if (threadIdx.x == 0) {
                    //     printf("store to O line 280: %d, %d\n", args.common.partial.dst.tensor.batch_idx, 4*args.common.partial.dst.tensor.seq_idx + warpgroup::warpid());
                    //     printf("O shapes: %d, %d, %d, %d\n", int(args.globals.O.batch), int(args.globals.O.depth), int(args.globals.O.rows), int(args.globals.O.cols));
                    //     printf("O subtile shape: %d, %d\n", int(o_st.rows), int(o_st.cols));
                    // }
                    // if (threadIdx.x == 0) {
                    //     printf("O tile: ");
                    //     for (int i = 0; i < 10; i++) {
                    //         printf("%f %f ", args.state.partial.o.tiles[0][0].data[i].x, args.state.partial.o.tiles[0][0].data[i].y);
                    //     }
                    //     printf("\n");
                    // }
                    // int num_valid_O_tokens = int(args.globals.O.depth) * int(args.globals.O.rows);
                    // TODO: the 4 * is probably wrong...
                    // if (threadIdx.x == 0) {
                    //     printf("O_smem tile\n");
                    //     for (int i = 0; i < o_smem[0].rows; i++) {
                    //         for (int j = 0; j < 10; j++) {
                    //             printf("%f ", __bfloat162float(o_smem[0].data[i * o_smem[0].cols + j]));
                    //         }
                    //         printf("\n");
                    //     }
                    // }
                    if (warpgroup::warpid() == 0 || warpgroup::warpid() == 4) {
                        store(args.globals.O, o_smem[warpgroup::groupid()], {args.common.partial.dst.tensor.batch_idx, args.common.partial.dst.tensor.seq_idx / 4, warpgroup::groupid()});
                    }
                    // if ((4*args.common.partial.dst.tensor.seq_idx + warpgroup::warpid() % 4) * o_st.rows < num_valid_O_tokens) {
                    //     // todo: need the partial store
                    //     store(args.globals.O, o_st, {args.common.partial.dst.tensor.batch_idx, 4*args.common.partial.dst.tensor.seq_idx + warpgroup::warpid() % 4, 0, warpgroup::groupid()});
                    // }
                    // if (threadIdx.x % 32 == 0) {
                    //     printf("output O: %d, %d, %d, %d, thread %d\n", int(args.common.partial.dst.tensor.batch_idx), int(4*args.common.partial.dst.tensor.seq_idx + warpgroup::warpid() % 4), 0, int(warpgroup::groupid()), threadIdx.x);
                    // }
                }
                else { // write out directly to O scratch, without going through smem
                    // if (threadIdx.x == 0) {
                    //     printf("store to O_scratch line 284: %d, %d\n", args.common.partial.dst.uniform.uid, group<8>::warpid());
                    // }
                    warpgroup::store(args.globals.O_scratch, args.state.partial.o, {args.common.partial.dst.uniform.uid, 0, 0});
                }
                warpgroup::sync(warpgroup::groupid());
                if(args.common.partial.dst.uniform.type < 0) {
                    if(group<8>::laneid() == 0) {
                        args.globals.Semaphore[{args.common.partial.dst.uniform.uid}] = 1;
                    }
                }
                if(warpgroup::laneid() == 0) arrive(args.finish_finished); // done!
            }
            else if(args.common.raw.op == layout::common_state::opcode::op_reduction) {
                // Increment the semaphore for the next stage, if this is not the last one.
                warpgroup::sync(warpgroup::groupid()); // Make sure memory has been flushed to global memory.
                if(args.common.reduction.dst.uniform.type < 0) {
                    if(group<8>::laneid() == 0) {
                        args.globals.Semaphore[{args.common.reduction.dst.uniform.uid}] = 1;
                    }
                    __syncwarp();
                }
                if(warpgroup::laneid() == 0) arrive(args.finish_finished); // done!
            }
            else { *(int*)0 = 0; } // Unreachable
        }
    };
};

PYBIND11_MODULE(mla_decode, m) {
    m.doc() = "mla_decode python module";
    py::bind_kernel<lcf::kernel<mla_decode_template>>(m, "mla_decode",
        &mla_decode_layout::globals::Ops,
        &mla_decode_layout::globals::Q,
        &mla_decode_layout::globals::Cache,
        &mla_decode_layout::globals::Table,
        &mla_decode_layout::globals::O,
        &mla_decode_layout::globals::O_scratch,
        &mla_decode_layout::globals::Lvec_scratch,
        &mla_decode_layout::globals::Semaphore,
        &mla_decode_layout::globals::softmax_scale
    );
}