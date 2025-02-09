#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"


using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
struct mla_decode_layout {
    static constexpr int QK_D = 576, VO_D = 512, VO_Dd2 = VO_D/2, NUM_ROWS = 64, PAGE_SIZE = 256, MAX_PAGES = 32768/PAGE_SIZE;
    using q_tile         = st_bf<64, QK_D>;
    using q_global       = kittens::gl<bf16, -1, -1, -1, QK_D>; // B * R * H * D_QK
    // using cache_tile     = st_bf<PAGE_SIZE, QK_D>; using v_tile = st_bf<PAGE_SIZE, VO_Dd2>; // we need the v_tile for later
    using cache_tile     = st_bf<NUM_ROWS, QK_D>; using v_tile = st_bf<NUM_ROWS, VO_Dd2>; // we need the v_tile for later
    using cache_global   = kittens::gl<bf16, 1, -1, PAGE_SIZE, QK_D, cache_tile>; // 1 * #page * pagesize * QK_D
    using lengths_global = kittens::gl<int, 1, 1, 1, -1>; // B
    using table_global   = kittens::gl<int, 1, 1, -1, MAX_PAGES>; // B * (max # pages)
    using o_tile         = st_bf<64, VO_Dd2>;
    using o_global       = kittens::gl<bf16, -1, -1, -1, VO_D, tma::descriptor<o_tile, 1>>; // B * R * H * D_VO
    struct globals {
        q_global Q;
        cache_global Cache;
        lengths_global Lengths;
        table_global Table;
        o_global O;
        const float softmax_scale;
        int dynamic_shared_memory() { return 224000; }
        dim3 grid()  { return dim3(Q.batch * Q.rows); }
        dim3 block() { return dim3((8+4)*WARP_THREADS); }
    };
    struct input_block    { cache_tile c; };
    struct scratch_block  { 
        q_tile q;
        sv_bf<64> max_vec_last_scaled, norm_vec;
        semaphore mma_semaphore;
    };
    struct common_state   { int batch, head; };
    struct consumer_state {
        col_vec<rt_fl<16, cache_tile::rows>> max_vec, norm_vec;
        col_vec<rt_fl<16, cache_tile::rows>> max_vec_last_scaled, max_vec_scaled;
        tmem<float, 64, cache_tile::rows> att_tmem;
        tmem<bf16,  64, cache_tile::rows> att_tmem_bf16[2];
        tmem<float, 64, VO_Dd2> o_tmem;
    };
};
struct mla_decode_template {
    static constexpr int NUM_CONSUMER_WARPS = 8, NUM_WORKERS = NUM_CONSUMER_WARPS/4, INPUT_PIPE_STAGES = 2, PRODUCER_BARRIER_ARRIVALS = 1, CONSUMER_BARRIER_ARRIVALS = 2;
    using layout = mla_decode_layout;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if(args.task_iter > 0) {
            args.num_iters = -1;
            return;
        }
        int task_id = blockIdx.x;
        args.common.batch = task_id / args.globals.Q.rows; task_id -= args.common.batch * args.globals.Q.rows;
        args.common.head  = task_id;
        args.num_iters = (args.globals.Lengths[args.common.batch] + layout::NUM_ROWS - 1) / layout::NUM_ROWS;
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                // next page we need to load?
                int global_load_idx = args.iter % 4;
                int next_page_id = args.globals.Table[coord<>{args.common.batch, args.iter / 4}];
                tma::expect(args.inputs_arrived, args.input);
                tma::load_async(args.input.c, args.globals.Cache, {next_page_id, global_load_idx, 0}, args.inputs_arrived);
            }
            warpgroup::sync(5);
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_WORKERS>();
            warpgroup::load<1, false>(args.scratch.q, args.globals.Q, {args.common.batch, 0, args.common.head, 0});
            rt_fl<16, layout::o_tile::cols> o_reg;
            zero(args.state.norm_vec);
            neg_infty(args.state.max_vec);
            args.state.o_tmem           = args.tmem.subtile<tmem<float, 64, layout::VO_Dd2>>(16*warpgroup::groupid(), 0); // Start, vertical stride
            args.state.att_tmem         = args.tmem.subtile<tmem<float, 64, layout::cache_tile::rows>>(0, 256); // Halfway across.
            args.state.att_tmem_bf16[0] = args.tmem.subtile<tmem<bf16,  64, layout::cache_tile::rows>>(0, 256); // Halfway across.
            args.state.att_tmem_bf16[1] = args.tmem.subtile<tmem<bf16,  64, layout::cache_tile::rows>>(16, 256); // Halfway across.
            zero(o_reg);
            warpgroup::store_async(args.state.o_tmem, o_reg);
            if(args.task_iter == 0 && group<8>::warpid() == 0) {
                init_semaphore(args.scratch.mma_semaphore, 0, 2);
            }
            tm_store_wait();
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            // 1.44269504089f is from exp2
            const float SOFTMAX_TEMPERATURE = args.globals.softmax_scale * 1.44269504089f;
            // constexpr float SOFTMAX_TEMPERATURE = 0.04166666666f * 1.44269504089f;
            if(warpgroup::groupid() == 0) {
                // A = Q @ K.T
                if(warpgroup::warpid() == 0) {
                    mm_ABt(args.state.att_tmem, args.scratch.q, args.input.c, args.scratch.mma_semaphore);
                    if(laneid() == 0) arrive(args.scratch.mma_semaphore); // one more, to trip it
                }
                mul(args.state.max_vec_last_scaled, args.state.max_vec, SOFTMAX_TEMPERATURE);
                wait(args.scratch.mma_semaphore, 0);
                rt_fl<16, layout::cache_tile::rows> att_block_fp32;
                warpgroup::load_async(att_block_fp32, args.state.att_tmem);
                // softmax
                if(args.iter == args.num_iters-1) { // need to mask out a bunch of entries in the last page
                    right_fill(att_block_fp32, att_block_fp32, args.globals.Lengths[args.common.batch] - args.iter*layout::NUM_ROWS, base_types::constants<float>::neg_infty());
                }
                if(args.iter >= args.num_iters-2) { // Need to (possibly) do a causal mask.
                    warpgroup::tril(att_block_fp32, att_block_fp32, args.globals.Lengths[args.common.batch] - args.iter*layout::NUM_ROWS - args.globals.Q.depth, base_types::constants<float>::neg_infty());
                }
                row_max(args.state.max_vec, att_block_fp32, args.state.max_vec); // accumulate onto the max_vec
                mul(args.state.max_vec_scaled, args.state.max_vec, SOFTMAX_TEMPERATURE);
                mul(att_block_fp32, att_block_fp32, SOFTMAX_TEMPERATURE);
                sub_row(att_block_fp32, att_block_fp32, args.state.max_vec_scaled);
                exp2(att_block_fp32, att_block_fp32);
                sub(args.state.max_vec_last_scaled, args.state.max_vec_last_scaled, args.state.max_vec_scaled);
                exp2(args.state.max_vec_last_scaled, args.state.max_vec_last_scaled);
                warpgroup::store(args.scratch.max_vec_last_scaled, args.state.max_vec_last_scaled);
                mul(args.state.norm_vec, args.state.norm_vec, args.state.max_vec_last_scaled);
                row_sum(args.state.norm_vec, att_block_fp32, args.state.norm_vec); // accumulate onto the norm_vec
                rt_bf<16, layout::cache_tile::rows> att_block_bf16;
                copy(att_block_bf16, att_block_fp32);
                warpgroup::store_async(args.state.att_tmem_bf16[0], att_block_bf16);
                warpgroup::store_async(args.state.att_tmem_bf16[1], att_block_bf16);
            }
            // if(warpgroup::laneid() == 0) arrive(args.scratch.mma_semaphore);
            // wait(args.scratch.mma_semaphore, 0);
            group<8>::sync(10);
            warpgroup::load(args.state.max_vec_last_scaled, args.scratch.max_vec_last_scaled);
            rt_fl<16, layout::o_tile::cols> o_reg;
            warpgroup::load_async(o_reg, args.state.o_tmem);
            mul_row(o_reg, o_reg, args.state.max_vec_last_scaled); // normalize o_reg before mma
            warpgroup::store_async(args.state.o_tmem, o_reg);
            tm_store_wait();
            warpgroup::sync(warpgroup::groupid());
            // O += A @ V
            auto (&v_tile)[2] = reinterpret_cast<typename layout::v_tile(&)[2]>(args.input.c);
            if(warpgroup::warpid() == 0) {
                mma_AB(args.state.o_tmem, args.state.att_tmem_bf16[warpgroup::groupid()], v_tile[warpgroup::groupid()], args.scratch.mma_semaphore);
            }
            wait(args.scratch.mma_semaphore, 1);
            if(warpgroup::laneid() == 0) arrive(args.inputs_finished); // done!
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if(warpgroup::groupid() == 0) {
                warpgroup::store(args.scratch.norm_vec, args.state.norm_vec);
            }
            group<8>::sync(10);
            rt_fl<16, layout::o_tile::cols> o_reg;
            warpgroup::load_async(o_reg, args.state.o_tmem);
            warpgroup::load(args.state.norm_vec, args.scratch.norm_vec);
            div_row(o_reg, o_reg, args.state.norm_vec);
            auto (&o_smem)[2] = reinterpret_cast<typename layout::o_tile(&)[2]>(args.scratch.q);
            warpgroup::store(o_smem[warpgroup::groupid()], o_reg);
            warpgroup::sync(warpgroup::groupid());
            if(warpgroup::warpid() == 0) {
                tma::store_async<1>(args.globals.O, o_smem[warpgroup::groupid()], {args.common.batch, 0, args.common.head, warpgroup::groupid()});
            }
            tma::store_async_read_wait();
            __syncwarp();
            if(warpgroup::laneid() == 0) arrive(args.finish_finished); // done!
        }
    };
};

PYBIND11_MODULE(mla_decode, m) {
    m.doc() = "mla_decode python module";
    py::bind_kernel<lcf::kernel<mla_decode_template>>(m, "mla_decode",
        &mla_decode_layout::globals::Q,
        &mla_decode_layout::globals::Cache,
        &mla_decode_layout::globals::Lengths,
        &mla_decode_layout::globals::Table,
        &mla_decode_layout::globals::O,
        &mla_decode_layout::globals::softmax_scale
    );
}