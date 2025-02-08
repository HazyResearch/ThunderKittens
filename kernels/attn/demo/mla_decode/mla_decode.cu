#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"


using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
struct mla_decode_layout {
    static constexpr int QK_D = 576, VO_D = 512, VO_Dd2 = VO_D/2, PAGE_SIZE = 64, MAX_PAGES = 32768/PAGE_SIZE;
    using q_tile         = st_bf<64, QK_D>;
    using q_global       = kittens::gl<bf16, -1, -1, -1, QK_D>; // B * R * H * D_QK
    using cache_tile     = st_bf<PAGE_SIZE, QK_D>; using v_tile = st_bf<PAGE_SIZE, VO_Dd2>; // we need the v_tile for later
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
    struct scratch_block  { q_tile q; st_bf<64, PAGE_SIZE> att_block; sv_bf<64> max_vec_last_scaled, norm_vec; };
    struct common_state   { int batch, head; };
    struct consumer_state {
        rt_fl<16, o_tile::cols> o_reg;
        col_vec<rt_fl<16, cache_tile::rows>> max_vec, norm_vec;
        col_vec<rt_fl<16, cache_tile::rows>> max_vec_last_scaled, max_vec_scaled;
        rt_fl<16, cache_tile::rows> att_block;
    };
};
struct mla_decode_template {
    static constexpr int NUM_CONSUMER_WARPS = 8, NUM_WORKERS = NUM_CONSUMER_WARPS/4, INPUT_PIPE_STAGES = 1, PRODUCER_BARRIER_ARRIVALS = 1, CONSUMER_BARRIER_ARRIVALS = 2;
    using layout = mla_decode_layout;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if(args.task_iter > 0) {
            args.num_iters = -1;
            return;
        }
        int task_id = blockIdx.x;
        args.common.batch = task_id / args.globals.Q.rows; task_id -= args.common.batch * args.globals.Q.rows;
        args.common.head  = task_id;
        args.num_iters = (args.globals.Lengths[args.common.batch] + layout::PAGE_SIZE - 1) / layout::PAGE_SIZE;
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                // next page we need to load?
                int next_page_id = args.globals.Table[coord<>{args.common.batch, args.iter}];
                tma::expect(args.inputs_arrived, args.input);
                tma::load_async(args.input.c, args.globals.Cache, {next_page_id, 0, 0}, args.inputs_arrived);
            }
            warpgroup::sync(5);
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_WORKERS>();
            warpgroup::load<1, false>(args.scratch.q, args.globals.Q, {args.common.batch, 0, args.common.head, 0});
            zero(args.state.o_reg);
            zero(args.state.norm_vec);
            neg_infty(args.state.max_vec);
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            // 1.44269504089f is from exp2
            const float SOFTMAX_TEMPERATURE = args.globals.softmax_scale * 1.44269504089f;
            // constexpr float SOFTMAX_TEMPERATURE = 0.04166666666f * 1.44269504089f;
            if(warpgroup::groupid() == 0) {
                // A = Q @ K.T
                warpgroup::mm_ABt(args.state.att_block, args.scratch.q, args.input.c);
                mul(args.state.max_vec_last_scaled, args.state.max_vec, SOFTMAX_TEMPERATURE);
                warpgroup::mma_async_wait();
                // softmax
                if(args.iter == args.num_iters-1) { // need to mask out a bunch of entries in the last page
                    right_fill(args.state.att_block, args.state.att_block, args.globals.Lengths[args.common.batch] - args.iter*layout::PAGE_SIZE, base_types::constants<float>::neg_infty());
                }
                if(args.iter >= args.num_iters-2) { // Need to (possibly) do a causal mask.
                    warpgroup::tril(args.state.att_block, args.state.att_block, args.globals.Lengths[args.common.batch] - args.iter*layout::PAGE_SIZE - args.globals.Q.depth, base_types::constants<float>::neg_infty());
                }
                row_max(args.state.max_vec, args.state.att_block, args.state.max_vec); // accumulate onto the max_vec
                mul(args.state.max_vec_scaled, args.state.max_vec, SOFTMAX_TEMPERATURE);
                mul(args.state.att_block, args.state.att_block, SOFTMAX_TEMPERATURE);
                sub_row(args.state.att_block, args.state.att_block, args.state.max_vec_scaled);
                exp2(args.state.att_block, args.state.att_block);
                sub(args.state.max_vec_last_scaled, args.state.max_vec_last_scaled, args.state.max_vec_scaled);
                exp2(args.state.max_vec_last_scaled, args.state.max_vec_last_scaled);
                warpgroup::store(args.scratch.max_vec_last_scaled, args.state.max_vec_last_scaled);
                mul(args.state.norm_vec, args.state.norm_vec, args.state.max_vec_last_scaled);
                row_sum(args.state.norm_vec, args.state.att_block, args.state.norm_vec); // accumulate onto the norm_vec
                warpgroup::store(args.scratch.att_block, args.state.att_block);
            }
            group<8>::sync(10);
            warpgroup::load(args.state.max_vec_last_scaled, args.scratch.max_vec_last_scaled);
            mul_row(args.state.o_reg, args.state.o_reg, args.state.max_vec_last_scaled); // normalize o_reg before mma
            // O += A @ V
            auto (&v_tile)[2] = reinterpret_cast<typename layout::v_tile(&)[2]>(args.input.c);
            warpgroup::mma_AB(args.state.o_reg, args.scratch.att_block, v_tile[warpgroup::groupid()]);
            warpgroup::mma_async_wait();
            if(warpgroup::laneid() == 0) arrive(args.inputs_finished); // done!
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if(warpgroup::groupid() == 0) {
                warpgroup::store(args.scratch.norm_vec, args.state.norm_vec);
            }
            group<8>::sync(10);
            warpgroup::load(args.state.norm_vec, args.scratch.norm_vec);
            div_row(args.state.o_reg, args.state.o_reg, args.state.norm_vec);
            auto (&o_smem)[2] = reinterpret_cast<typename layout::o_tile(&)[2]>(args.scratch.q);
            warpgroup::store(o_smem[warpgroup::groupid()], args.state.o_reg);
            warpgroup::sync(warpgroup::groupid());
            if(warpgroup::warpid() == 0) {
                tma::store_async<1>(args.globals.O, o_smem[warpgroup::groupid()], {args.common.batch, 0, args.common.head, warpgroup::groupid()});
            }
            tma::store_async_read_wait();
            __syncwarp();
            // warpgroup::store<1, false>(args.globals.O, o_smem[warpgroup::groupid()], {args.common.batch, 0, args.common.head, warpgroup::groupid()});
            // warpgroup::sync(warpgroup::groupid());
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