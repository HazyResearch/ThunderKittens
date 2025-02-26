#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

/*

consumer_state.att_block in a warp looks like this:

[ 
  x x x x x x ... 192 or 128 columns ... x x x x x x
  ... 16 rows of it
  x x x x x x ... 192 or 128 columns ... x x x x x x
]

Logically in a warpgroup, it looks like this:
[
  x x x x x x ... 192 or 128 columns ... x x x x x x
  ... 64 rows of it
  x x x x x x ... 192 or 128 columns ... x x x x x x
]

*/

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int D, int NUM_WORKERS> struct attn_fwd_layout {
    using qo_tile   = st_bf<64, D>;
    using kv_tile   = st_bf<D==64?192:128, D>;
    using qo_global = kittens::gl<bf16, -1, -1, -1, D, qo_tile>;
    using kv_global = kittens::gl<bf16, -1, -1, -1, D, kv_tile>;
    struct globals {
        qo_global O, Q;
        kv_global K, V;
        int dynamic_shared_memory() { return MAX_SHARED_MEMORY - 2000; }
        dim3 grid()  { return dim3(132); }
        dim3 block() { return dim3((12+4)*WARP_THREADS); }
    };
    struct input_block    { kv_tile k, v; };
    struct scratch_block  { qo_tile q[NUM_WORKERS]; };
    struct common_state   { int batch, head, seq; };
    struct consumer_state {
        rt_fl<16, qo_tile::cols> o_reg;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec, norm_vec;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec_last_scaled, max_vec_scaled;
        rt_fl<16, kv_tile::rows> att_block;
        rt_bf<16, kv_tile::rows> att_block_mma;
    };
};
template<int D, bool causal> struct attn_fwd_template {
    static constexpr int NUM_CONSUMER_WARPS = 12, NUM_WORKERS = NUM_CONSUMER_WARPS/4, INPUT_PIPE_STAGES = 2;
    using layout = attn_fwd_layout<D, NUM_WORKERS>;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int task_id = gridDim.x*args.task_iter + blockIdx.x;
        int seq_q = (args.globals.Q.rows + NUM_WORKERS*layout::qo_tile::rows - 1)/(NUM_WORKERS*layout::qo_tile::rows);
        args.common.batch = task_id / (seq_q*args.globals.K.depth); task_id -= args.common.batch * seq_q * args.globals.K.depth;
        args.common.head  = task_id / seq_q;                        task_id -= args.common.head  * seq_q;
        args.common.seq   = task_id;
        args.num_iters = args.common.batch < args.globals.Q.batch ? (args.globals.K.rows + layout::kv_tile::rows - 1)/(layout::kv_tile::rows) : -1;
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                tma::load_async(args.input.k, args.globals.K, {args.common.batch, args.common.head, args.iter, 0}, args.inputs_arrived);
                tma::load_async(args.input.v, args.globals.V, {args.common.batch, args.common.head, args.iter, 0}, args.inputs_arrived);
            }
            else if(laneid() == 0) arrive(args.inputs_arrived);
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_WORKERS>();
            if((args.common.seq*NUM_WORKERS + warpgroup::groupid())*layout::qo_tile::rows < args.globals.Q.rows) // out of bounds?
                warpgroup::load(args.scratch.q[warpgroup::groupid()], args.globals.Q,
                                {args.common.batch, args.common.head, args.common.seq*NUM_WORKERS+warpgroup::groupid(), 0});
            zero(args.state.o_reg);
            zero(args.state.norm_vec);
            neg_infty(args.state.max_vec);
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            int qidx = (args.common.seq*NUM_WORKERS+warpgroup::groupid())*layout::qo_tile::rows;
            int kvidx = args.iter*layout::kv_tile::rows;
            /*
            How to know whether to skip a whole block if it's causal?
            Q index is (args.common.seq*NUM_WORKERS+warpgroup::groupid()) * layout::qo_tile::rows
            K and V index is (args.iter*layout::kv_tile::rows) * args.globals.K.depth

            We can skip this block if:
            * causal is True and K/V index is greater than Q index
            * which happens when args.iter*layout::kv_tile::rows > args.common.seq*NUM_WORKERS+warpgroup::groupid()
            */
            if (!causal || kvidx <= qidx) {
                constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;
                // A = Q @ K.T
                warpgroup::mm_ABt(args.state.att_block, args.scratch.q[warpgroup::groupid()], args.input.k);
                mul(args.state.max_vec_last_scaled, args.state.max_vec, TEMPERATURE_SCALE);
                warpgroup::mma_async_wait();
                // softmax
                right_fill(args.state.att_block, args.state.att_block, args.globals.K.rows - args.iter*layout::kv_tile::rows, base_types::constants<float>::neg_infty());
                if (causal) {
                    // if qidx - kvidx is less than the number of columns, this tile passes the diagonal
                    if (qidx - kvidx < layout::qo_tile::cols) {
                        // blocks are wider than they are tall, so we have mulitple blocks on the diagonal
                        // kvidx - qidx gives the (negative) offset based on block
                        // 16 * (warpgroup::warpid() % 4) gives the offset based on the warp
                        tril(args.state.att_block, args.state.att_block, kvidx - qidx - 16 * (warpgroup::warpid() % 4), base_types::constants<float>::neg_infty());
                    }
                }
                row_max(args.state.max_vec, args.state.att_block, args.state.max_vec); // accumulate onto the max_vec
                mul(args.state.max_vec_scaled, args.state.max_vec, TEMPERATURE_SCALE);
                mul(args.state.att_block, args.state.att_block, TEMPERATURE_SCALE);
                sub_row(args.state.att_block, args.state.att_block, args.state.max_vec_scaled);
                exp2(args.state.att_block, args.state.att_block);
                sub(args.state.max_vec_last_scaled, args.state.max_vec_last_scaled, args.state.max_vec_scaled);
                exp2(args.state.max_vec_last_scaled, args.state.max_vec_last_scaled);
                mul(args.state.norm_vec, args.state.norm_vec, args.state.max_vec_last_scaled);
                row_sum(args.state.norm_vec, args.state.att_block, args.state.norm_vec); // accumulate onto the norm_vec
                mul_row(args.state.o_reg, args.state.o_reg, args.state.max_vec_last_scaled); // normalize o_reg before mma
                copy(args.state.att_block_mma, args.state.att_block); // convert to bf16 for mma
                // O += A @ V
                warpgroup::mma_AB(args.state.o_reg, args.state.att_block_mma, args.input.v);
                warpgroup::mma_async_wait();
            }
            if(laneid() == 0) arrive(args.inputs_finished); // done!
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if((args.common.seq*NUM_WORKERS+warpgroup::groupid())*64 < args.globals.Q.rows) { // out of bounds?
                div_row(args.state.o_reg, args.state.o_reg, args.state.norm_vec);
                auto &o_smem = reinterpret_cast<typename layout::qo_tile&>(args.scratch.q[warpgroup::groupid()]);
                warpgroup::store(o_smem, args.state.o_reg);
                warpgroup::sync(warpgroup::groupid());
                if(warpgroup::warpid() == 0)
                    tma::store_async(args.globals.O, o_smem, {args.common.batch, args.common.head, args.common.seq*NUM_WORKERS+warpgroup::groupid(), 0});
                tma::store_async_read_wait();
            }
            __syncwarp();
            if(laneid() == 0) arrive(args.finish_finished); // done!
        }
    };
};
// kernel is kittens::prototype::lcf::kernel<attn_fwd_template<HEAD_DIM>>;

PYBIND11_MODULE(attn_fwd_pybind, m) {
    m.doc() = "TK Attention Forward Demo";
    py::bind_kernel<kittens::prototype::lcf::kernel<attn_fwd_template<128, false>>>(m, "fwd_128_noncausal",
        &attn_fwd_layout<128, 3>::globals::O,
        &attn_fwd_layout<128, 3>::globals::Q,
        &attn_fwd_layout<128, 3>::globals::K,
        &attn_fwd_layout<128, 3>::globals::V
    );
    py::bind_kernel<kittens::prototype::lcf::kernel<attn_fwd_template<128, true>>>(m, "fwd_128_causal",
        &attn_fwd_layout<128, 3>::globals::O,
        &attn_fwd_layout<128, 3>::globals::Q,
        &attn_fwd_layout<128, 3>::globals::K,
        &attn_fwd_layout<128, 3>::globals::V
    );
}