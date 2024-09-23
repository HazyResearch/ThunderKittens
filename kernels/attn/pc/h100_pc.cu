#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
template<int NUM_CONSUMER_WARPGROUPS>
struct attn_fwd_layout {
	using qo_tile   = st_bf<64, 128>;
	using kv_tile   = st_bf<128,128>;
	using qo_global = kittens::gl<bf16, -1, -1, -1, 128, qo_tile>;
	using kv_global = kittens::gl<bf16, -1, -1, -1, 128, kv_tile>;
	struct globals {
		qo_global O, Q;
		kv_global K, V;
	};
	struct input_block    { kv_tile k, v; };
	struct scratch_block  { qo_tile q[NUM_CONSUMER_WARPGROUPS]; };
	struct finish_block   { qo_tile o[NUM_CONSUMER_WARPGROUPS]; };
	struct consumer_state {
		rt_fl<16, qo_tile::cols> o_reg;
		col_vec<rt_fl<16, kv_tile::rows>> max_vec_last, max_vec;
		col_vec<rt_fl<16, kv_tile::rows>> norm_vec_last, norm_vec;
		int id, n_blocks;
	};
};
struct attn_fwd_template {
	static constexpr int NUM_CONSUMER_WARPS = 12, NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS/4;
	using layout = attn_fwd_layout<NUM_CONSUMER_WARPGROUPS>;
	__device__ static inline int iters(typename layout::globals &g) {
    return g.K.rows/layout::kv_tile::rows;
  }
	struct producer {
		__device__ static void setup(producer_setup_args<layout> args) {
			warpgroup::producer_registers();
		}
		__device__ static void load(producer_load_args<layout> args) {
			if(warpgroup::warpid() != 0) return;
      tma::expect(args.inputs_arrived, args.input);
      tma::load_async(args.input.k, args.globals.K, {blockIdx.z,blockIdx.y,args.iter,0}, args.inputs_arrived);
      tma::load_async(args.input.v, args.globals.V, {blockIdx.z,blockIdx.y,args.iter,0}, args.inputs_arrived);
      arrive(args.inputs_arrived, 3);
		}
	};
	struct consumer {
		__device__ static void setup(consumer_setup_args<layout> args) {
			warpgroup::consumer_registers<NUM_CONSUMER_WARPGROUPS>();
			args.state.id = warpgroup::groupid();
			if((blockIdx.x*NUM_CONSUMER_WARPGROUPS + args.state.id)*layout::qo_tile::rows < args.globals.Q.rows)
				warpgroup::load(args.scratch.q[args.state.id], args.globals.Q,
								        {blockIdx.z, blockIdx.y, blockIdx.x*NUM_CONSUMER_WARPGROUPS + args.state.id, 0});
			zero(args.state.o_reg);
			neg_infty(args.state.max_vec);
			zero(args.state.norm_vec);
			warpgroup::sync();
			warpgroup::mul(args.scratch.q[args.state.id], args.scratch.q[args.state.id],__float2bfloat16(0.088379));
			warpgroup::sync();
		}
		__device__ static bool work(consumer_work_args<layout> args) {
      // A = Q @ K.T
			rt_fl<16, layout::kv_tile::rows> att_block;
			warpgroup::mm_ABt(att_block, args.scratch.q[args.state.id], args.input.k);
			copy(args.state.norm_vec_last, args.state.norm_vec);
			copy(args.state.max_vec_last, args.state.max_vec);
			warpgroup::mma_async_wait();
      // softmax
			row_max(args.state.max_vec, att_block, args.state.max_vec); // accumulate onto the max_vec
			sub_row(att_block, att_block, args.state.max_vec);
			exp(att_block, att_block);
			sub(args.state.max_vec_last, args.state.max_vec_last, args.state.max_vec);
			exp(args.state.max_vec_last, args.state.max_vec_last);
			mul(args.state.norm_vec, args.state.norm_vec, args.state.max_vec_last);
			row_sum(args.state.norm_vec, att_block, args.state.norm_vec); // accumulate onto the norm_vec
			div_row(att_block, att_block, args.state.norm_vec);
			mul(args.state.norm_vec_last, args.state.norm_vec_last, args.state.max_vec_last);
			div(args.state.norm_vec_last, args.state.norm_vec_last, args.state.norm_vec);
			rt_bf<16, layout::kv_tile::rows> att_block_mma;
			copy(att_block_mma, att_block); // convert to bf16 for mma
			mul_row(args.state.o_reg, args.state.o_reg, args.state.norm_vec_last); // normalize o_reg before mma
      // O += A @ V
			warpgroup::mma_AB(args.state.o_reg, att_block_mma, args.input.v);
			warpgroup::mma_async_wait();
			arrive(args.inputs_finished); // done!
		}
		__device__ static void finish(consumer_finish_args<layout> args) {
			if((blockIdx.x*NUM_CONSUMER_WARPGROUPS+args.state.id)*64 >= args.globals.Q.rows) return; // OOB
      warpgroup::store(args.finish.o[args.state.id], args.state.o_reg);
      warpgroup::sync();
      if(warpgroup::warpid() != 0) return;
      tma::store_async(args.globals.O, args.finish.o[args.state.id],
                       {blockIdx.z, blockIdx.y, blockIdx.x*NUM_CONSUMER_WARPGROUPS + args.state.id, 0});
    }
	};
};

#include "harness.impl"