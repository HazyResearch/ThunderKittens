#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int NUM_CONSUMER_WARPGROUPS>
struct attn_fwd_layout {
	using qo_tile   = st_bf<64, 128>;
	using kv_tile   = st_bf<128, 128>;
	using qo_global = kittens::gl<bf16, -1, -1, -1, 128, qo_tile>;
	using kv_global = kittens::gl<bf16, -1, -1, -1, 128, kv_tile>;
	struct globals {
		qo_global O, Q;
		kv_global K, V;
	};
	struct input_block    { kv_tile k, v; };
	struct scratch_block  { qo_tile q[NUM_CONSUMER_WARPGROUPS]; };
	struct common_state { int batch, head, seq; };
	struct consumer_state {
		rt_fl<16, qo_tile::cols> o_reg;
		col_vec<rt_fl<16, kv_tile::rows>> max_vec, norm_vec;
		col_vec<rt_fl<16, kv_tile::rows>> max_vec_last_scaled, max_vec_scaled;
		rt_fl<16, kv_tile::rows> att_block;
		rt_bf<16, kv_tile::rows> att_block_mma;
	};
};
struct attn_fwd_template {
	static constexpr int NUM_CONSUMER_WARPS = 12, NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS/4,
						 INPUT_PIPE_STAGES = 2, DEBUG = 0;
	using layout = attn_fwd_layout<NUM_CONSUMER_WARPGROUPS>;
	__device__ static inline void common_setup(common_setup_args<layout> args) {
		constexpr int ROWS_PER_TASK = 16*NUM_CONSUMER_WARPS;
		int TASKS_PER_HEAD = (args.globals.Q.rows + ROWS_PER_TASK - 1) / ROWS_PER_TASK;
		int task_id = args.task_iter*gridDim.x + blockIdx.x;
		args.common.batch = task_id/(TASKS_PER_HEAD*args.globals.Q.depth); // batch
		task_id -= args.common.batch*TASKS_PER_HEAD*args.globals.Q.depth;
		args.common.head = task_id/TASKS_PER_HEAD;
		task_id -= args.common.head*TASKS_PER_HEAD;
		args.common.seq = task_id;
		args.num_iters = (args.common.batch < args.globals.Q.batch) ? args.globals.K.rows/layout::kv_tile::rows : -1;
	}
	struct producer {
		__device__ static inline void setup(producer_setup_args<layout> args) {
			warpgroup::producer_registers();
		}
		__device__ static inline void load(producer_load_args<layout> args) {
			if(warpgroup::warpid() != 0) return;
			tma::expect(args.inputs_arrived, args.input);
			tma::load_async(args.input.k, args.globals.K, {args.common.batch, args.common.head, args.iter, 0}, args.inputs_arrived);
			tma::load_async(args.input.v, args.globals.V, {args.common.batch, args.common.head, args.iter, 0}, args.inputs_arrived);
			if(laneid() == 0) arrive(args.inputs_arrived, 3);
		}
	};
	struct consumer {
		__device__ static inline void setup(consumer_setup_args<layout> args) {
			warpgroup::consumer_registers<NUM_CONSUMER_WARPGROUPS>();
			if((args.common.seq*NUM_CONSUMER_WARPGROUPS + warpgroup::groupid())*layout::qo_tile::rows < args.globals.Q.rows)
				warpgroup::load(args.scratch.q[warpgroup::groupid()], args.globals.Q,
								        {args.common.batch, args.common.head, args.common.seq*NUM_CONSUMER_WARPGROUPS + warpgroup::groupid(), 0});
			zero(args.state.o_reg);
			neg_infty(args.state.max_vec);
			zero(args.state.norm_vec);
			warpgroup::sync();
		}
		__device__ static inline void compute(consumer_compute_args<layout> args) {
      		// A = Q @ K.T
			warpgroup::mm_ABt(args.state.att_block, args.scratch.q[warpgroup::groupid()], args.input.k);
			mul(args.state.max_vec_last_scaled, args.state.max_vec, 0.08838834764f * 1.44269504089f);
			warpgroup::mma_async_wait();
			// softmax
			row_max(args.state.max_vec, args.state.att_block, args.state.max_vec); // accumulate onto the max_vec
			mul(args.state.max_vec_scaled, args.state.max_vec, 0.08838834764f * 1.44269504089f);
			mul(args.state.att_block, args.state.att_block, 0.08838834764f * 1.44269504089f);
			sub_row(args.state.att_block, args.state.att_block, args.state.max_vec_scaled);
			exp2(args.state.att_block, args.state.att_block);
			sub(args.state.max_vec_last_scaled, args.state.max_vec_last_scaled, args.state.max_vec_scaled);
			exp2(args.state.max_vec_last_scaled, args.state.max_vec_last_scaled);
			mul(args.state.norm_vec, args.state.norm_vec, args.state.max_vec_last_scaled);
			row_sum(args.state.norm_vec, args.state.att_block, args.state.norm_vec); // accumulate onto the norm_vec
			mul_row(args.state.o_reg, args.state.o_reg, args.state.max_vec_last_scaled); // normalize o_reg before mma
			add(args.state.o_reg, args.state.o_reg, 0.f);
			copy(args.state.att_block_mma, args.state.att_block); // convert to bf16 for mma
      		// O += A @ V
			warpgroup::mma_AB(args.state.o_reg, args.state.att_block_mma, args.input.v);
			warpgroup::mma_async_wait();
			if(laneid() == 0) arrive(args.inputs_finished); // done!
		}
		__device__ static inline void finish(consumer_finish_args<layout> args) {
			if(laneid() == 0) arrive(args.finish_finished); // safe for producers to start loading next tiles
			if((args.common.seq*NUM_CONSUMER_WARPGROUPS+warpgroup::groupid())*64 < args.globals.Q.rows) { // OOB
				div_row(args.state.o_reg, args.state.o_reg, args.state.norm_vec);
				auto &o_smem = reinterpret_cast<typename layout::qo_tile&>(args.scratch.q[warpgroup::groupid()]);
				warpgroup::store(o_smem, args.state.o_reg);
				warpgroup::sync();
				if(warpgroup::warpid() == 0) {
					tma::store_async(args.globals.O, o_smem,
							{args.common.batch, args.common.head, args.common.seq*NUM_CONSUMER_WARPGROUPS + warpgroup::groupid(), 0});
					tma::store_async_read_wait();
				}
				warpgroup::sync();
			}
		}
	};
};

#include "harness.impl"