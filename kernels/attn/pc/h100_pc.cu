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
	struct consumer_state {
		rt_fl<16, qo_tile::cols> o_reg;
		col_vec<rt_fl<16, kv_tile::rows>> max_vec_last, max_vec;
		col_vec<rt_fl<16, kv_tile::rows>> norm_vec_last, norm_vec;
		rt_fl<16, kv_tile::rows> att_block;
		rt_bf<16, kv_tile::rows> att_block_mma;
	};
};
struct attn_fwd_template {
	static constexpr int NUM_CONSUMER_WARPS = 12, NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS/4;
	using layout = attn_fwd_layout<NUM_CONSUMER_WARPGROUPS>;
	__device__ static inline bool task_coord(coord &coords, const typename layout::globals &g, int task_id) {
		constexpr int ROWS_PER_TASK = 16*NUM_CONSUMER_WARPS;
		int TASKS_PER_HEAD = (g.Q.rows + ROWS_PER_TASK - 1) / ROWS_PER_TASK;
		coords.b = task_id/(TASKS_PER_HEAD*g.Q.depth); // batch
		task_id -= coords.b*TASKS_PER_HEAD*g.Q.depth;
		coords.d = task_id/TASKS_PER_HEAD;
		task_id -= coords.d*TASKS_PER_HEAD;
		coords.r = task_id;
		return coords.b < g.Q.batch;
	}
	__device__ static inline int iters(const typename layout::globals &g, const coord &task_coord) {
		return g.K.rows/layout::kv_tile::rows;
	}
	struct producer {
		__device__ static void setup(producer_setup_args<layout> args) {
			warpgroup::producer_registers();
		}
		__device__ static void load(producer_load_args<layout> args) {
			if(warpgroup::warpid() != 0) return;
			tma::expect(args.inputs_arrived, args.input);
			tma::load_async(args.input.k, args.globals.K, {args.task_coord.b, args.task_coord.d, args.iter, 0}, args.inputs_arrived);
			tma::load_async(args.input.v, args.globals.V, {args.task_coord.b, args.task_coord.d, args.iter, 0}, args.inputs_arrived);
			arrive(args.inputs_arrived, 3);
		}
	};
	struct consumer {
		__device__ static void setup(consumer_setup_args<layout> args) {
			warpgroup::consumer_registers<NUM_CONSUMER_WARPGROUPS>();
			if((args.task_coord.r*NUM_CONSUMER_WARPGROUPS + warpgroup::groupid())*layout::qo_tile::rows < args.globals.Q.rows)
				warpgroup::load(args.scratch.q[warpgroup::groupid()], args.globals.Q,
								        {args.task_coord.b, args.task_coord.d, args.task_coord.r*NUM_CONSUMER_WARPGROUPS + warpgroup::groupid(), 0});
			zero(args.state.o_reg);
			neg_infty(args.state.max_vec);
			zero(args.state.norm_vec);
			warpgroup::sync();
		}
		__device__ static bool work(consumer_work_args<layout> args) {
      		// A = Q @ K.T
			warpgroup::mm_ABt(args.state.att_block, args.scratch.q[warpgroup::groupid()], args.input.k);
			copy(args.state.norm_vec_last, args.state.norm_vec);
			copy(args.state.max_vec_last, args.state.max_vec);
			warpgroup::mma_async_wait();
			// softmax
			row_max(args.state.max_vec, args.state.att_block, args.state.max_vec); // accumulate onto the max_vec
			mul(args.state.max_vec, args.state.max_vec, 0.127517430824598685218f);
			mul(args.state.att_block, args.state.att_block, 0.127517430824598685218f);
			sub_row(args.state.att_block, args.state.att_block, args.state.max_vec);
			exp2(args.state.att_block, args.state.att_block);
			sub(args.state.max_vec_last, args.state.max_vec_last, args.state.max_vec);
			exp2(args.state.max_vec_last, args.state.max_vec_last);
			mul(args.state.norm_vec, args.state.norm_vec, args.state.max_vec_last);
			row_sum(args.state.norm_vec, args.state.att_block, args.state.norm_vec); // accumulate onto the norm_vec
			div_row(args.state.att_block, args.state.att_block, args.state.norm_vec);
			mul(args.state.norm_vec_last, args.state.norm_vec_last, args.state.max_vec_last);
			div(args.state.norm_vec_last, args.state.norm_vec_last, args.state.norm_vec);
			copy(args.state.att_block_mma, args.state.att_block); // convert to bf16 for mma
			mul_row(args.state.o_reg, args.state.o_reg, args.state.norm_vec_last); // normalize o_reg before mma
      		// O += A @ V
			warpgroup::mma_AB(args.state.o_reg, args.state.att_block_mma, args.input.v);
			warpgroup::mma_async_wait();
			arrive(args.inputs_finished); // done!
		}
		__device__ static void finish(consumer_finish_args<layout> args) {
			arrive(args.finish_finished); // safe for producers to start loading next tiles, without question
			if((args.task_coord.r*NUM_CONSUMER_WARPGROUPS+warpgroup::groupid())*64 < args.globals.Q.rows) { // OOB
				auto &o_smem = reinterpret_cast<typename layout::qo_tile&>(args.scratch.q[warpgroup::groupid()]);
				warpgroup::store(o_smem, args.state.o_reg);
				warpgroup::sync();
				if(warpgroup::warpid() == 0) {
					tma::store_async(args.globals.O, o_smem,
							{args.task_coord.b, args.task_coord.d, args.task_coord.r*NUM_CONSUMER_WARPGROUPS + warpgroup::groupid(), 0});
					tma::store_async_read_wait();
				}
				warpgroup::sync();
			}
		}
	};
};

#include "harness.impl"