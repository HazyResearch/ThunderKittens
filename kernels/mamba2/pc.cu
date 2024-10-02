#include "kittens.cuh"
#include "prototype.cuh"

// I am doing it with attention notation because that is what I understand.

using namespace kittens;
using namespace kittens::prototype;
struct mamba2_fwd_layout {
	using q_tile   = st_bf<64, 64>;
	using k_tile   = st_bf<64, 64>;
	using v_tile   = st_bf<64, 64>;
	using o_tile   = st_bf<64, 64>;
    using a_vec    = sv_fl<64>; // decays
	using q_global = kittens::gl<bf16, -1, -1, -1, 64, q_tile>; // B, H, N, S
	using k_global = kittens::gl<bf16, -1, -1, -1, 64, k_tile>;
	using v_global = kittens::gl<bf16, -1, -1, -1, 64, v_tile>;
	using o_global = kittens::gl<bf16, -1, -1, -1, 64, o_tile>;
    using a_global = kittens::gl<float, -1, -1,  1, -1, a_vec>;
	struct globals {
        q_global Q;
        k_global K;
        v_global V;
        o_global O;
        a_global A;
	};
	struct input_block    { 
        q_tile q;
        k_tile k;
        v_tile v;
        a_vec  a;
        a_vec  padding[7];
    };
    struct output_block {
        o_tile o;
    };
	struct scratch_block  { 
        st_bf<64, 64> kv, k;
        a_vec         a_cumsum;
        a_vec         padding[7];
    };
	struct consumer_state {
		rt_fl<16, 64> o_reg;
		rt_fl<16, 64> att_block;
		rt_bf<16, 64> att_block_mma;
        rt_fl<16, 64> local_decay;
        rt_bf<16, 64> q_reg, k_reg;
        rt_fl<16, 64> kv;
	};
};
struct mamba2_fwd_template {
	static constexpr int NUM_CONSUMER_WARPS = 4, NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS/4, OUTPUT_PIPE_STAGES=1, INPUT_PIPE_STAGES=1, DEBUG=0;
	using layout = mamba2_fwd_layout;
	__device__ static inline bool task_coord(coord &coords, const typename layout::globals &g, int task_id) {
		constexpr int ROWS_PER_TASK = 16*NUM_CONSUMER_WARPS;
		int TASKS_PER_HEAD = (g.Q.rows + ROWS_PER_TASK - 1) / ROWS_PER_TASK;
		coords.b = task_id/g.Q.depth; // batch = id / heads.
		task_id -= coords.b*g.Q.depth;
		coords.d = task_id;
		return coords.b < g.Q.batch;
	}
	__device__ static inline int iters(const typename layout::globals &g, const coord &task_coord) {
		return g.K.rows/layout::k_tile::rows;
	}
	struct producer {
		__device__ static void setup(producer_setup_args<layout> args) {
			warpgroup::producer_registers();
		}
		__device__ static void load(producer_load_args<layout> args) {
			if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input.q, args.input.k, args.input.v, args.input.a);
                tma::load_async(args.input.q, args.globals.Q, {args.task_coord.b,                 0, args.iter, 0}, args.inputs_arrived);
                tma::load_async(args.input.k, args.globals.K, {args.task_coord.b,                 0, args.iter, 0}, args.inputs_arrived);
                tma::load_async(args.input.v, args.globals.V, {args.task_coord.b, args.task_coord.d, args.iter, 0}, args.inputs_arrived);
                tma::load_async(args.input.a, args.globals.A, {args.task_coord.b, args.task_coord.d, 0, args.iter}, args.inputs_arrived);
                arrive(args.inputs_arrived, 3);
            }
		}
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == 0) {;
                tma::store_async(args.globals.O, args.output.o, {args.task_coord.b, args.task_coord.d, args.iter, 0});
                tma::store_async_read_wait();
                __syncwarp();
                arrive(args.outputs_finished, 4);
            }
        }
	};
	struct consumer {
		__device__ static void setup(consumer_setup_args<layout> args) {
			warpgroup::increase_registers<224>();
            zero(args.state.kv);
		}
		__device__ static bool work(consumer_work_args<layout> args) {
            // Start by doing cumsum into shared memory
            warpgroup::sync();
            warpgroup::copy(args.scratch.a_cumsum, args.input.a);
            warpgroup::sync();
            if(warpid() <= 1) {
                // Perform the prefix sum (Hillis-Steele scan)
                for (int offset = 1; offset < 64; offset *= 2) {
                    // Store the value from the previous iteration
                    float temp = (threadIdx.x >= offset) ? args.scratch.a_cumsum[threadIdx.x - offset] : 0.0f;
                    group<2>::sync(14);
                    // Update the shared memory with the new cumulative sum
                    args.scratch.a_cumsum[threadIdx.x] += temp;
                    group<2>::sync(14);
                }
            }
            warpgroup::sync(); // cumulative sum done
            // if(threadIdx.x == 0) {
            //     printf("\n\ncumsum, iter %d: \n\n", args.iter);
            //     for(int i = 0; i < 64; i++) {
            //         printf("%f ", args.scratch.a_cumsum[i]);
            //     }
            //     printf("\n");
            // }
            // warpgroup::sync();
            // Calculate decays
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                int base_row = warpgroup::warpid()*16 + laneid()/4;
                int base_col = i*16 + (laneid()%4)*2;
                args.state.local_decay.tiles[0][i].data[0].x = args.scratch.a_cumsum[base_row + 0] - args.scratch.a_cumsum[base_col + 0];
                args.state.local_decay.tiles[0][i].data[0].y = args.scratch.a_cumsum[base_row + 0] - args.scratch.a_cumsum[base_col + 1];
                args.state.local_decay.tiles[0][i].data[1].x = args.scratch.a_cumsum[base_row + 8] - args.scratch.a_cumsum[base_col + 0];
                args.state.local_decay.tiles[0][i].data[1].y = args.scratch.a_cumsum[base_row + 8] - args.scratch.a_cumsum[base_col + 1];
                args.state.local_decay.tiles[0][i].data[2].x = args.scratch.a_cumsum[base_row + 0] - args.scratch.a_cumsum[base_col + 8];
                args.state.local_decay.tiles[0][i].data[2].y = args.scratch.a_cumsum[base_row + 0] - args.scratch.a_cumsum[base_col + 9];
                args.state.local_decay.tiles[0][i].data[3].x = args.scratch.a_cumsum[base_row + 8] - args.scratch.a_cumsum[base_col + 8];
                args.state.local_decay.tiles[0][i].data[3].y = args.scratch.a_cumsum[base_row + 8] - args.scratch.a_cumsum[base_col + 9];
            }
            exp(args.state.local_decay, args.state.local_decay);
            // causal mask
            #pragma unroll
            for(int i = 0; i < 4; i++) { // causal mask
                auto &decay_subtile = reinterpret_cast<rt_fl<16,16>&>(args.state.local_decay.tiles[0][i]);
                if      (i >  warpgroup::warpid()) { zero       (decay_subtile); }
                else if (i == warpgroup::warpid()) { make_causal(decay_subtile, decay_subtile, kittens::base_types::constants<float>::zero()); }
            }
      		// A = Q @ K.T
            warpgroup::load(args.state.q_reg, args.input.q); // we need this later, anyways
			warpgroup::mm_ABt(args.state.att_block, args.state.q_reg, args.input.k);
			warpgroup::mma_async_wait();
            mul(args.state.att_block, args.state.att_block, args.state.local_decay);
            copy(args.state.att_block_mma, args.state.att_block);
            warpgroup::mm_AB(args.state.o_reg, args.state.att_block_mma, args.input.v);
            warpgroup::mma_async_wait();

            // // multiply q by decays
            {
                int base_row = warpgroup::warpid()*16 + laneid()/4;
                bf16 top = __float2bfloat16(expf(args.scratch.a_cumsum[base_row + 0]));
                bf16 bottom = __float2bfloat16(expf(args.scratch.a_cumsum[base_row +8]));
                #pragma unroll
                for(int i = 0; i < 4; i++) {
                    args.state.q_reg.tiles[0][i].data[0].x *= top;
                    args.state.q_reg.tiles[0][i].data[0].y *= top;
                    args.state.q_reg.tiles[0][i].data[1].x *= bottom;
                    args.state.q_reg.tiles[0][i].data[1].y *= bottom;
                    args.state.q_reg.tiles[0][i].data[2].x *= top;
                    args.state.q_reg.tiles[0][i].data[2].y *= top;
                    args.state.q_reg.tiles[0][i].data[3].x *= bottom;
                    args.state.q_reg.tiles[0][i].data[3].y *= bottom;
                }
            }

            warpgroup::store(args.scratch.kv, args.state.kv);
            warpgroup::sync();

            warpgroup::mma_AB(args.state.o_reg, args.state.q_reg, args.scratch.kv);
            warpgroup::mma_async_wait();

            warpgroup::store(args.output.o, args.state.o_reg);
            warpgroup::sync();
            arrive(args.outputs_arrived);


            float last_decay = args.scratch.a_cumsum[args.scratch.a_cumsum.length-1]; // last element
            float total_decay = expf(last_decay);
            mul(args.state.kv, args.state.kv, total_decay); // decay kv
            warpgroup::load(args.state.k_reg, args.input.k); // multiply k's by decays
            {
                int base_row = warpgroup::warpid()*16 + laneid()/4;
                bf16 top = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[base_row + 0]));
                bf16 bottom = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[base_row +8]));
                #pragma unroll
                for(int i = 0; i < 4; i++) {
                    args.state.k_reg.tiles[0][i].data[0].x *= top;
                    args.state.k_reg.tiles[0][i].data[0].y *= top;
                    args.state.k_reg.tiles[0][i].data[1].x *= bottom;
                    args.state.k_reg.tiles[0][i].data[1].y *= bottom;
                    args.state.k_reg.tiles[0][i].data[2].x *= top;
                    args.state.k_reg.tiles[0][i].data[2].y *= top;
                    args.state.k_reg.tiles[0][i].data[3].x *= bottom;
                    args.state.k_reg.tiles[0][i].data[3].y *= bottom;
                }
            }
            warpgroup::store(args.scratch.k, args.state.k_reg); // using as dummy memory
            warpgroup::sync();

            warpgroup::mma_AtB(args.state.kv, args.scratch.k, args.input.v);
            warpgroup::mma_async_wait();

            arrive(args.inputs_finished);

		}
	};
};

#include "harness.impl"