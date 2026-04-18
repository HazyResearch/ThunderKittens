#include "kittens.cuh"
#include "prototype.cuh"

#ifdef TORCH_COMPILE
#define TK_COMPILE_MAMBA3
#endif

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcsf;
static constexpr int MIMO_RANK = 4; // appendix D
struct mamba3_fwd_layout {
    using q_tile = st_bf<64, 64>;
    using k_tile = st_bf<64, 64>;
    using v_tile = st_bf<64, 64>;
    using o_tile = st_bf<64, 64>;
    using a_vec = sv_fl<64>; 
    using b_vec = sv_fl<64>; // m3: beta
    using angle_vec = sv_fl<64>; // kept for API compatibility; trap-only path leaves this unused
    using q_global = kittens::gl<bf16, -1, -1, -1, 64, q_tile>; // B, H, N, S
    using k_global = kittens::gl<bf16, -1, -1, -1, 64, k_tile>; 
    using v_global = kittens::gl<bf16, -1, -1, -1, 64, v_tile>; 
    using o_global = kittens::gl<bf16, -1, -1, -1, 64, o_tile>; 
    using a_global = kittens::gl<float, -1, -1, 1, -1, a_vec>;
    using b_global = kittens::gl<float, -1, -1, 1, -1, b_vec>; // m3: beta
    using angle_global = kittens::gl<float, -1, -1, 1, -1, angle_vec>; // m3: rotary trick
    
    struct globals { 
        q_global Q; 
        k_global K; 
        v_global V; 
        o_global O; 
        a_global A; 
        b_global B; // m3: beta
        angle_global Angles; // m3: Rotary Angles
    };
    struct input_block {
        q_tile q;
        k_tile k;
        v_tile v[2];
        a_vec a[2];
        a_vec a_padding[4];
        // m3 addition
        b_vec b[2];
        b_vec b_padding[6];
        angle_vec angle[2];
    };
    struct output_block {
        o_tile o[2];
    };
    struct scratch_block {
        st_bf<64, 64> kv[2], k[2];
        a_vec a_cumsum[2];
        a_vec b_scale[2];
        a_vec padding[6]; 
    };
    struct common_state {
        int batch, head;
    };
    struct consumer_state {
        rt_fl<16, 64> o_reg;
		rt_fl<16, 64> att_block;
		rt_bf<16, 64> att_block_mma;
        rt_fl<16, 64> local_decay;
        rt_bf<16, 64> q_reg, k_reg;
        rt_fl<16, 64> kv;
        float trap_a_boundary, trap_b_boundary;
    };
    
};

using mamba3_siso_layout = mamba3_fwd_layout;
struct mamba3_mimo_layout : mamba3_fwd_layout {
    static constexpr int RANKS_PER_PASS = MIMO_RANK / 2;
    struct input_block {
        v_tile v[2];
        a_vec a[2];
        b_vec b[2];
        q_tile q[RANKS_PER_PASS];
        k_tile k[RANKS_PER_PASS];
        a_vec a_padding[4];
        b_vec b_padding[6];
        angle_vec angle[2];
    };
};

template<typename Layout>
struct mamba3_fwd_common_template {
    static constexpr int NUM_CONSUMER_WARPS=8, OUTPUT_PIPE_STAGES=2, INPUT_PIPE_STAGES=2, PRODUCER_BARRIER_ARRIVALS=1, CONSUMER_BARRIER_ARRIVALS=NUM_CONSUMER_WARPS/4;
    using layout = Layout;

    __device__ static inline float trap_scale(float a_next, float b_next) {
        return 1.0f + b_next * expf(-a_next);
    }

    template<typename Args>
    __device__ static inline void prefetch_trap_boundary(Args args, int next_chunk) {
        if (warpgroup::warpid() <= 1 && warpgroup::laneid() == 63) {
            int warpgroupid = warpgroup::groupid();
            int total_chunks = args.globals.K.rows() / layout::k_tile::rows;
            if (next_chunk < total_chunks) {
                coord<> idx = {args.common.batch, args.common.head + warpgroupid, 0, next_chunk * 64};
                args.state.trap_a_boundary = args.globals.A[idx];
                args.state.trap_b_boundary = args.globals.B[idx];
            } else {
                args.state.trap_a_boundary = 0.0f;
                args.state.trap_b_boundary = 0.0f;
            }
        }
    }

    __device__ static inline void build_trapezoidal_scale(consumer_compute_args<layout> args, int warpgroupid) {
        if (warpgroup::warpid() <= 1) {
            int tid = warpgroup::laneid();
            float s;
            if (tid < 63) {
                float a_next = args.input.a[warpgroupid][tid + 1];
                float b_next = args.input.b[warpgroupid][tid + 1];
                s = trap_scale(a_next, b_next);
            } else {
                s = trap_scale(args.state.trap_a_boundary, args.state.trap_b_boundary);
            }
            args.scratch.b_scale[warpgroupid][tid] = s;
        }
    }

    __device__ static inline void apply_trapezoidal_scale(consumer_compute_args<layout> args, int warpgroupid) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int base_row = warpgroup::warpid() * 16 + laneid() / 4;
            int base_col = i * 16 + (laneid() % 4) * 2;

            float s00 = args.scratch.b_scale[warpgroupid][base_col + 0];
            float s01 = args.scratch.b_scale[warpgroupid][base_col + 1];
            float s08 = args.scratch.b_scale[warpgroupid][base_col + 8];
            float s09 = args.scratch.b_scale[warpgroupid][base_col + 9];

            int r0 = base_row + 0;
            int r8 = base_row + 8;

            if (r0 > base_col + 0) args.state.local_decay.tiles[0][i].data[0].x *= s00;
            if (r0 > base_col + 1) args.state.local_decay.tiles[0][i].data[0].y *= s01;
            if (r8 > base_col + 0) args.state.local_decay.tiles[0][i].data[1].x *= s00;
            if (r8 > base_col + 1) args.state.local_decay.tiles[0][i].data[1].y *= s01;
            if (r0 > base_col + 8) args.state.local_decay.tiles[0][i].data[2].x *= s08;
            if (r0 > base_col + 9) args.state.local_decay.tiles[0][i].data[2].y *= s09;
            if (r8 > base_col + 8) args.state.local_decay.tiles[0][i].data[3].x *= s08;
            if (r8 > base_col + 9) args.state.local_decay.tiles[0][i].data[3].y *= s09;
        }
    }
    // applies sin/cos
#if 0
    // Rotary is intentionally disabled in the current TK kernel so this stays a trap-only
    // variant of the Mamba-2 inner scan. The angle argument is still threaded through the API
    // for now to avoid a wider signature churn.
    __device__ static inline void get_rotary_factors(consumer_compute_args<layout> args, int warpgroupid, int idx, float &s, float &c) {
        float theta = args.input.angle[warpgroupid][idx];
        sincosf(theta, &s, &c);
    }
    
    __device__ static inline void apply_rotary_to_pair_bf16(bf16 &x0, bf16 &x1, float s, float c) {
        float a = __bfloat162float(x0);
        float b = __bfloat162float(x1);
        float y0 = c * a - s * b;
        float y1 = s * a + c * b;
        x0 = __float2bfloat16(y0);
        x1 = __float2bfloat16(y1);
    }

    template<typename RegTile>
    __device__ static inline void apply_rotary_to_reg(consumer_compute_args<layout> args, int warpgroupid, RegTile &reg) {
        int base_row = warpgroup::warpid() * 16 + laneid() / 4;
        float s_top, c_top, s_bottom, c_bottom;

        get_rotary_factors(args, warpgroupid, base_row + 0, s_top, c_top);
        get_rotary_factors(args, warpgroupid, base_row + 8, s_bottom, c_bottom);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            apply_rotary_to_pair_bf16(reg.tiles[0][i].data[0].x, reg.tiles[0][i].data[0].y, s_top, c_top);
            apply_rotary_to_pair_bf16(reg.tiles[0][i].data[1].x, reg.tiles[0][i].data[1].y, s_bottom, c_bottom);
            apply_rotary_to_pair_bf16(reg.tiles[0][i].data[2].x, reg.tiles[0][i].data[2].y, s_top, c_top);
            apply_rotary_to_pair_bf16(reg.tiles[0][i].data[3].x, reg.tiles[0][i].data[3].y, s_bottom, c_bottom);
        }
    }
#endif

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int task_id = args.task_iter * gridDim.x + blockIdx.x;
        args.common.batch = task_id / (args.globals.V.depth()/(NUM_CONSUMER_WARPS/4));
        task_id -= args.common.batch*(args.globals.V.depth()/(NUM_CONSUMER_WARPS/4));
        args.common.head = task_id*2;
        args.num_iters = args.common.batch < args.globals.Q.batch() ? args.globals.K.rows()/layout::k_tile::rows : -1;
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }

        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                warp::tma::expect(args.inputs_arrived, args.input.q, args.input.k, args.input.v[0], args.input.a[0], args.input.b[0], args.input.v[1], args.input.a[1], args.input.b[1]);
                warp::tma::load_async(args.input.q, args.globals.Q, {args.common.batch, 0, args.iter, 0}, args.inputs_arrived);
                warp::tma::load_async(args.input.k, args.globals.K, {args.common.batch, 0, args.iter, 0}, args.inputs_arrived);
                
                #pragma unroll
                for (int i = 0; i < NUM_CONSUMER_WARPS/4; i++) {
                    warp::tma::load_async(args.input.v[i], args.globals.V, {args.common.batch, args.common.head+i, args.iter, 0}, args.inputs_arrived);
                    warp::tma::load_async(args.input.a[i], args.globals.A, {args.common.batch, args.common.head+i, 0, args.iter}, args.inputs_arrived);
                    warp::tma::load_async(args.input.b[i], args.globals.B, {args.common.batch, args.common.head+i, 0, args.iter}, args.inputs_arrived);
                    // warp::tma::load_async(args.input.angle[i], args.globals.Angles, {args.common.batch, args.common.head+i, 0, args.iter}, args.inputs_arrived);
                }
                
                __syncwarp();
            }
        }

        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                #pragma unroll
                for (int i = 0; i < NUM_CONSUMER_WARPS/4; i++) {
                    warp::tma::store_async(args.globals.O, args.output.o[i], {args.common.batch, args.common.head+i, args.iter, 0});
                }
                warp::tma::store_async_read_wait();
                __syncwarp();
                if (laneid() == 0) arrive(args.outputs_finished);
                __syncwarp();
            }
        }

    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/WARPGROUP_WARPS>();
            warp::zero(args.state.kv);
            prefetch_trap_boundary(args, 1);
        }

        __device__ static void compute(consumer_compute_args<layout> args) {
            int warpgroupid = warpgroup::groupid();
            warpgroup::sync(warpgroupid);
            warpgroup::copy(args.scratch.a_cumsum[warpgroupid], args.input.a[warpgroupid]);
            warpgroup::sync(warpgroupid);
            
            // hillis-steele scan
            if (warpgroup::warpid() <= 1) {
                int tid = warpgroup::laneid();
                for (int offset = 1; offset < 64; offset *= 2) {
                    float temp = (tid >= offset) ? args.scratch.a_cumsum[warpgroupid][tid - offset] : 0.0f;
                    group<2>::sync(warpgroupid + 2);
                    args.scratch.a_cumsum[warpgroupid][tid] += temp;
                    group<2>::sync(warpgroupid + 2);
                }
            }

            // trapezoidal discretization correction.
            build_trapezoidal_scale(args, warpgroupid);

            warpgroup::sync(warpgroupid);

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int base_row = warpgroup::warpid() * 16 + laneid() / 4;
                int base_col = i * 16 + (laneid() % 4) * 2;
                args.state.local_decay.tiles[0][i].data[0].x = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 0];
                args.state.local_decay.tiles[0][i].data[0].y = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 1];
                args.state.local_decay.tiles[0][i].data[1].x = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 0];
                args.state.local_decay.tiles[0][i].data[1].y = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 1];
                args.state.local_decay.tiles[0][i].data[2].x = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 8];
                args.state.local_decay.tiles[0][i].data[2].y = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 9];
                args.state.local_decay.tiles[0][i].data[3].x = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 8];
                args.state.local_decay.tiles[0][i].data[3].y = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 9];
            }
      
            warp::exp(args.state.local_decay, args.state.local_decay);
            apply_trapezoidal_scale(args, warpgroupid);
      
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                auto &decay_subtile = reinterpret_cast<rt_fl<16, 16>&>(args.state.local_decay.tiles[0][i]);
                if (i > warpgroup::warpid()) {
                    warp::zero(decay_subtile);
                } else if (i == warpgroup::warpid()) {
                    warp::make_causal(decay_subtile, decay_subtile, kittens::base_types::constants<float>::zero());
                }
            }
            // A = Q @ K.T
            warpgroup::load(args.state.q_reg, args.input.q);
            // Rotary intentionally disabled in the current TK kernel. q/k are consumed as-is.
            warpgroup::mm_ABt(args.state.att_block, args.state.q_reg, args.input.k);
            warpgroup::mma_async_wait();
            warp::mul(args.state.att_block, args.state.att_block, args.state.local_decay);
            warp::copy(args.state.att_block_mma, args.state.att_block);
            warpgroup::mm_AB(args.state.o_reg, args.state.att_block_mma, args.input.v[warpgroupid]);
            warpgroup::mma_async_wait();
      
            {
                int base_row = warpgroup::warpid() * 16 + laneid() / 4;
                bf16 top = __float2bfloat16(expf(args.scratch.a_cumsum[warpgroupid][base_row + 0]));
                bf16 bottom = __float2bfloat16(expf(args.scratch.a_cumsum[warpgroupid][base_row + 8]));
                #pragma unroll
                for (int i = 0; i < 4; i++) {
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


            warpgroup::store(args.scratch.kv[warpgroupid], args.state.kv);
            warpgroup::sync(warpgroupid);
            warpgroup::mma_AB(args.state.o_reg, args.state.q_reg, args.scratch.kv[warpgroupid]);
            warpgroup::mma_async_wait();
            warpgroup::store(args.output.o[warpgroupid], args.state.o_reg);
            warpgroup::sync(warpgroupid);
      
            float last_decay = args.scratch.a_cumsum[warpgroupid][args.scratch.a_cumsum[warpgroupid].length - 1];
            float total_decay = expf(last_decay);
            warp::mul(args.state.kv, args.state.kv, total_decay); // decay kv
            warpgroup::load(args.state.k_reg, args.input.k);
            {
                int base_row = warpgroup::warpid() * 16 + laneid() / 4;
                bf16 top = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[warpgroupid][base_row + 0]));
                bf16 bottom = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[warpgroupid][base_row + 8]));
                #pragma unroll
                for (int i = 0; i < 4; i++) {
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
      
            warpgroup::store(args.scratch.k[warpgroupid], args.state.k_reg);
            warpgroup::sync(warpgroupid);
            warpgroup::mma_AtB(args.state.kv, args.scratch.k[warpgroupid], args.input.v[warpgroupid]);
            warpgroup::mma_async_wait();

            prefetch_trap_boundary(args, args.iter + 2);

            if (warpgroup::laneid() == 0) {
                arrive(args.outputs_arrived);
                arrive(args.inputs_finished);
            }
            __syncwarp();
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            if(warpgroup::laneid() == 0) arrive(args.finish_finished);
            __syncwarp();
        }
    };
};

using mamba3_siso_fwd_template = mamba3_fwd_common_template<mamba3_siso_layout>;

struct mamba3_mimo_fwd_template : mamba3_fwd_common_template<mamba3_mimo_layout> {
    using base = mamba3_fwd_common_template<mamba3_mimo_layout>;
    using layout = typename base::layout;
    static_assert(MIMO_RANK == 4 && layout::RANKS_PER_PASS == 2,
                  "MIMO pipeline assumes 4 ranks split into 2 passes of 2.");

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int task_id = args.task_iter * gridDim.x + blockIdx.x;
        args.common.batch = task_id / (args.globals.V.depth()/(NUM_CONSUMER_WARPS/4));
        task_id -= args.common.batch*(args.globals.V.depth()/(NUM_CONSUMER_WARPS/4));
        args.common.head = task_id*2;
        int chunk_iters = args.common.batch < args.globals.Q.batch() ? args.globals.K.rows()/layout::k_tile::rows : -1;
        args.num_iters = chunk_iters >= 0 ? chunk_iters * 2 : -1;
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }

        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::warpid() == args.iter % 4) {
                int chunk_iter = args.iter / 2;
                int rank_pass  = args.iter % 2; // 0: ranks 0-1, 1: ranks 2-3
                int rank_base  = rank_pass * layout::RANKS_PER_PASS;

                warp::tma::expect(
                    args.inputs_arrived,
                    args.input.q[0], args.input.k[0],
                    args.input.q[1], args.input.k[1],
                    args.input.v[0], args.input.a[0], args.input.b[0],
                    args.input.v[1], args.input.a[1], args.input.b[1]
                );
                #pragma unroll
                for (int r = 0; r < layout::RANKS_PER_PASS; ++r) {
                    warp::tma::load_async(args.input.q[r], args.globals.Q, {args.common.batch, rank_base + r, chunk_iter, 0}, args.inputs_arrived);
                    warp::tma::load_async(args.input.k[r], args.globals.K, {args.common.batch, rank_base + r, chunk_iter, 0}, args.inputs_arrived);
                }
                #pragma unroll
                for (int i = 0; i < NUM_CONSUMER_WARPS / 4; ++i) {
                    warp::tma::load_async(args.input.v[i], args.globals.V, {args.common.batch, args.common.head + i, chunk_iter, 0}, args.inputs_arrived);
                    warp::tma::load_async(args.input.a[i], args.globals.A, {args.common.batch, args.common.head + i, 0, chunk_iter}, args.inputs_arrived);
                    warp::tma::load_async(args.input.b[i], args.globals.B, {args.common.batch, args.common.head + i, 0, chunk_iter}, args.inputs_arrived);
                }
                __syncwarp();
            }
        }

        __device__ static void store(producer_store_args<layout> args) {
            if (warpgroup::warpid() == args.iter % 4) {
                int rank_pass = args.iter % 2;
                if (rank_pass == 1) {
                    int chunk_iter = args.iter / 2;
                    #pragma unroll
                    for (int i = 0; i < NUM_CONSUMER_WARPS / 4; ++i) {
                        warp::tma::store_async(args.globals.O, args.output.o[i], {args.common.batch, args.common.head + i, chunk_iter, 0});
                    }
                    warp::tma::store_async_read_wait();
                }
                __syncwarp();
                if (laneid() == 0) arrive(args.outputs_finished);
                __syncwarp();
            }
        }
    };


    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<base::NUM_CONSUMER_WARPS/WARPGROUP_WARPS>();
            warp::zero(args.state.kv);
            base::prefetch_trap_boundary(args, 1);
        }

        __device__ static void compute(consumer_compute_args<layout> args) {
            int warpgroupid = warpgroup::groupid();
            int rank_pass = args.iter % 2;

            if (rank_pass == 0) {
                warpgroup::sync(warpgroupid);
                warpgroup::copy(args.scratch.a_cumsum[warpgroupid], args.input.a[warpgroupid]);
                warpgroup::sync(warpgroupid);

                // hillis-steele scan
                if (warpgroup::warpid() <= 1) {
                    int tid = warpgroup::laneid();
                    for (int offset = 1; offset < 64; offset *= 2) {
                        float temp = (tid >= offset) ? args.scratch.a_cumsum[warpgroupid][tid - offset] : 0.0f;
                        group<2>::sync(warpgroupid + 2);
                        args.scratch.a_cumsum[warpgroupid][tid] += temp;
                        group<2>::sync(warpgroupid + 2);
                    }
                }

                base::build_trapezoidal_scale(args, warpgroupid);

                warpgroup::sync(warpgroupid);

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int base_row = warpgroup::warpid() * 16 + laneid() / 4;
                    int base_col = i * 16 + (laneid() % 4) * 2;
                    args.state.local_decay.tiles[0][i].data[0].x = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 0];
                    args.state.local_decay.tiles[0][i].data[0].y = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 1];
                    args.state.local_decay.tiles[0][i].data[1].x = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 0];
                    args.state.local_decay.tiles[0][i].data[1].y = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 1];
                    args.state.local_decay.tiles[0][i].data[2].x = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 8];
                    args.state.local_decay.tiles[0][i].data[2].y = args.scratch.a_cumsum[warpgroupid][base_row + 0] - args.scratch.a_cumsum[warpgroupid][base_col + 9];
                    args.state.local_decay.tiles[0][i].data[3].x = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 8];
                    args.state.local_decay.tiles[0][i].data[3].y = args.scratch.a_cumsum[warpgroupid][base_row + 8] - args.scratch.a_cumsum[warpgroupid][base_col + 9];
                }

                warp::exp(args.state.local_decay, args.state.local_decay);
                base::apply_trapezoidal_scale(args, warpgroupid);

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    auto &decay_subtile = reinterpret_cast<rt_fl<16, 16>&>(args.state.local_decay.tiles[0][i]);
                    if (i > warpgroup::warpid()) {
                        warp::zero(decay_subtile);
                    } else if (i == warpgroup::warpid()) {
                        warp::make_causal(decay_subtile, decay_subtile, kittens::base_types::constants<float>::zero());
                    }
                }

                warp::zero(args.state.o_reg);

                #pragma unroll
                for (int r = 0; r < layout::RANKS_PER_PASS; ++r) {
                    warpgroup::load(args.state.q_reg, args.input.q[r]);
                    warpgroup::mm_ABt(args.state.att_block, args.state.q_reg, args.input.k[r]);
                    warpgroup::mma_async_wait();
                    warp::mul(args.state.att_block, args.state.att_block, args.state.local_decay);
                    warp::copy(args.state.att_block_mma, args.state.att_block);
                    warpgroup::mma_AB(args.state.o_reg, args.state.att_block_mma, args.input.v[warpgroupid]);
                    warpgroup::mma_async_wait();
                }
                warpgroup::store(args.scratch.kv[warpgroupid], args.state.kv);
                warpgroup::sync(warpgroupid);
                #pragma unroll
                for (int r = 0; r < layout::RANKS_PER_PASS; ++r) {
                    warpgroup::load(args.state.q_reg, args.input.q[r]);
                    {
                        int base_row = warpgroup::warpid() * 16 + laneid() / 4;
                        bf16 top = __float2bfloat16(expf(args.scratch.a_cumsum[warpgroupid][base_row + 0]));
                        bf16 bottom = __float2bfloat16(expf(args.scratch.a_cumsum[warpgroupid][base_row + 8]));
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
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
                    warpgroup::mma_AB(args.state.o_reg, args.state.q_reg, args.scratch.kv[warpgroupid]);
                    warpgroup::mma_async_wait();
                }

                float last_decay = args.scratch.a_cumsum[warpgroupid][args.scratch.a_cumsum[warpgroupid].length - 1];
                float total_decay = expf(last_decay);
                warp::mul(args.state.kv, args.state.kv, total_decay);

                #pragma unroll
                for (int r = 0; r < layout::RANKS_PER_PASS; ++r) {
                    warpgroup::load(args.state.k_reg, args.input.k[r]);
                    {
                        int base_row = warpgroup::warpid() * 16 + laneid() / 4;
                        bf16 top = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[warpgroupid][base_row + 0]));
                        bf16 bottom = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[warpgroupid][base_row + 8]));
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
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
                    warpgroup::store(args.scratch.k[warpgroupid], args.state.k_reg);
                    warpgroup::sync(warpgroupid);
                    warpgroup::mma_AtB(args.state.kv, args.scratch.k[warpgroupid], args.input.v[warpgroupid]);
                    warpgroup::mma_async_wait();
                }

            } else {
                #pragma unroll
                for (int r = 0; r < layout::RANKS_PER_PASS; ++r) {
                    warpgroup::load(args.state.q_reg, args.input.q[r]);
                    warpgroup::mm_ABt(args.state.att_block, args.state.q_reg, args.input.k[r]);
                    warpgroup::mma_async_wait();
                    warp::mul(args.state.att_block, args.state.att_block, args.state.local_decay);
                    warp::copy(args.state.att_block_mma, args.state.att_block);
                    warpgroup::mma_AB(args.state.o_reg, args.state.att_block_mma, args.input.v[warpgroupid]);
                    warpgroup::mma_async_wait();
                }

                #pragma unroll
                for (int r = 0; r < layout::RANKS_PER_PASS; ++r) {
                    warpgroup::load(args.state.q_reg, args.input.q[r]);
                    {
                        int base_row = warpgroup::warpid() * 16 + laneid() / 4;
                        bf16 top = __float2bfloat16(expf(args.scratch.a_cumsum[warpgroupid][base_row + 0]));
                        bf16 bottom = __float2bfloat16(expf(args.scratch.a_cumsum[warpgroupid][base_row + 8]));
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
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
                    warpgroup::mma_AB(args.state.o_reg, args.state.q_reg, args.scratch.kv[warpgroupid]);
                    warpgroup::mma_async_wait();
                }

                warpgroup::store(args.output.o[warpgroupid], args.state.o_reg);
                warpgroup::sync(warpgroupid);

                float last_decay = args.scratch.a_cumsum[warpgroupid][args.scratch.a_cumsum[warpgroupid].length - 1];
                #pragma unroll
                for (int r = 0; r < layout::RANKS_PER_PASS; ++r) {
                    warpgroup::load(args.state.k_reg, args.input.k[r]);
                    {
                        int base_row = warpgroup::warpid() * 16 + laneid() / 4;
                        bf16 top = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[warpgroupid][base_row + 0]));
                        bf16 bottom = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[warpgroupid][base_row + 8]));
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
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
                    warpgroup::store(args.scratch.k[warpgroupid], args.state.k_reg);
                    warpgroup::sync(warpgroupid);
                    warpgroup::mma_AtB(args.state.kv, args.scratch.k[warpgroupid], args.input.v[warpgroupid]);
                    warpgroup::mma_async_wait();
                }
            }

            if (rank_pass == 1) {
                base::prefetch_trap_boundary(args, args.iter / 2 + 2);
            }

            if (warpgroup::laneid() == 0) {
                arrive(args.outputs_arrived);
                arrive(args.inputs_finished);
            }
            __syncwarp();

        }

        __device__ static void finish(consumer_finish_args<layout> args) {
            if (warpgroup::laneid() == 0) arrive(args.finish_finished);
            __syncwarp();
        }
    };
};

#ifdef TK_COMPILE_MAMBA3
#include "pyutils/torchutils.cuh"
#include <iostream>
#include <ATen/cuda/CUDAContext.h> 
#include <ATen/Functions.h>

void dispatch_mamba3(
    bf16 *d_q, bf16 *d_k, bf16 *d_v,
    bf16 *d_o, float *d_a, float *d_b, float *d_angle, int B, int H, int N, bool use_mimo
) {
    if (!d_q || !d_k || !d_v || !d_o || !d_a || !d_b || !d_angle) {
        throw std::runtime_error("Null pointer passed to dispatch_mamba3");
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(132, 1, 1);

    if (use_mimo) {
        using layout = mamba3_mimo_fwd_template::layout;

        layout::q_global Qg(d_q, B, MIMO_RANK, N, nullptr);
        layout::k_global Kg(d_k, B, MIMO_RANK, N, nullptr);
        layout::a_global Ag(d_a, B, H, nullptr, N);
        layout::b_global Bg(d_b, B, H, nullptr, N);
        layout::angle_global AngleG(d_angle, B, H, nullptr, N);
        layout::v_global Vg(d_v, B, H, N, nullptr);
        layout::o_global Og(d_o, B, H, N, nullptr);

        layout::globals globals = {Qg, Kg, Vg, Og, Ag, Bg, AngleG};

        unsigned long mem_size = kittens::prototype::detail::MAX_SHARED_MEMORY_v<mamba3_mimo_fwd_template>;
        cudaFuncSetAttribute(
            prototype::lcsf::kernel<mamba3_mimo_fwd_template>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        constexpr int BLOCK_SIZE = prototype::detail::NUM_THREADS_v<mamba3_mimo_fwd_template>;
        prototype::lcsf::kernel<mamba3_mimo_fwd_template><<<grid, BLOCK_SIZE, mem_size, stream>>>(globals);
    } else {
        using layout = mamba3_siso_fwd_template::layout;

        layout::q_global Qg(d_q, B, 1, N, nullptr);
        layout::k_global Kg(d_k, B, 1, N, nullptr);
        layout::a_global Ag(d_a, B, H, nullptr, N);
        layout::b_global Bg(d_b, B, H, nullptr, N);
        layout::angle_global AngleG(d_angle, B, H, nullptr, N);
        layout::v_global Vg(d_v, B, H, N, nullptr);
        layout::o_global Og(d_o, B, H, N, nullptr);

        layout::globals globals = {Qg, Kg, Vg, Og, Ag, Bg, AngleG};

        unsigned long mem_size = kittens::prototype::detail::MAX_SHARED_MEMORY_v<mamba3_siso_fwd_template>;
        cudaFuncSetAttribute(
            prototype::lcsf::kernel<mamba3_siso_fwd_template>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        constexpr int BLOCK_SIZE = prototype::detail::NUM_THREADS_v<mamba3_siso_fwd_template>;
        prototype::lcsf::kernel<mamba3_siso_fwd_template><<<grid, BLOCK_SIZE, mem_size, stream>>>(globals);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel: %s\n", cudaGetErrorString(err));
    }
}

at::Tensor mamba3(
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    const at::Tensor a,
    const at::Tensor b,
    const at::Tensor angle,
    const bool use_mimo
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(angle);

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(angle.is_contiguous(), "angle must be contiguous");

    int B = v.size(0);
    int H = v.size(1);
    int N = v.size(2);
    int D = v.size(3);

    int expected_qk_heads = use_mimo ? MIMO_RANK : 1;
    TORCH_CHECK(q.size(0) == B, "q has incompatible batch");
    TORCH_CHECK(q.size(1) == expected_qk_heads, "q has incompatible heads");
    TORCH_CHECK(q.size(2) == N, "q has incompatible sequence shape");
    TORCH_CHECK(q.size(3) == D, "q has incompatible dimension");

    TORCH_CHECK(k.size(0) == B, "k has incompatible batch");
    TORCH_CHECK(k.size(1) == expected_qk_heads, "k has incompatible heads");
    TORCH_CHECK(k.size(2) == N, "k has incompatible sequence");
    TORCH_CHECK(k.size(3) == D, "k has incompatible dimension");

    TORCH_CHECK(v.size(0) == B, "v has incompatible batch");
    TORCH_CHECK(v.size(1) == H, "v has incompatible heads");
    TORCH_CHECK(v.size(2) == N, "v has incompatible sequence");
    TORCH_CHECK(v.size(3) == D, "v has incompatible dimension");

    TORCH_CHECK(a.size(0) == B, "a has incompatible batch");
    TORCH_CHECK(a.size(1) == H, "a has incompatible heads");
    TORCH_CHECK(a.size(2) == N, "a has incompatible sequence");

    TORCH_CHECK(b.size(0) == B, "b has incompatible batch");
    TORCH_CHECK(b.size(1) == H, "b has incompatible heads");
    TORCH_CHECK(b.size(2) == N, "b has incompatible sequence");

    TORCH_CHECK(angle.size(0) == B, "angle has incompatible batch");
    TORCH_CHECK(angle.size(1) == H, "angle has incompatible heads");
    TORCH_CHECK(angle.size(2) == N, "angle has incompatible sequence");

    auto options = at::TensorOptions()
        .dtype(q.dtype())
        .device(q.device())
        .requires_grad(q.requires_grad());
    at::Tensor out = at::empty({B, H, N, D}, options);

    TORCH_CHECK(out.is_contiguous(), "Output tensor must be contiguous");

    auto q_ptr = q.data_ptr<c10::BFloat16>();
    auto k_ptr = k.data_ptr<c10::BFloat16>();
    auto v_ptr = v.data_ptr<c10::BFloat16>();
    auto a_ptr = a.data_ptr<float>();
    auto b_ptr = b.data_ptr<float>();
    auto angle_ptr = angle.data_ptr<float>();
    auto out_ptr = out.data_ptr<c10::BFloat16>();

    TORCH_CHECK(q_ptr != nullptr, "q data pointer is null");
    TORCH_CHECK(k_ptr != nullptr, "k data pointer is null");
    TORCH_CHECK(v_ptr != nullptr, "v data pointer is null");
    TORCH_CHECK(a_ptr != nullptr, "a data pointer is null");
    TORCH_CHECK(b_ptr != nullptr, "b data pointer is null");
    TORCH_CHECK(angle_ptr != nullptr, "angle data pointer is null");
    TORCH_CHECK(out_ptr != nullptr, "output data pointer is null");

    bf16 *d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16 *d_k = reinterpret_cast<bf16*>(k_ptr);
    bf16 *d_v = reinterpret_cast<bf16*>(v_ptr);
    float *d_a = a_ptr;
    float *d_b = b_ptr;
    float *d_angle = angle_ptr;
    bf16 *d_o = reinterpret_cast<bf16*>(out_ptr);

    dispatch_mamba3(d_q, d_k, d_v, d_o, d_a, d_b, d_angle, B, H, N, use_mimo);
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream());
    CHECK_CUDA_ERROR(cudaGetLastError());

    return out;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mamba3", mamba3, "Mamba3 TK. Takes tensors (q, k, v, a, b, angle, use_mimo). q, k, v tensors are bf16, and a, b, angle are float.");
}

#else
#include "harness.impl"
#endif
