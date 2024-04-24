#include "../../src/kittens.cuh"
#include <cuda/pipeline>

#define HEAD_DIM (64)
#define NUM_WORKERS (16)
#define QO_ITERS (2)

using namespace kittens;
template<int N> __global__  __launch_bounds__(NUM_WORKERS*kittens::WARP_THREADS, 1)
__global__ void attend_ker(const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__) {

    auto warpid        = kittens::warpid();
    auto block_start   = blockIdx.y*(N*64);
    const bf16 *_q = __q__ + block_start, *_k = __k__ + block_start, *_v = __v__ + block_start;
          bf16 *_o = __o__ + block_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&k_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&v_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();

    rt_bf_1x4<> q_reg, k_reg, v_reg;
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;
    rt_fl_1x4<> o_prev;
    rt_fl_1x1<>::col_vec max_vec_last, max_vec;
    rt_fl_1x1<>::col_vec norm_vec_last, norm_vec;

    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> kv_barrier;
    if (threadIdx.x == 0) {init(&kv_barrier, NUM_WORKERS*kittens::WARP_THREADS);}
    __syncthreads();
    if(warpid < NUM_WORKERS/2) {
        load_async(k_smem[warpid], _k + warpid*q_reg.num_elements, HEAD_DIM, kv_barrier);
        load_async(v_smem[warpid], _v + warpid*q_reg.num_elements, HEAD_DIM, kv_barrier);
    }
    
    constexpr int kv_blocks = N / (k_reg.rows*NUM_WORKERS);

    for(auto qo_idx = 0; qo_idx < QO_ITERS; qo_idx++) {

        load(q_reg, _q + ((blockIdx.x*QO_ITERS + qo_idx)*NUM_WORKERS + warpid)*q_reg.num_elements, HEAD_DIM);
        mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment

        neg_infty(max_vec); // zero registers for the Q chunk
        zero(norm_vec);
        zero(o_prev);

        for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

            for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {

                // simple poor man's async pipeline turns out to be good enough on the 4090
                if(subtile == 0) {
                    kv_barrier.arrive_and_wait();
                    if(warpid >= NUM_WORKERS/2) { // load the other half
                        load_async(k_smem[warpid], _k + (kv_idx*NUM_WORKERS + warpid)*q_reg.num_elements, HEAD_DIM, kv_barrier);
                        load_async(v_smem[warpid], _v + (kv_idx*NUM_WORKERS + warpid)*q_reg.num_elements, HEAD_DIM, kv_barrier);
                    }
                }
                else if(subtile == NUM_WORKERS/2) {
                    kv_barrier.arrive_and_wait();
                    if(warpid < NUM_WORKERS/2) { // load the other half
                        load_async(k_smem[warpid], _k + (((kv_idx+1)%kv_blocks)*NUM_WORKERS + warpid)*q_reg.num_elements, HEAD_DIM, kv_barrier);
                        load_async(v_smem[warpid], _v + (((kv_idx+1)%kv_blocks)*NUM_WORKERS + warpid)*q_reg.num_elements, HEAD_DIM, kv_barrier);
                    }
                }

                load(k_reg, k_smem[subtile]);

                copy(norm_vec_last, norm_vec);
                copy(max_vec_last,  max_vec);

                zero(att_block);
                dot(att_block, q_reg, k_reg, att_block);

                row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
                sub_row(att_block, att_block, max_vec);
                exp(att_block, att_block);

                sub(max_vec_last, max_vec_last, max_vec);
                exp(max_vec_last, max_vec_last);
                mul(norm_vec, norm_vec, max_vec_last);

                row_sum(norm_vec, att_block, norm_vec); // accumulate onto the norm_vec
                div_row(att_block, att_block, norm_vec);

                mul(norm_vec_last, norm_vec_last, max_vec_last);
                div(norm_vec_last, norm_vec_last, norm_vec);

                copy(att_block_mma, att_block); // convert to bf16 for mma

                load(v_reg, v_smem[subtile]);
                rt_bf_1x4<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

                mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma'ing onto it
                mma(o_prev, att_block_mma, v_reg_col, o_prev);
            }
        }

        store(_o + ((blockIdx.x*QO_ITERS + qo_idx)*NUM_WORKERS + warpid)*q_reg.num_elements, o_prev, HEAD_DIM);
    }
}

#include "harness_4090_fwd.impl"
