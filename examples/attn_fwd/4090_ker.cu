#include "../../src/kittens.cuh"

#define NUM_WORKERS 16
using namespace kittens;
__global__ void attend_ker(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__) {

    auto warpid        = kittens::warpid();
    auto block_start   = blockIdx.x*(n*64);
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
    
    int qo_blocks = n / (q_reg.rows*NUM_WORKERS), kv_blocks = n / (q_reg.rows*NUM_WORKERS);

    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

        load(q_reg, _q + (q_blk*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
        mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment

        neg_infty(max_vec); // zero registers for the Q chunk
        zero(norm_vec);
        zero(o_prev);

        for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

            load(v_smem[warpid], _v + (kv_idx*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
            load(k_smem[warpid], _k + (kv_idx*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
            __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

            for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {

                load(k_reg, k_smem[subtile]);

                zero(att_block);
                mma_ABt(att_block, q_reg, k_reg, att_block);

                copy(norm_vec_last, norm_vec);
                copy(max_vec_last,  max_vec);

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

                copy(att_block_mma, att_block); // convert to bf16 for mma_AB

                load(v_reg, v_smem[subtile]);
                rt_bf_1x4<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

                mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma_AB'ing onto it
                mma_AB(o_prev, att_block_mma, v_reg_col, o_prev);
            }
            __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
        }

        store(_o + (q_blk*NUM_WORKERS + warpid)*q_reg.num_elements, o_prev, d); // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
    }
}

#include "harness.impl"
