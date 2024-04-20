#include "src/kittens.cuh"

#define NUM_WORKERS 4 // this comes from the fact that we want a 64-long sliding window
using namespace kittens;

__global__ __launch_bounds__(NUM_WORKERS*kittens::WARP_THREADS, 2)
void sliding_window(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__) {

    using G = kittens::group<NUM_WORKERS>;

    auto warpid        = kittens::warpid();
    auto block_start   = blockIdx.x*(n*64);
    const bf16 *_q = __q__ + block_start, *_k = __k__ + block_start, *_v = __v__ + block_start;
          bf16 *_o = __o__ + block_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&k_smem)[2][NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, 2, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&v_smem)[2][NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, 2, NUM_WORKERS>();

    rt_bf_1x4<> q_reg, k_reg, v_reg;
    rt_fl_1x1<> att_block[5];
    rt_bf_1x1<> att_block_bf;
    rt_fl_1x4<> o_reg;
    rt_fl_1x1<>::col_vec max_vec, norm_vec;
    
    int qo_blocks = n / (q_reg.rows*NUM_WORKERS), kv_blocks = n / (q_reg.rows*NUM_WORKERS);

    int tic = 0, toc = 1;
    for(auto qo_blk = 0; qo_blk < qo_blocks; qo_blk++, tic ^= 1, toc ^= 1) {

        __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk

        // load the curent k, v blocks into toc. If qo_blk > 0, then the previous is in tic.
        load(k_smem[toc][warpid], _k + (qo_blk*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
        load(v_smem[toc][warpid], _v + (qo_blk*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);

        // load q registers
        load(q_reg, _q + (qo_blk*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
        mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment

        neg_infty(max_vec); // zero registers for the Q chunk
        zero(norm_vec);
        zero(o_reg);

        __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

        for(int subtile = 0; subtile < 5; subtile++) {
            int src_idx = warpid+subtile;
            if (qo_blk != 0 || src_idx>=4) {

                load(k_reg, k_smem[tic^(src_idx/4)][src_idx%4]);

                zero(att_block[subtile]);
                dot(att_block[subtile], q_reg, k_reg, att_block[subtile]);
                if(subtile == 4) {
                    // last tile becomes causal
                    make_causal(att_block[subtile], att_block[subtile], base_types::constants<float>::neg_infty());
                }
            }
            else {
                neg_infty(att_block[subtile]); // initial blocks must be zero
            }
        }
        // now do the softmax. first we subtract max for numerical stability. then exp.
        #pragma unroll
        for(int subtile = 0; subtile < 5; subtile++) {
            row_max(max_vec, att_block[subtile], max_vec); // accumulate onto the max_vec
        }
        #pragma unroll
        for(int subtile = 0; subtile < 5; subtile++) {
            sub_row(att_block[subtile], att_block[subtile], max_vec);
            exp(att_block[subtile], att_block[subtile]);
        }
        // now we sum so that we can divide.
        #pragma unroll
        for(int subtile = 0; subtile < 5; subtile++) {
            row_sum(norm_vec, att_block[subtile], norm_vec);
        }
        #pragma unroll
        for(int subtile = 0; subtile < 5; subtile++) {
            div_row(att_block[subtile], att_block[subtile], norm_vec);
        }
        for(int subtile = 0; subtile < 5; subtile++) {
            int src_idx = warpid+subtile;
            if (qo_blk != 0 || src_idx>=4) {
                load(v_reg, v_smem[tic^(src_idx/4)][src_idx%4]);
                rt_bf_1x4<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

                copy(att_block_bf, att_block[subtile]);
                mma(o_reg, att_block_bf, v_reg_col, o_reg); // accumulate
            }
        }

        store(_o + (qo_blk*NUM_WORKERS + warpid)*q_reg.num_elements, o_reg, d); // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
    }
}

#include "harness.impl"