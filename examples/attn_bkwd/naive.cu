#include "../../src/kittens.cuh"

#define NUM_WORKERS 8
using namespace kittens;
__global__ void attend_ker_bwd(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, const bf16* __restrict__ __o__, const bf16* __restrict__ __o_grad__, 
                                bf16* __restrict__ __q_grad__, bf16* __restrict__ __k_grad__, bf16* __restrict__ __v_grad__) {

    auto warpid        = kittens::warpid();
    auto block_start   = blockIdx.x*(n*64);

    const bf16 *_q = __q__ + block_start; 
    const bf16 *_k = __k__ + block_start; 
    const bf16 *_v = __v__ + block_start;
    const bf16 *_o = __o__ + block_start;
    const bf16 *_o_grad = __o_grad__ + block_start;

    bf16 *_q_grad = __q_grad__ + block_start;
    bf16 *_k_grad = __k_grad__ + block_start;
    bf16 *_v_grad = __v_grad__ + block_start;

    extern __shared__ int __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    st_bf_1x4<ducks::st_layout::xor_swizzle> (&q_smem)  [NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&o_smem)  [NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&og_smem) [NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();

    rt_bf_1x4<> q_reg, k_reg, v_reg; 
    rt_bf_1x4<> o_reg, og_reg;
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;

    rt_fl_1x1<> dOV_block;

    rt_fl_1x4<> q_grad; 
    rt_fl_1x4<> k_grad;
    rt_fl_1x4<> v_grad;

    rt_fl_1x4<> o_prev;
    rv<bf16, 1, 2> d_col_vec; 
    rt_fl_1x1<>::col_vec max_vec_last, max_vec;
    rt_fl_1x1<>::col_vec norm_vec_last, norm_vec;
    
    int qo_blocks = n / (q_reg.rows*NUM_WORKERS), kv_blocks = n / (q_reg.rows*NUM_WORKERS);

    for (auto kv_blk = 0; kv_blk < kv_blocks; kv_blk++) {

        load(v_reg, _v + (kv_blk*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
        load(k_reg, _k + (kv_blk*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);

        zero(k_grad);
        zero(v_grad);

        neg_infty(max_vec); // zero registers for the Q chunk
        zero(norm_vec);

        for (auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

            load(q_smem[warpid],  _q +      (q_blk*NUM_WORKERS + warpid)*q_smem[0].num_elements,  q_smem[0].cols);
            load(o_smem[warpid],  _o +      (q_blk*NUM_WORKERS + warpid)*o_smem[0].num_elements,  o_smem[0].cols);
            load(og_smem[warpid], _o_grad + (q_blk*NUM_WORKERS + warpid)*og_smem[0].num_elements, og_smem[0].cols);
            __syncthreads();

            for (auto subtile = 0; subtile < NUM_WORKERS; subtile++) {
                // prep for QK
                load(q_reg, q_smem[subtile]);
                mul(q_reg, q_reg, __float2bfloat16(1.0f/sqrtf(d)));
                zero(att_block);
                // QK
                dot(att_block, q_reg, k_reg, att_block);

                // FMAS
                copy(norm_vec_last, norm_vec);
                copy(max_vec_last,  max_vec);
                // 
                row_max(max_vec, att_block, max_vec);
                sub_row(att_block, att_block, max_vec);
                exp(att_block, att_block);
                //
                sub(max_vec_last, max_vec_last, max_vec);
                exp(max_vec_last, max_vec_last);
                mul(norm_vec, norm_vec, max_vec_last);
                //
                row_sum(norm_vec, att_block, norm_vec);
                div_row(att_block, att_block, norm_vec);
                //
                mul(norm_vec_last, norm_vec_last, max_vec_last);
                div(norm_vec_last, norm_vec_last, norm_vec);

                // prep for v grad comp
                copy(att_block_mma, att_block);
                load(og_reg, og_smem[subtile]);
                // unsafe prep for v grad comp
                rt_bf_1x1<ducks::rt_layout::col> &att_block_mma_col = swap_layout_inplace(att_block_mma);
                auto* att_block_row_cast = reinterpret_cast<rt_bf_1x1<>*>(&att_block_mma_col.tiles[0][0].data[0]);
                auto* og_reg_col_cast    = reinterpret_cast<rt_bf_1x4<ducks::rt_layout::col>*>(&og_reg.tiles[0][0].data[0]);
                // v grad comp
                mma(v_grad, *att_block_row_cast, *og_reg_col_cast, v_grad);

                // // prep for dOV
                // zero(dOV_block);
                // // dOV
                // dot(dOV_block, og_reg, v_reg, dOV_block);

                // // FMAS
                // zero(d_col_vec); 
                // load(o_reg, o_smem[subtile]);
                // mul(o_reg, o_reg, og_reg);
                // // row_sum(d_col_vec, o_reg);
                // // //
                // // sub_row(dOV_block, dOV_block, d_col_vec);
                // mul(att_block, att_block, dOV_block);

                // // unsafe prep for q grad comp
                // rt_bf_1x1<ducks::rt_layout::row> &att_block_mma_2 = swap_layout_inplace(att_block_mma_col);
                // copy(att_block_mma_2, att_block);
                // auto* k_reg_col_cast = reinterpret_cast<rt_bf_1x4<ducks::rt_layout::col>*>(&k_reg.tiles[0][0].data[0]);
                // load(q_grad, _q_grad + (q_blk*NUM_WORKERS + warpid)*q_grad.num_elements, q_grad.cols);
                // // q grad comp + STORE
                // mma(q_grad, att_block_mma_2, *k_reg_col_cast, q_grad);
                // store(_q_grad + (q_blk*NUM_WORKERS + warpid)*q_grad.num_elements, q_grad, q_grad.cols);

                // // unsafe prep for k grad comp
                // rt_bf_1x1<ducks::rt_layout::col> &att_block_mma_2_col = swap_layout_inplace(att_block_mma_2);
                // auto* att_block_mma_2_row_cast = reinterpret_cast<rt_bf_1x1<>*>(&att_block_mma_2_col.tiles[0][0].data[0]);
                // auto* q_reg_col_cast = reinterpret_cast<rt_bf_1x4<ducks::rt_layout::col>*>(&q_reg.tiles[0][0].data[0]);
                // // k grad comp
                // mma(k_grad, *att_block_mma_2_row_cast, *q_reg_col_cast, k_grad);
            }
        }

        // store k and v grads
        store(_k_grad + (kv_blk*NUM_WORKERS + warpid)*k_grad.num_elements, k_grad, k_grad.cols);
        store(_v_grad + (kv_blk*NUM_WORKERS + warpid)*v_grad.num_elements, v_grad, v_grad.cols);
    }
}

#include "harness.impl"
