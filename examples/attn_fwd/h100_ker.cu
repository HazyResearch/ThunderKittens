#include <cuda/pipeline>
#include <cooperative_groups.h>

#define KITTENS_HOPPER // just in case the user forgot to put it in compilation flags

#include "../../src/kittens.cuh"

#define NUM_WORKERS 8
#define NUM_WARPGROUPS (NUM_WORKERS/4)

using namespace kittens;

__global__ void attend_ker(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__) {

    auto warpid        = threadIdx.x / 32;
    auto warpgroupid   = threadIdx.x / 128;
    auto lane          = threadIdx.x % 32;
    auto block_start   = blockIdx.x*(n*d);

    const bf16 *_q = __q__ + block_start; // packed type means more!
    const bf16 *_k = __k__ + block_start;
    const bf16 *_v = __v__ + block_start;
          bf16 *_o = __o__ + block_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    rt_bf_2x4<> q_reg;
    rt_fl_2x2<> att_block;
    rt_bf_2x2<> att_block_mma;
    rt_fl_2x4<> o_prev;
    rt_fl_2x2<>::col_vec max_vec_last, max_vec;
    rt_fl_2x2<>::col_vec norm_vec_last, norm_vec;

    using layout = st_wgmma_row_32b_layout;
    st_bf<8,4,layout> (&q_smem)[NUM_WARPGROUPS] = al.allocate<st_bf<8,4,layout>, NUM_WARPGROUPS>();
    st_bf_2x4<layout> (&k_smem)[2][NUM_WORKERS] = al.allocate<st_bf_2x4<layout>, 2, NUM_WORKERS>();
    st_bf_2x4<layout> (&v_smem)[2][NUM_WORKERS] = al.allocate<st_bf_2x4<layout>, 2, NUM_WORKERS>();
    
    int qo_blocks = n / (q_smem[warpgroupid].rows*NUM_WARPGROUPS);
    int kv_blocks = n / (q_reg.rows*NUM_WORKERS);

    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> q_barrier;
    if (threadIdx.x == 0) {init(&q_barrier, block.size());}
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> kv_barrier;
    if (threadIdx.x == 0) {init(&kv_barrier, block.size());}
    block.sync();

    int tic = 0, toc = 1;

    warpgroup::load_async(q_smem[warpgroupid], _q + warpgroupid * q_smem[warpgroupid].rows*d, d, q_barrier); //start getting block 0

    load_async(k_smem[tic][warpid], _k + warpid * q_reg.rows*d, d, kv_barrier);
    load_async(v_smem[tic][warpid], _v + warpid * q_reg.rows*d, d, kv_barrier);

    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

        q_barrier.arrive_and_wait();
        warpgroup::load(q_reg, q_smem[warpgroupid]);
        mul(q_reg, q_reg, __float2bfloat16(0.125f));
        if(q_blk+1 < qo_blocks) {
            warpgroup::load_async(q_smem[warpgroupid], _q + ((q_blk+1)*NUM_WARPGROUPS + warpgroupid) * q_smem[warpgroupid].rows*d, d, q_barrier); // stride is a whole row of the array, in bytes
        }

        neg_infty(max_vec);
        zero(norm_vec);

        // zero registers
        zero(o_prev);

        for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

            kv_barrier.arrive_and_wait(); // wait for the k fragments.
 
            if(kv_idx+1 < kv_blocks) {
                load_async(k_smem[toc][warpid], _k + ((kv_idx+1)*NUM_WORKERS + warpid) * q_reg.rows*d, d, kv_barrier);
                load_async(v_smem[toc][warpid], _v + ((kv_idx+1)*NUM_WORKERS + warpid) * q_reg.rows*d, d, kv_barrier);
            }
            else if(q_blk+1 < qo_blocks) {
                load_async(k_smem[toc][warpid], _k + warpid * q_reg.rows*d, d, kv_barrier);
                load_async(v_smem[toc][warpid], _v + warpid * q_reg.rows*d, d, kv_barrier);
            }

            for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {

                rt_bf_2x4 local_reg;

                load(local_reg, k_smem[tic][subtile]);

                zero(att_block);
                dot(att_block, q_reg, local_reg, att_block);

                copy(norm_vec_last, norm_vec);
                copy(max_vec_last,  max_vec);

                row_max(max_vec, att_block, max_vec); // max-accumulate ONTO the max_vec
                sub_row(att_block, att_block, max_vec);
                exp(att_block, att_block);

                sub(max_vec_last, max_vec_last, max_vec);
                exp(max_vec_last, max_vec_last);
                mul(norm_vec, norm_vec, max_vec_last);

                row_sum(norm_vec, att_block, norm_vec); // add-accumulate ONTO the norm_vec
                div_row(att_block, att_block, norm_vec);

                mul(norm_vec_last, norm_vec_last, max_vec_last);
                div(norm_vec_last, norm_vec_last, norm_vec);

                copy(att_block_mma, att_block);
                
                load(local_reg, v_smem[tic][subtile]);
                rt_bf_2x4<rt_col_layout> &v_reg_col = swap_layout_inplace(local_reg); // this is a reference and the call has invalidated v_reg

                mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma'ing onto it
                mma(o_prev, att_block_mma, v_reg_col, o_prev);
            }

            tic ^= 1;
            toc ^= 1;
        }

        store(_o + (q_blk*NUM_WORKERS + warpid) * q_reg.rows*d, o_prev, d);
    }
}

#include "harness.impl"