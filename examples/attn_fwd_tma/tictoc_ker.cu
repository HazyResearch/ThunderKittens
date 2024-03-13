#include <cuda/pipeline>
#include <cooperative_groups.h>

#include "../../src/kittens.cuh"

#define NUM_WORKERS 8
#define NUM_WARPGROUPS (NUM_WORKERS/4)

using namespace kittens;

__global__ void attend_ker(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__, 
                           CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o) 
{
    auto warpid        = kittens::warp_id();
    auto warpgroupid   = threadIdx.x / 128;
    auto lane          = kittens::laneid();
    auto block_start   = blockIdx.x*(n*d);

    const bf16 *_q = __q__ + block_start; // packed type means more!
    const bf16 *_k = __k__ + block_start;
    const bf16 *_v = __v__ + block_start;
          bf16 *_o = __o__ + block_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al = shared_allocator::create_allocator((int*)&__shm[0]); 

    rt_bf_2x4<> q_reg;
    rt_fl_2x2<> att_block;
    rt_bf_2x2<> att_block_mma;
    rt_fl_2x4<> o_prev;
    rt_fl_2x2<>::col_vec max_vec_last, max_vec;
    rt_fl_2x2<>::col_vec norm_vec_last, norm_vec;

    using layout_row = st_wgmma_row_0b_layout;
    st_bf<8,4,layout_row> (&q_smem)[NUM_WARPGROUPS] = al.allocate<st_bf<8,4,layout_row>, NUM_WARPGROUPS>();
    st_bf_2x4<layout_row> (&k_smem)[2][NUM_WORKERS] = al.allocate<st_bf_2x4<layout_row>, 2, NUM_WORKERS>();
    st_bf_2x4<layout_row> (&v_smem)[2][NUM_WORKERS] = al.allocate<st_bf_2x4<layout_row>, 2, NUM_WORKERS>();
    // st_bf_2x4<layout_row> (&o_smem)[NUM_WORKERS]    = al.allocate<st_bf_2x4<layout_row>, NUM_WORKERS>();
    
    int qo_blocks = n / (q_smem[warpgroupid].rows * NUM_WARPGROUPS);
    int kv_blocks = n / (q_reg.rows * NUM_WORKERS);

    auto block = cooperative_groups::this_thread_block();

    int tile_idx_kv = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + warpid;
    int tile_idx_q  = ((blockIdx.x) * NUM_WARPGROUPS * qo_blocks) + warpgroupid;  

    __shared__ uint64_t smem_barrier[NUM_WARPGROUPS + (2 * NUM_WORKERS)];
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> q_barrier;
    if (threadIdx.x == 0) {init(&q_barrier, block.size());}

    constexpr int size_q_bytes  = sizeof(bf16) * q_smem[0].num_elements;
    constexpr int size_kv_bytes = sizeof(bf16) * k_smem[0][0].num_elements; 

    tma::init_barrier(smem_barrier[warpgroupid], block.size());
    tma::set_barrier_bytes(smem_barrier[warpgroupid], size_q_bytes);

    tma::init_barrier(smem_barrier[NUM_WARPGROUPS + warpid], block.size());
    tma::set_barrier_bytes(smem_barrier[NUM_WARPGROUPS + warpid], size_kv_bytes);

    tma::init_barrier(smem_barrier[NUM_WARPGROUPS + NUM_WORKERS + warpid], block.size());
    tma::set_barrier_bytes(smem_barrier[NUM_WARPGROUPS + NUM_WORKERS + warpid], size_kv_bytes);

    block.sync();

    int tic = 0, toc = 1;
    
    warpgroup::load_async(q_smem[warpgroupid], _q + warpgroupid * q_smem[warpgroupid].rows*d, d, q_barrier); //start getting block 0

    tma::load_async(k_smem[tic][warpid], tma_k, tile_idx_kv, smem_barrier[NUM_WARPGROUPS + warpid]);
    tma::load_async(v_smem[tic][warpid], tma_v, tile_idx_kv, smem_barrier[NUM_WARPGROUPS + NUM_WORKERS + warpid]);

    constexpr int kPhaseBit_k = 1; 
    constexpr int kPhaseBit_v = 1;
    constexpr int kPhaseBit_q = 1;

    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

        tma::arrive_wait(smem_barrier[warpgroupid], kPhaseBit_q); 
        q_barrier.arrive_and_wait();

        // __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
        // if (threadIdx.x == 0 && blockIdx.x == 0 && q_blk == 0) {
        //     printf("q_smem[]:");
        //     for (int w = 0; w < 2; w++) {
        //         printf("q_smem[%d]: \n", w);
        //         for (int r = 0; r < q_smem[w].rows; r++) {
        //             for (int c = 0; c < q_smem[w].cols; c++) {
        //                 printf("%f ", __bfloat162float(q_smem[w].data[c + r * q_smem[w].cols]));
        //             }
        //             printf("\n");
        //         }
        //         printf("\n");
        //     }
        // }
        // __syncthreads();
        
        warpgroup::load(q_reg, q_smem[warpgroupid]);
        mul(q_reg, q_reg, __float2bfloat16(0.125f));
        if(q_blk+1 < qo_blocks) {
            if (threadIdx.x % 128 == 0) {
                tile_idx_q = ((blockIdx.x) * NUM_WARPGROUPS * qo_blocks) + (q_blk+1)*NUM_WARPGROUPS + warpgroupid;
                tma::load_async(q_smem[warpgroupid], tma_q, tile_idx_q, smem_barrier[warpgroupid]);
            }
        }

        neg_infty(max_vec);
        zero(norm_vec);

        // zero registers
        zero(o_prev);

        for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

            // kv_barrier.arrive_and_wait(); // wait for the k fragments.
            tma::arrive_wait(smem_barrier[NUM_WARPGROUPS + warpid], kPhaseBit_k);
            tma::arrive_wait(smem_barrier[NUM_WARPGROUPS + NUM_WORKERS + warpid], kPhaseBit_v);
            __syncthreads();

            // __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
            // if (threadIdx.x == 0 && blockIdx.x == 0 && q_blk == 0) {
            //     printf("k_smem[]: \n");
            //     for (int w = 0; w < NUM_WORKERS; w++) {
            //         printf("k_smem[%d]: \n", w);
            //         for (int r = 0; r < k_smem[tic][w].rows; r++) {
            //             for (int c = 0; c < k_smem[tic][w].cols; c++) {
            //                 printf("%f ", __bfloat162float(k_smem[tic][w].data[c + r * k_smem[tic][w].cols]));
            //             }
            //             printf("\n");
            //         }
            //         printf("\n");
            //     }
            //     printf("v_smem[]:");
            //     for (int w = 0; w < NUM_WORKERS; w++) {
            //         printf("v_smem[%d]: \n", w);
            //         for (int r = 0; r < v_smem[tic][w].rows; r++) {
            //             for (int c = 0; c < v_smem[tic][w].cols; c++) {
            //                 printf("%f ", __bfloat162float(v_smem[tic][w].data[c + r * v_smem[tic][w].cols]));
            //             }
            //             printf("\n");
            //         }
            //         printf("\n");
            //     }
            // }
            // __syncthreads();
 
            if(kv_idx+1 < kv_blocks) {
                tile_idx_kv = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + (kv_idx+1)*NUM_WORKERS + warpid;
                tma::load_async(k_smem[toc][warpid], tma_k, tile_idx_kv, smem_barrier[NUM_WARPGROUPS + warpid]);
                tma::load_async(v_smem[toc][warpid], tma_v, tile_idx_kv, smem_barrier[NUM_WARPGROUPS + NUM_WORKERS + warpid]);
            }
            else if(q_blk+1 < qo_blocks) {
                tile_idx_kv = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + warpid;
                tma::load_async(k_smem[toc][warpid], tma_k, tile_idx_kv, smem_barrier[NUM_WARPGROUPS + warpid]);
                tma::load_async(v_smem[toc][warpid], tma_v, tile_idx_kv, smem_barrier[NUM_WARPGROUPS + NUM_WORKERS + warpid]);
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

        // store(o_smem[warpid], o_prev);
        // __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
        // if (threadIdx.x == 0 && blockIdx.x == 0 && q_blk == 0) {
        //     printf("o_smem[]: \n");
        //     for (int w = 0; w < NUM_WORKERS; w++) {
        //         printf("o_smem[%d]: \n", w);
        //         for (int r = 0; r < o_smem[w].rows; r++) {
        //             for (int c = 0; c < o_smem[w].cols; c++) {
        //                 printf("%f ", __bfloat162float(o_smem[w].data[c + r * o_smem[w].cols]));
        //             }
        //             printf("\n");
        //         }
        //         printf("\n");
        //     }
        // }
        // __syncthreads();
 
        store(_o + (q_blk*NUM_WORKERS + warpid) * q_reg.rows*d, o_prev, d);
    } 
}

#include "harness.impl"