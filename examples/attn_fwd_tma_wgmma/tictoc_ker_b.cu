#include <cuda/pipeline>
#include <cooperative_groups.h>

#include "../../src/kittens.cuh"

#define NUM_WORKERS 8
#define NUM_WARPGROUPS (NUM_WORKERS/4)

#define ATTN_B 16
#define ATTN_H 16
#define ATTN_N (1024 * 4)
#define ATTN_D 64

#define Q_ROWS 32

using namespace kittens;

using layout_row = st_wgmma_row_0b_layout; 
using layout_col = st_wgmma_col_t_0b_layout;

__global__ void attend_ker(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__, 
                           CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o) 
{
    auto warpid        = kittens::warpid();
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
    static_assert(Q_ROWS == q_reg.rows);
    rt_fl_2x2<> att_block;
    rt_bf_2x2<> att_block_mma;
    rt_fl_2x4<> o_prev;
    rt_fl_2x2<>::col_vec max_vec_last, max_vec;
    rt_fl_2x2<>::col_vec norm_vec_last, norm_vec;

    st_bf<2,4,layout_row> (&q_smem)[NUM_WORKERS]    = al.allocate<st_bf<2,4,layout_row>, NUM_WORKERS>();
    st_bf_2x4<layout_row> (&k_smem)[2][NUM_WORKERS] = al.allocate<st_bf_2x4<layout_row>, 2, NUM_WORKERS>();
    st_bf_2x4<layout_col> (&v_smem)[2][NUM_WORKERS] = al.allocate<st_bf_2x4<layout_col>, 2, NUM_WORKERS>();
    st_bf_2x4<layout_row> (&o_smem)[NUM_WORKERS]    = al.allocate<st_bf_2x4<layout_row>, NUM_WORKERS>(); 
    
    constexpr int qo_blocks = ATTN_N / (Q_ROWS * NUM_WORKERS);
    constexpr int kv_blocks = ATTN_N / (Q_ROWS * NUM_WORKERS);

    auto block = cooperative_groups::this_thread_block();

    __shared__ uint64_t qsmem_barrier[1]; 
    __shared__ uint64_t ksmem_barrier[1];
    __shared__ uint64_t vsmem_barrier[1];
    // __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> q_barrier;
    // if (threadIdx.x == 0) {init(&q_barrier, block.size());}

    constexpr int tile_bytes = sizeof(bf16) * k_smem[0][0].num_elements * NUM_WORKERS;
    int tile_idx; 

    int tic = 0, toc = 1;

    if (threadIdx.x == 0) {
        tma::init_barrier(qsmem_barrier[0], block.size());
        tma::set_barrier_bytes(qsmem_barrier[0], tile_bytes);

        tma::init_barrier(ksmem_barrier[0], block.size());
        tma::set_barrier_bytes(ksmem_barrier[0], tile_bytes);

        tma::init_barrier(vsmem_barrier[0], block.size());
        tma::set_barrier_bytes(vsmem_barrier[0], tile_bytes);

        for (int i = 0; i < NUM_WORKERS; i++) {
            tile_idx = ((blockIdx.x) * NUM_WORKERS * qo_blocks) + warpid + i; 
            tma::load_async(q_smem[i], tma_q, tile_idx, qsmem_barrier[0]); 

            tile_idx = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + warpid + i;
            tma::load_async(k_smem[tic][i], tma_k, tile_idx, ksmem_barrier[0]); 
            tma::load_async(v_smem[tic][i], tma_v, tile_idx, vsmem_barrier[0]);
        }
    }
    block.sync();

    // load_async(q_smem[warpid], _q + warpid * q_smem[warpid].num_elements, d, q_barrier);

    constexpr int kPhaseBit_k = 1; 
    constexpr int kPhaseBit_v = 1;
    constexpr int kPhaseBit_q = 1;

    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

        tma::arrive_wait(qsmem_barrier[0], kPhaseBit_q);
        // q_barrier.arrive_and_wait();

        // __syncthreads(); 
        // if (threadIdx.x == 0 && blockIdx.x == 0 && q_blk == 0) {
        //     //print out qsmem
        //     for (int w = 0; w < NUM_WORKERS; w++) {
        //         for (int i = 0; i < q_smem[w].rows; i++) {
        //             for (int j = 0; j < q_smem[w].cols; j++) {
        //                 printf("%f ", __bfloat162float(q_smem[w].data[i * q_smem[w].cols + j]));
        //             }
        //             printf("\n");
        //         }
        //         printf("\n");
        //     }
        // }
        // __syncthreads(); 

        load(q_reg, q_smem[warpid]);

        if constexpr (ATTN_D == 64) {
            mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment head 64
        }
        else if constexpr (ATTN_D == 128) {
            mul(q_reg, q_reg, __float2bfloat16(0.08838834764831843f)); // temperature adjustment head 128
        }

        if(q_blk+1 < qo_blocks) {
            if (threadIdx.x == 0) {
                tma::init_barrier(qsmem_barrier[0], block.size());
                tma::set_barrier_bytes(qsmem_barrier[0], tile_bytes);

                for (int i = 0; i < NUM_WORKERS; i++) {
                    tile_idx = ((blockIdx.x) * NUM_WORKERS * qo_blocks) + (q_blk+1)*NUM_WORKERS + warpid + i; 
                    tma::load_async(q_smem[i], tma_q, tile_idx, qsmem_barrier[0]);
                }
            }
        }

        neg_infty(max_vec);
        zero(norm_vec);

        // zero registers
        zero(o_prev);

        tma::arrive_wait(ksmem_barrier[0], kPhaseBit_k);
        tma::arrive_wait(vsmem_barrier[0], kPhaseBit_v);

        for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

            tma::arrive_wait(ksmem_barrier[0], kPhaseBit_k);
            tma::arrive_wait(vsmem_barrier[0], kPhaseBit_v);
            if (threadIdx.x == 0) {
                tma::init_barrier(ksmem_barrier[0], block.size());
                tma::set_barrier_bytes(ksmem_barrier[0], tile_bytes);

                tma::init_barrier(vsmem_barrier[0], block.size());
                tma::set_barrier_bytes(vsmem_barrier[0], tile_bytes);
            }
            __syncthreads();

            for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {
                warpgroup::fence(att_block);
                warpgroup::dot_reset(att_block, q_reg, k_smem[tic][subtile]);
                warpgroup::commit_group();

                copy(norm_vec_last, norm_vec);
                copy(max_vec_last,  max_vec);

                if ((kv_idx+1 < kv_blocks || q_blk+1 < qo_blocks)) {
                    if (threadIdx.x == 0) {
                        if (kv_idx+1 < kv_blocks) {
                            tile_idx = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + (kv_idx+1)*NUM_WORKERS + warpid + subtile; 
                        }
                        else if (q_blk+1 < qo_blocks) {
                            tile_idx = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + warpid + subtile; 
                        }
                        tma::load_async(k_smem[toc][subtile], tma_k, tile_idx, ksmem_barrier[0]);
                        tma::load_async(v_smem[toc][subtile], tma_v, tile_idx, vsmem_barrier[0]);
                    }
                }

                warpgroup::mma_async_wait();

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

                mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma'ing onto it

                warpgroup::fence(o_prev);
                warpgroup::mma_accum(o_prev, att_block_mma, v_smem[tic][subtile]);
                warpgroup::commit_group();
            }

            tic ^= 1;
            toc ^= 1;
        }

        store(o_smem[warpid], o_prev);

        tile_idx = ((blockIdx.x) * NUM_WORKERS * qo_blocks) + (q_blk)*NUM_WORKERS + warpid; 
        tma::store_async(tma_o, o_smem[warpid], tile_idx); 

        tma::commit_group(); 
        tma::wait_for_store_complete<0>();
    } 
}

#include "harness_b.impl"