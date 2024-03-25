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
#define CLUSTER_SIZE 1

using namespace kittens;

using layout_row = st_wgmma_row_0b_layout; 
using layout_col = st_wgmma_col_t_0b_layout;

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
attend_ker(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o) 
{
    auto warpid        = kittens::warpid();
    auto warpgroupid   = threadIdx.x / 128;
    auto lane          = kittens::laneid();

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

    __shared__ uint64_t qsmem_barrier[NUM_WORKERS]; 
    __shared__ uint64_t ksmem_barrier[NUM_WORKERS];
    __shared__ uint64_t vsmem_barrier[NUM_WORKERS];
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> q_barrier;
    if (threadIdx.x == 0) {init(&q_barrier, block.size());}

    constexpr int tile_bytes = sizeof(bf16) * k_smem[0][0].num_elements;
    int tile_idx; 

    tma::init_barrier(qsmem_barrier[warpid], block.size());
    tma::set_barrier_bytes(qsmem_barrier[warpid], tile_bytes);

    tma::init_barrier(ksmem_barrier[warpid], block.size());
    tma::set_barrier_bytes(ksmem_barrier[warpid], tile_bytes);

    tma::init_barrier(vsmem_barrier[warpid], block.size());
    tma::set_barrier_bytes(vsmem_barrier[warpid], tile_bytes);

    block.sync();

    int tic = 0, toc = 1;

    tile_idx = ((blockIdx.x) * NUM_WORKERS * qo_blocks) + (0)*NUM_WORKERS + warpid; 
    tma::load_async(q_smem[warpid], tma_q, tile_idx, qsmem_barrier[warpid]);

    if (warpid == 0) {
        for (int w = 0; w < NUM_WORKERS; w++) {
            tile_idx = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + warpid + w;
            tma::load_async(k_smem[tic][w], tma_k, tile_idx, ksmem_barrier[w]); 
            tma::load_async(v_smem[tic][w], tma_v, tile_idx, vsmem_barrier[w]);
        }
    }

    constexpr int kPhaseBit_k = 1; 
    constexpr int kPhaseBit_v = 1;
    constexpr int kPhaseBit_q = 1;

    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

        tma::arrive_wait(qsmem_barrier[warpid], kPhaseBit_q);
        __syncthreads(); 
        load(q_reg, q_smem[warpid]);

        if(q_blk+1 < qo_blocks) {
            tma::init_barrier(qsmem_barrier[warpid], block.size());
            tma::set_barrier_bytes(qsmem_barrier[warpid], tile_bytes);

            if (warpid == 0) {
                for (int w = 0; w < NUM_WORKERS; w++) {
                    tile_idx = ((blockIdx.x) * NUM_WORKERS * qo_blocks) + (q_blk+1)*NUM_WORKERS + warpid + w; 
                    tma::load_async(q_smem[w], tma_q, tile_idx, qsmem_barrier[w]);
                }
            }
        }

        if constexpr (ATTN_D == 64) {
            mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment head 64
        }
        else if constexpr (ATTN_D == 128) {
            mul(q_reg, q_reg, __float2bfloat16(0.08838834764831843f)); // temperature adjustment head 128
        }

        neg_infty(max_vec);
        zero(norm_vec);

        // zero registers
        zero(o_prev);

        for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
            __syncthreads();

            for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {
                warpgroup::fence(att_block);
                tma::arrive_wait(ksmem_barrier[warpid], kPhaseBit_k);
                warpgroup::dot_reset(att_block, q_reg, k_smem[tic][subtile]);
                warpgroup::commit_group();

                copy(norm_vec_last, norm_vec);
                copy(max_vec_last,  max_vec);

                if (subtile == 0) {
                    tma::init_barrier(ksmem_barrier[warpid], block.size());
                    tma::set_barrier_bytes(ksmem_barrier[warpid], tile_bytes);
                }

                if (warpid == 0) {
                    if (kv_idx+1 < kv_blocks) {
                        tile_idx = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + (kv_idx+1)*NUM_WORKERS + warpid + subtile;
                        tma::load_async(k_smem[toc][subtile], tma_k, tile_idx, ksmem_barrier[subtile]); 
                    }
                    else if (q_blk+1 < qo_blocks) {
                        tile_idx = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + warpid + subtile; 
                        tma::load_async(k_smem[toc][subtile], tma_k, tile_idx, ksmem_barrier[subtile]);
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
                tma::arrive_wait(vsmem_barrier[warpid], kPhaseBit_v);
                warpgroup::mma_accum(o_prev, att_block_mma, v_smem[tic][subtile]);
                warpgroup::commit_group();

                if (subtile == 0) {
                    tma::init_barrier(vsmem_barrier[warpid], block.size());
                    tma::set_barrier_bytes(vsmem_barrier[warpid], tile_bytes);
                }

                if (warpid == 0) {
                    if (kv_idx+1 < kv_blocks) {
                        tile_idx = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + (kv_idx+1)*NUM_WORKERS + warpid + subtile; 
                        tma::load_async(v_smem[toc][subtile], tma_v, tile_idx, vsmem_barrier[subtile]);
                    }
                    else if (q_blk+1 < qo_blocks) {
                        tile_idx = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + warpid + subtile; 
                        tma::load_async(v_smem[toc][subtile], tma_v, tile_idx, vsmem_barrier[subtile]);
                    }
                }
            }
            warpgroup::mma_async_wait();

            tic ^= 1;
            toc ^= 1;
        }

        tma::wait_for_store_complete<0>(); 
        store(o_smem[warpid], o_prev);
        
        tile_idx = ((blockIdx.x) * NUM_WORKERS * qo_blocks) + (q_blk)*NUM_WORKERS + warpid;
        tma::store_async(tma_o, o_smem[warpid], tile_idx); 
        tma::commit_group();
    }
    tma::wait_for_store_complete<0>(); 
}

#include "harness_rep.impl"