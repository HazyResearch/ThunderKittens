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

using layout = st_wgmma_row_0b_layout;
// using layout = st_naive_row_layout; 
// using layout = st_tma_row_layout; 
// using layout = st_wgmma_col_t_0b_layout;  

__global__ void attend_ker(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o) 
{
    auto warpid        = kittens::warpid();
    auto warpgroupid   = threadIdx.x / 128;
    auto lane          = kittens::laneid();

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al = shared_allocator::create_allocator((int*)&__shm[0]); 

    // rt_bf_2x4<> q_reg;
    // static_assert(Q_ROWS == q_reg.rows);
    // rt_fl_2x2<> att_block;
    // rt_bf_2x2<> att_block_mma;
    // rt_fl_2x4<> o_prev;
    // rt_fl_2x2<>::col_vec max_vec_last, max_vec;
    // rt_fl_2x2<>::col_vec norm_vec_last, norm_vec;

    st_bf<2,4,layout> (&q_smem)[NUM_WORKERS]    = al.allocate<st_bf<2,4,layout>, NUM_WORKERS>();
    st_bf_2x4<layout> (&k_smem)[2][NUM_WORKERS] = al.allocate<st_bf_2x4<layout>, 2, NUM_WORKERS>();
    st_bf_2x4<layout> (&v_smem)[2][NUM_WORKERS] = al.allocate<st_bf_2x4<layout>, 2, NUM_WORKERS>();
    
    constexpr int qo_blocks = ATTN_N / (Q_ROWS * NUM_WORKERS);
    constexpr int kv_blocks = ATTN_N / (Q_ROWS * NUM_WORKERS);

    auto block = cooperative_groups::this_thread_block();

    __shared__ uint64_t qsmem_barrier[1]; 
    __shared__ uint64_t ksmem_barrier[1];
    __shared__ uint64_t vsmem_barrier[1];

    constexpr int tile_bytes = sizeof(bf16) * k_smem[0][0].num_elements * NUM_WORKERS;
    int tile_idx; 

    if (threadIdx.x == 0) {
        tma::init_barrier(qsmem_barrier[0], block.size());
        tma::set_barrier_bytes(qsmem_barrier[0], tile_bytes);

        tma::init_barrier(ksmem_barrier[0], block.size());
        tma::set_barrier_bytes(ksmem_barrier[0], tile_bytes);

        tma::init_barrier(vsmem_barrier[0], block.size());
        tma::set_barrier_bytes(vsmem_barrier[0], tile_bytes);
    }

    block.sync();

    int tic = 0, toc = 1;

    if (threadIdx.x == 0) {
        for (int i = 0; i < NUM_WORKERS; i++) {
            tile_idx = ((blockIdx.x * NUM_WORKERS * qo_blocks)) + warpid + i; 
            tma::load_async(q_smem[i], tma_q, tile_idx, qsmem_barrier[0]); 

            tile_idx = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + warpid + i;
            tma::load_async(k_smem[tic][i], tma_k, tile_idx, ksmem_barrier[0]); 
            tma::load_async(v_smem[tic][i], tma_v, tile_idx, vsmem_barrier[0]);
        }
    }

    constexpr int kPhaseBit_k = 1; 
    constexpr int kPhaseBit_v = 1;
    constexpr int kPhaseBit_q = 1;

    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

        tma::arrive_wait(qsmem_barrier[0], kPhaseBit_q);  

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
            }

            for (int subtile = 0; subtile < NUM_WORKERS; subtile++) {
                if (threadIdx.x == 0) {
                    tile_idx = tile_idx = ((blockIdx.x) * NUM_WORKERS * kv_blocks) + (kv_idx)*NUM_WORKERS + warpid + subtile; 
                    tma::store_async(tma_k, k_smem[tic][subtile], tile_idx); 
                    tma::commit_group(); 

                    tma::store_async(tma_v, v_smem[tic][subtile], tile_idx); 
                    tma::commit_group(); 
                }
            }
            tma::wait_for_store_complete<0>(); 

            tic ^= 1;
            toc ^= 1;
        }

        for (int i = 0; i < NUM_WORKERS; i++) {
            tile_idx = ((blockIdx.x) * NUM_WORKERS * qo_blocks) + (q_blk)*NUM_WORKERS + warpid + i; 
            
            tma::store_async(tma_q, q_smem[i], tile_idx); 
            tma::commit_group(); 
        }
        tma::wait_for_store_complete<0>(); 
    }
}

#include "harness_sweep.impl"