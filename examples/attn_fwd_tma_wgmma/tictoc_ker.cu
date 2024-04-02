#include <cooperative_groups.h>

#include "../../src/kittens.cuh"


#define NUM_WORKERS 16 // to reduce __syncwarp() stalls.
#define WARPGROUP_SIZE 4
#define NUM_WARPGROUPS (NUM_WORKERS/WARPGROUP_SIZE)

#define QO_BLOCKS 1 // CANNOT change rn

// shared tile
#define qo_height 4
#define kv_height 4
#define NUM_WORKERS_KV (NUM_WORKERS/kv_height)

// register tile
#define width  (ATTN_D/16)

#define ATTN_B 16
#define ATTN_H 16
#define ATTN_N 4096
#define ATTN_D 64 // hardcoded into this kernel
#define BLOCK_SIZE (32*NUM_WORKERS)

#define KITTENS_HOPPER

using namespace kittens;

using layout_row = ducks::st_layout::wgmma_row_0b;
using layout_col = ducks::st_layout::wgmma_col_t_0b;

template<int N>
__global__ void attend_ker(int d, CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o) {

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al = shared_allocator::create_allocator((int*)&__shm[0]);

    st_bf<qo_height, width, layout_row> (&q_smem)[NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, width, layout_row>, NUM_WARPGROUPS>();
    st_bf<kv_height, width, layout_row> (&k_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, width, layout_row>, NUM_WORKERS_KV, 2>();
    st_bf<kv_height, width, layout_col> (&v_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, width, layout_col>, NUM_WORKERS_KV, 2>();

    int tic = 0; 
    int toc = 1; 
 
    rt_fl<1, kv_height> att_block;
    rt_bf<1, kv_height> att_block_mma;
    rt_fl<1, width> o_prev;
    rt_fl<1, kv_height>::col_vec max_vec_last, max_vec;
    rt_fl<1, kv_height>::col_vec norm_vec_last, norm_vec;

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/WARPGROUP_SIZE; 

    constexpr int kv_blocks = N / (NUM_WORKERS_KV*k_smem[0][0].rows);
    auto block = cooperative_groups::this_thread_block();

    __shared__ uint64_t qsmem_barrier[1]; 
    __shared__ uint64_t ksmem_barrier[1];
    __shared__ uint64_t vsmem_barrier[1];

    constexpr int tile_bytes = sizeof(bf16) * k_smem[0][0].num_elements * NUM_WORKERS_KV;
    int tile_idx;

    if (warpid == 0) {
        tma::init_barrier(qsmem_barrier[0], block.size()); 
        tma::set_barrier_bytes(qsmem_barrier[0], tile_bytes); 

        tma::init_barrier(ksmem_barrier[0], block.size());
        tma::set_barrier_bytes(ksmem_barrier[0], tile_bytes);

        tma::init_barrier(vsmem_barrier[0], block.size());
        tma::set_barrier_bytes(vsmem_barrier[0], tile_bytes);
    }

    constexpr int kPhaseBit_q = 1;
    constexpr int kPhaseBit_k = 1; 
    constexpr int kPhaseBit_v = 1;

    block.sync();

    if (warpid == 0) {
        for (int wg = 0; wg < 4; wg++) {
            tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid + wg; 
            tma::load_async(q_smem[wg], tma_q, tile_idx, qsmem_barrier[0]); 
        }
        
        for (int w = 0; w < NUM_WORKERS_KV; w++) {        
            tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + (0 * NUM_WORKERS_KV) + warpid + w; 
            tma::load_async(k_smem[tic][w], tma_k, tile_idx, ksmem_barrier[0]); 
            tma::load_async(v_smem[tic][w], tma_v, tile_idx, vsmem_barrier[0]); 
        }
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);

    tma::arrive_and_wait(qsmem_barrier[0], kPhaseBit_q); 
    __syncthreads(); 
    warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f));

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

        tma::arrive_and_wait(ksmem_barrier[0], kPhaseBit_k); 
        tma::arrive_and_wait(vsmem_barrier[0], kPhaseBit_v); 

        if (threadIdx.x == 0) {
            tma::init_barrier(ksmem_barrier[0], block.size()); 
            tma::set_barrier_bytes(ksmem_barrier[0], tile_bytes); 

            tma::init_barrier(vsmem_barrier[0], block.size()); 
            tma::set_barrier_bytes(vsmem_barrier[0], tile_bytes); 
        }
        __syncthreads(); 

        if ((kv_idx + 1 < kv_blocks) && (warpid == 0)) {
            for (int w = 0; w < NUM_WORKERS_KV; w++) {        
                tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS_KV) + warpid + w; 
                tma::load_async(k_smem[toc][w], tma_k, tile_idx, ksmem_barrier[0]); 
                tma::load_async(v_smem[toc][w], tma_v, tile_idx, vsmem_barrier[0]); 
            }
        }

        for(int subtile = 0; subtile < NUM_WORKERS_KV; subtile++) {
            warpgroup::fence(att_block);
            warpgroup::dot_reset(att_block, q_smem[warpgroupid], k_smem[tic][subtile]);
            warpgroup::mma_commit_group();

            copy(norm_vec_last, norm_vec);
            copy(max_vec_last,  max_vec);

            warpgroup::mma_async_wait();

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
            mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma'ing onto it

            warpgroup::fence(o_prev);
            warpgroup::mma_accum(o_prev, att_block_mma, v_smem[tic][subtile]);
            warpgroup::mma_commit_group();
        }

        tic ^= 1;
        toc ^= 1;
    }

    warpgroup::store(q_smem[warpgroupid], o_prev); 

    if (warpid == 0) {
        for (int wg = 0; wg < 4; wg++) {
            tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid + wg; 
            tma::store_async(tma_o, q_smem[wg], tile_idx); 
            tma::store_commit_group(); 
        }
    }

    tma::store_async_wait();
}

#include "harness.impl"