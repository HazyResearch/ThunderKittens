#include <cooperative_groups.h>

#include "../../src/kittens.cuh"


#define NUM_WORKERS 16 // to reduce __syncwarp() stalls.
#define WARPGROUP_SIZE 4
#define NUM_WARPGROUPS (NUM_WORKERS/WARPGROUP_SIZE)

#define QO_BLOCKS 1 // CANNOT change rn

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
__global__ void attend_ker(int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__, 
                            CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o) {

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al = shared_allocator::create_allocator((int*)&__shm[0]);

    st_bf_4x4<layout_row> (&q_smem)[NUM_WARPGROUPS] = al.allocate<st_bf_4x4<layout_row>, NUM_WARPGROUPS>();
    st_bf_1x4<layout_row> (&k_smem)[2][NUM_WORKERS] = al.allocate<st_bf_1x4<layout_row>, NUM_WORKERS, 2>();
    st_bf_1x4<layout_col> (&v_smem)[2][NUM_WORKERS] = al.allocate<st_bf_1x4<layout_col>, NUM_WORKERS, 2>();
    st_bf_1x4<layout_row> (&o_smem)[NUM_WORKERS]    = al.allocate<st_bf_1x4<layout_row>, NUM_WORKERS>(); 

    int tic = 0; 
    int toc = 1; 
 
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;
    rt_fl_1x4<> o_prev;
    rt_fl_1x1<>::col_vec max_vec_last, max_vec;
    rt_fl_1x1<>::col_vec norm_vec_last, norm_vec;

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/WARPGROUP_SIZE; 

    constexpr int kv_blocks = N / (NUM_WARPGROUPS*q_smem[0].rows);
    auto cluster_start   = blockIdx.y*(N*64);
    auto block_start     = blockIdx.x*(QO_BLOCKS*NUM_WARPGROUPS*q_smem[0].num_elements) + cluster_start;
    const bf16 *_q = __q__ + block_start, *_k = __k__ + cluster_start, *_v = __v__ + cluster_start;
          bf16 *_o = __o__ + block_start;

    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> q_barrier;
    if (threadIdx.x == 0) {init(&q_barrier, block.size());}

    __shared__ uint64_t qsmem_barrier[NUM_WARPGROUPS]; 
    __shared__ uint64_t ksmem_barrier[NUM_WORKERS];
    __shared__ uint64_t vsmem_barrier[NUM_WORKERS];

    constexpr int tile_bytes = sizeof(bf16) * k_smem[0][0].num_elements;
    int tile_idx;

    tma::init_barrier(qsmem_barrier[warpgroupid], block.size()); 
    tma::set_barrier_bytes(qsmem_barrier[warpgroupid], tile_bytes * WARPGROUP_SIZE); 

    tma::init_barrier(ksmem_barrier[warpid], 32);
    tma::set_barrier_bytes(ksmem_barrier[warpid], tile_bytes);

    tma::init_barrier(vsmem_barrier[warpid], 32);
    tma::set_barrier_bytes(vsmem_barrier[warpid], tile_bytes);

    constexpr int kPhaseBit_q = 1;
    constexpr int kPhaseBit_k = 1; 
    constexpr int kPhaseBit_v = 1;

    block.sync();

    if (warpid == 0) {
        for (int wg = 0; wg < 4; wg++) {
            tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid + wg; 
            tma::load_async(q_smem[wg], tma_q, tile_idx, qsmem_barrier[wg]); 
        }
        
        for (int w = 0; w < NUM_WORKERS; w++) {        
            tile_idx = (blockIdx.y * NUM_WORKERS * kv_blocks) + (0 * NUM_WORKERS) + warpid + w; 
            tma::load_async(k_smem[tic][w], tma_k, tile_idx, ksmem_barrier[w]); 
            tma::load_async(v_smem[tic][w], tma_v, tile_idx, vsmem_barrier[w]); 
        }
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);

    tma::arrive_and_wait(qsmem_barrier[warpgroupid], kPhaseBit_q); 
    __syncthreads(); 
    warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f));

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
        for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {
            warpgroup::fence(att_block);
            tma::arrive_and_wait(ksmem_barrier[warpid], kPhaseBit_k); 
            warpgroup::dot_reset(att_block, q_smem[warpgroupid], k_smem[tic][subtile]);
            warpgroup::mma_commit_group();

            copy(norm_vec_last, norm_vec);
            copy(max_vec_last,  max_vec);

            if (subtile == 0) {
                tma::init_barrier(ksmem_barrier[warpid], 32); 
                tma::set_barrier_bytes(ksmem_barrier[warpid], tile_bytes); 
            }

            if (warpid == 0) {
                if (kv_idx + 1 < kv_blocks) {
                    tile_idx = (blockIdx.y * NUM_WORKERS * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS) + warpid + subtile; 
                    tma::load_async(k_smem[toc][subtile], tma_k, tile_idx, ksmem_barrier[subtile]);
                }
            }

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
            tma::arrive_and_wait(vsmem_barrier[warpid], kPhaseBit_v); 
            warpgroup::mma_accum(o_prev, att_block_mma, v_smem[tic][subtile]);
            warpgroup::mma_commit_group();

            if (subtile == 0) {
                tma::init_barrier(vsmem_barrier[warpid], 32);
                tma::set_barrier_bytes(vsmem_barrier[warpid], tile_bytes);
            }

            if (warpid == 0) {
                if (kv_idx+1 < kv_blocks) {
                    tile_idx = ((blockIdx.y) * NUM_WORKERS * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS) + warpid + subtile; 
                    tma::load_async(v_smem[toc][subtile], tma_v, tile_idx, vsmem_barrier[subtile]);
                }
            }
        }

        tic ^= 1;
        toc ^= 1;
    }

    store(o_smem[warpid], o_prev);

    tile_idx = ((blockIdx.y) * NUM_WORKERS * blockDim.x) + (blockIdx.x)*NUM_WORKERS + warpid; 
    tma::store_async(tma_o, o_smem[warpid], tile_idx); 
    
    tma::store_commit_group(); 
    tma::store_async_wait();
}

#include "harness.impl"