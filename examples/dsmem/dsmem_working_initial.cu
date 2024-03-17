#include <cooperative_groups.h>

#include "../../src/kittens.cuh"

#define ATTN_B 16
#define ATTN_H 16
#define ATTN_N 2048
#define ATTN_D 64

#define NUM_WORKERS 8
#define BLOCK_SIZE (32*NUM_WORKERS)

#define Q_ROWS 32
#define CLUSTER_SIZE (ATTN_N/(NUM_WORKERS * Q_ROWS))

using namespace kittens;

using layout_row = st_wgmma_row_0b_layout;
using layout_col = st_wgmma_col_t_0b_layout;

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) 
attend_ker(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__, 
            CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o)
{
    auto warpid        = threadIdx.x / 32;
    
    namespace cg = cooperative_groups;

    auto grid = cg::this_grid();
    auto cluster = cg::this_cluster();
    unsigned int cluster_size = CLUSTER_SIZE; // cluster.num_blocks(); but i want this at compile time
    unsigned int block_idx    = cluster.block_rank();;
    unsigned int cluster_idx  = grid.cluster_rank();

    auto block = cg::this_thread_block();

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al = shared_allocator::create_allocator((int*)&__shm[0]); 

    // layout:
    // index 0: which part of the cache is this? (0,1) are used as a tic-toc for message passing, 2 is async load.
    // index 1: which worker is responsible
    st_bf_2x4<layout_row> (&k_smem)[3][NUM_WORKERS] = al.allocate<st_bf_2x4<layout_row>, 3, NUM_WORKERS>();
    st_bf_2x4<layout_col> (&v_smem)[3][NUM_WORKERS] = al.allocate<st_bf_2x4<layout_col>, 3, NUM_WORKERS>();

    rt_bf_2x4<> q_reg;
    static_assert(Q_ROWS == q_reg.rows);
    rt_fl_2x2<> att_block;
    rt_bf_2x2<> att_block_mma;
    rt_fl_2x4<> o_prev;
    rt_fl_2x2<>::col_vec max_vec_last, max_vec;
    rt_fl_2x2<>::col_vec norm_vec_last, norm_vec;

    //**// General set-up
    int tic = 0, toc = 1;
    int warp_idx  = (cluster_idx * cluster_size + block_idx) * NUM_WORKERS + warpid;

    //**// TMA set-up
    __shared__ uint64_t ksmem_barrier[NUM_WORKERS];
    __shared__ uint64_t vsmem_barrier[NUM_WORKERS];

    constexpr int tma_tile_bytes = sizeof(bf16) * k_smem[0][0].num_elements;

    tma::init_barrier(ksmem_barrier[warpid], block.size());
    tma::set_barrier_bytes(ksmem_barrier[warpid], tma_tile_bytes); 

    tma::init_barrier(vsmem_barrier[warpid], block.size());
    tma::set_barrier_bytes(vsmem_barrier[warpid], tma_tile_bytes); 
    block.sync();

    constexpr int kPhaseBit_k = 1; 
    constexpr int kPhaseBit_v = 1;
    constexpr int kPhaseBit_q = 1; 

    tma::load_async(k_smem[tic][warpid], tma_k, warp_idx, ksmem_barrier[warpid]);
    tma::load_async(v_smem[tic][warpid], tma_v, warp_idx, vsmem_barrier[warpid]);
    // load(k_smem[tic][warpid], __k__ + warp_idx*q_reg.num_elements, ATTN_D);
    // load(v_smem[tic][warpid], __v__ + warp_idx*q_reg.num_elements, ATTN_D);

    //**// DSMEM set-up
    __shared__ uint64_t k_dsmem_barrier[2];
    __shared__ uint64_t v_dsmem_barrier[2];
    
    constexpr int size_bytes = sizeof(bf16) * k_smem[0][0].num_elements * NUM_WORKERS;

    // dsmem works at threadblock level (not using warp)
    for(int i = 0; i < 2; i++) {
        dsmem::init_barrier(k_dsmem_barrier[i], block.size());
        dsmem::set_barrier_bytes(k_dsmem_barrier[i], size_bytes);
        dsmem::init_barrier(v_dsmem_barrier[i], block.size());
        dsmem::set_barrier_bytes(v_dsmem_barrier[i], size_bytes);
    }

    constexpr int kPhaseBit_dsmem_kv = 1;
    constexpr int kPhaseBit_tma_k = 1;
    constexpr int kPhaseBit_tma_v = 1;

    load(q_reg, __q__ + warp_idx*q_reg.num_elements, ATTN_D);
    if constexpr (ATTN_D == 64) {
        mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment head 64
    }
    else if constexpr (ATTN_D == 128) {
        mul(q_reg, q_reg, __float2bfloat16(0.08838834764831843f)); // temperature adjustment head 128
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);

    // load(k_smem[tic][warpid], __k__ + warp_idx*q_reg.num_elements, ATTN_D);
    // load(v_smem[tic][warpid], __v__ + warp_idx*q_reg.num_elements, ATTN_D);
    tma::arrive_wait(ksmem_barrier[warpid], kPhaseBit_tma_k);
    tma::arrive_wait(vsmem_barrier[warpid], kPhaseBit_tma_v);

    cluster.sync(); // make sure all the memory has arrived!
    
    const int &kv_blocks = cluster_size; // just for clarity
    for(auto kv_itr = 0; kv_itr < kv_blocks; kv_itr++) {

        if(kv_itr > 0) {
            dsmem::distribution_wait(k_dsmem_barrier[tic], kPhaseBit_dsmem_kv);
            dsmem::distribution_wait(v_dsmem_barrier[tic], kPhaseBit_dsmem_kv);
        }

        if(kv_itr+1 < kv_blocks) {
            int neighbor_idx = (block_idx + 1) % cluster_size; // pass down by 1
            dsmem::tile_distribute_smem(k_smem[toc][0], k_smem[tic][0], cluster_size, neighbor_idx, size_bytes, k_dsmem_barrier[toc]);
            dsmem::tile_distribute_smem(v_smem[toc][0], v_smem[tic][0], cluster_size, neighbor_idx, size_bytes, v_dsmem_barrier[toc]);
        }

        for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {

            warpgroup::fence(att_block); 
            warpgroup::dot_reset(att_block, q_reg, k_smem[tic][subtile]); 
            warpgroup::commit_group();

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
            warpgroup::commit_group();
            warpgroup::mma_async_wait();
        }

        tic ^= 1;
        toc ^= 1;
        // cluster.sync(); I would think this is necessary but seems to work without it? Saves a lot of time too.
    }

    store(__o__ + warp_idx*q_reg.num_elements, o_prev, ATTN_D);

}

#include "harness.impl"