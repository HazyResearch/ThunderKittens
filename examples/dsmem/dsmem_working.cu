#include <cooperative_groups.h>

#include "../../src/kittens.cuh"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define ATTN_B 16
#define ATTN_H 16
#define ATTN_N (4096 * 8)
#define ATTN_D 128

#define NUM_WORKERS 8
#define BLOCK_SIZE (32*NUM_WORKERS)

#define Q_ROWS 16
#define CLUSTER_SIZE MIN(ATTN_N / (NUM_WORKERS * Q_ROWS), 16)

using namespace kittens;

template<typename T> __device__ inline void swap(T & a, T & b) { T tmp = a; a = b; b = tmp; }

using layout_row = st_wgmma_row_0b_layout;
using layout_col = st_wgmma_col_t_0b_layout;

// head dim 64 to 128
// tma only (no wgmma) with swizzling
// tma stores
// tma threadblock level
// sweep seqlens

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) 
attend_ker(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__,
    CUtensorMap *q_desc, CUtensorMap *k_desc, CUtensorMap *v_desc, CUtensorMap *o_desc)
{
    auto warpid        = threadIdx.x / 32;
    
    namespace cg = cooperative_groups;

    auto grid = cg::this_grid();
    auto cluster = cg::this_cluster();

    unsigned int cluster_size = CLUSTER_SIZE; // cluster.num_blocks(); but i want this at compile time
    unsigned int block_idx    = cluster.block_rank();;
    unsigned int cluster_idx  = grid.cluster_rank();

    static_assert(CLUSTER_SIZE <= 16, "CLUSTER_SIZE must not exceed 16");

    auto block = cg::this_thread_block();

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al = shared_allocator::create_allocator((int*)&__shm[0]); 

    // layout:
    // index 0: which part of the cache is this? (0,1) are used as a tic-toc for message passing, 2 is async load.
    // index 1: which worker is responsible
    st_bf_1x8<layout_row> (&k_smem)[3][NUM_WORKERS] = al.allocate<st_bf_1x8<layout_row>, 3, NUM_WORKERS>();
    st_bf_1x8<layout_col> (&v_smem)[3][NUM_WORKERS] = al.allocate<st_bf_1x8<layout_col>, 3, NUM_WORKERS>();
    st_bf_1x8<layout_row> (&o_smem)[NUM_WORKERS]    = al.allocate<st_bf_1x8<layout_row>, NUM_WORKERS>();

    rt_bf_1x8<> q_reg;
    static_assert(Q_ROWS == q_reg.rows);
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;
    rt_fl_1x8<> o_prev;
    rt_fl_1x1<>::col_vec max_vec_last, max_vec;
    rt_fl_1x1<>::col_vec norm_vec_last, norm_vec;

    //**// General set-up
    int tic = 0, toc = 1, async = 2;
    int warp_idx  = (cluster_idx * cluster_size + block_idx) * NUM_WORKERS + warpid;

    //**// TMA set-up
    __shared__ uint64_t k_tma_barrier[NUM_WORKERS]; 
    __shared__ uint64_t v_tma_barrier[NUM_WORKERS];

    constexpr int tma_bytes = sizeof(bf16) * k_smem[0][0].num_elements;
    constexpr int kPhaseBit_tma = 1;
    constexpr int vPhaseBit_tma = 1;

    //**// DSMEM set-up
    __shared__ uint64_t k_dsmem_barrier[3];
    __shared__ uint64_t v_dsmem_barrier[3];

    constexpr int size_bytes = sizeof(bf16) * k_smem[0][0].num_elements * NUM_WORKERS;
    constexpr int kPhaseBit_dsmem_kv = 0;

    int qo_blocks = ATTN_N / (Q_ROWS * NUM_WORKERS * cluster_size);
    int kv_blocks = ATTN_N / (Q_ROWS * NUM_WORKERS * cluster_size); 

    for (auto q_itr = 0; q_itr < qo_blocks; q_itr++) {
        warp_idx = (cluster_idx * cluster_size * NUM_WORKERS) + (block_idx * NUM_WORKERS) + (0 * (NUM_WORKERS * cluster_size)) + warpid;

        tma::init_barrier(k_tma_barrier[warpid], block.size());
        tma::set_barrier_bytes(k_tma_barrier[warpid], tma_bytes);

        tma::init_barrier(v_tma_barrier[warpid], block.size());
        tma::set_barrier_bytes(v_tma_barrier[warpid], tma_bytes);

        tma::load_async(k_smem[async][warpid], k_desc, warp_idx, k_tma_barrier[warpid]);
        tma::load_async(v_smem[async][warpid], v_desc, warp_idx, v_tma_barrier[warpid]);

        warp_idx = (cluster_idx * cluster_size * NUM_WORKERS) + (block_idx * NUM_WORKERS) + (q_itr * (NUM_WORKERS * cluster_size)) + warpid;
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

        for (auto kv_itr = 0; kv_itr < kv_blocks; kv_itr++) {
            tma::arrive_wait(k_tma_barrier[warpid], kPhaseBit_tma);
            tma::arrive_wait(v_tma_barrier[warpid], vPhaseBit_tma);

            swap(tic, async);

            cluster.sync(); // make sure all the memory has arrived! 

            if (kv_itr + 1 < kv_blocks) {
                warp_idx = (cluster_idx * cluster_size * NUM_WORKERS) + (block_idx * NUM_WORKERS) + ((kv_itr + 1) * (NUM_WORKERS * cluster_size)) + warpid;

                tma::init_barrier(k_tma_barrier[warpid], block.size());
                tma::set_barrier_bytes(k_tma_barrier[warpid], tma_bytes);

                tma::init_barrier(v_tma_barrier[warpid], block.size());
                tma::set_barrier_bytes(v_tma_barrier[warpid], tma_bytes);

                tma::load_async(k_smem[async][warpid], k_desc, warp_idx, k_tma_barrier[warpid]);
                tma::load_async(v_smem[async][warpid], v_desc, warp_idx, v_tma_barrier[warpid]);
            }

            for (auto kv_block = 0; kv_block < cluster_size; kv_block++) {

                if(kv_block > 0) {
                    if (threadIdx.x == 0) {
                        dsmem::distribution_wait(k_dsmem_barrier[tic], kPhaseBit_dsmem_kv);
                        dsmem::distribution_wait(v_dsmem_barrier[tic], kPhaseBit_dsmem_kv);
                    }
                }
                cluster.sync(); 
        
                if(kv_block+1 < cluster_size) {
                    
                    dsmem::init_barrier(k_dsmem_barrier[toc], 1);
                    dsmem::init_barrier(v_dsmem_barrier[toc], 1);

                    dsmem::set_barrier_bytes(k_dsmem_barrier[toc], size_bytes);
                    dsmem::set_barrier_bytes(v_dsmem_barrier[toc], size_bytes);

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
                swap(tic, toc);
            }
        }

        warp_idx = (cluster_idx * cluster_size * NUM_WORKERS) + (block_idx * NUM_WORKERS) + (q_itr * (NUM_WORKERS * cluster_size)) + warpid;
        store(__o__ + warp_idx*q_reg.num_elements, o_prev, ATTN_D);
    }
}

#include "harness_working_initial.impl"