#include <cooperative_groups.h>

#include "../../src/kittens.cuh"

#include "clocker.cuh"

#define ATTN_B 16
#define ATTN_H 16
#define ATTN_N 1024
#define ATTN_D 128

#define NUM_WORKERS 8
#define BLOCK_SIZE (32*NUM_WORKERS)

#define REG_ROWS 32
#define CLUSTER_SIZE (ATTN_N/(NUM_WORKERS * REG_ROWS))

using namespace kittens;

using layout_row = st_wgmma_row_0b_layout;
using layout_col = st_wgmma_col_t_0b_layout;

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) 
attend_ker(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__, TIMINGDETAIL_ARGS) {
    auto warpid        = threadIdx.x / 32;
    
    INITTHREADTIMER();

    namespace cg = cooperative_groups;

    auto grid = cg::this_grid();
    auto cluster = cg::this_cluster();
    unsigned int cluster_size = cluster.num_blocks();
    unsigned int block_idx    = cluster.block_rank();;
    unsigned int cluster_idx  = grid.cluster_rank();

    auto block = cg::this_thread_block();

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al = shared_allocator::create_allocator((int*)&__shm[0]); 

    st_bf<2,8,layout_row> (&q_smem)[NUM_WORKERS] = al.allocate<st_bf<2,8,layout_row>, NUM_WORKERS>();
    st_bf<2,8,layout_row> (&k_smem)[NUM_WORKERS] = al.allocate<st_bf<2,8,layout_row>, NUM_WORKERS>();
    st_bf<2,8,layout_col> (&v_smem)[NUM_WORKERS] = al.allocate<st_bf<2,8,layout_col>, NUM_WORKERS>();

    rt_bf<2,8> q_reg;
    static_assert(q_reg.rows == REG_ROWS);
    rt_fl_2x2<> att_block;
    rt_bf_2x2<> att_block_mma;
    rt_fl<2,8> o_prev;
    rt_fl_2x2<>::col_vec max_vec_last, max_vec;
    rt_fl_2x2<>::col_vec norm_vec_last, norm_vec;

    __shared__ uint64_t k_dsmem_barrier[1];
    __shared__ uint64_t v_dsmem_barrier[1];
    __shared__ uint64_t smem_barrier[NUM_WORKERS];
    
    constexpr int size_bytes = sizeof(bf16) * k_smem[0].num_elements * NUM_WORKERS; 

    // dsmem works at threadblock level (not using warp)
    dsmem::init_barrier(k_dsmem_barrier[0], block.size());
    dsmem::set_barrier_bytes(k_dsmem_barrier[0], size_bytes);

    dsmem::init_barrier(v_dsmem_barrier[0], block.size());
    dsmem::set_barrier_bytes(v_dsmem_barrier[0], size_bytes);

    constexpr int size_kv_bytes = sizeof(bf16) * k_smem[0].num_elements;

    int warp_idx  = (cluster_idx * cluster_size + block_idx) * NUM_WORKERS + warpid;

    constexpr int kPhaseBit_dsmem_k = 1; 
    constexpr int kPhaseBit_dsmem_v = 1;
    constexpr int kPhaseBit_tma_k = 1;
    constexpr int kPhaseBit_tma_v = 1;

    load(q_reg, __q__ + warp_idx*q_reg.num_elements, d);
    // mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment head 64
    mul(q_reg, q_reg, __float2bfloat16(0.08838834764831843f)); // temperature adjustment head 128

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);

    load(v_smem[warpid], __v__ + warp_idx*q_reg.num_elements, d);
    load(k_smem[warpid], __k__ + warp_idx*q_reg.num_elements, d);

    cluster.sync(); // make sure all the memory has arrived!

    CLOCKPOINT(0, "init");
    
    const int &kv_blocks = cluster_size; // just for clarity
    for(auto kv_itr = 0; kv_itr < kv_blocks; kv_itr++) {

        if (kv_itr > 0) {
            int neighbor_idx = (block_idx + kv_itr) % cluster_size;

            dsmem::tile_distribute_smem(k_smem[0], k_smem[0], cluster_size, neighbor_idx, size_bytes, k_dsmem_barrier[0]); 
            dsmem::tile_distribute_smem(v_smem[0], v_smem[0], cluster_size, neighbor_idx, size_bytes, v_dsmem_barrier[0]);

            dsmem::distribution_wait(k_dsmem_barrier[0], kPhaseBit_dsmem_k);
            dsmem::distribution_wait(v_dsmem_barrier[0], kPhaseBit_dsmem_v);
        }
        cluster.sync();
        CLOCKPOINT(1, "load");

        for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {

            warpgroup::fence(att_block); 
            warpgroup::dot_reset(att_block, q_reg, k_smem[subtile]); 
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
            warpgroup::mma_accum(o_prev, att_block_mma, v_smem[subtile]); 
            warpgroup::commit_group();

            warpgroup::mma_async_wait();
        }
        cluster.sync();
        CLOCKPOINT(2, "compute");
    }

    store(__o__ + warp_idx*q_reg.num_elements, o_prev, d);
    CLOCKPOINT(3, "store");

    FINISHTHREADTIMER();
}

#include "harness.impl"