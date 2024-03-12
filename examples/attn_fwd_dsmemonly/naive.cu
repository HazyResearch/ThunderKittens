#include <cooperative_groups.h>

#include "../../src/kittens.cuh"

#define NUM_WORKERS 16

#define CLUSTER_SIZE (1024/(8 * 2 * 16))

using namespace kittens;


__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) 
attend_ker(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__) {

    auto warpid        = threadIdx.x / 32;
    
    // auto block_start   = blockIdx.x*(n*d);
    // const bf16 *_q = __q__ + block_start, *_k = __k__ + block_start, *_v = __v__ + block_start;
    //       bf16 *_o = __o__ + block_start;
    namespace cg = cooperative_groups;

    auto cluster = cg::this_cluster();
    unsigned int cluster_size = cluster.num_blocks();
    unsigned int block_idx    = blockIdx.x % cluster_size;
    unsigned int cluster_idx  = blockIdx.x / cluster_size;

    auto block = cg::this_thread_block();

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al = shared_allocator::create_allocator((int*)&__shm[0]); 
    
    st_bf_1x4<st_xor_row_layout> (&k_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<st_xor_row_layout>, NUM_WORKERS>();
    st_bf_1x4<st_xor_row_layout> (&v_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<st_xor_row_layout>, NUM_WORKERS>();

    rt_bf_1x4<> q_reg, k_reg, v_reg;
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;
    rt_fl_1x4<> o_prev;
    rt_fl_1x1<>::col_vec max_vec_last, max_vec;
    rt_fl_1x1<>::col_vec norm_vec_last, norm_vec;

    __shared__ uint64_t k_dsmem_barrier[1];
    __shared__ uint64_t v_dsmem_barrier[1];
    constexpr int size_bytes = sizeof(bf16) * k_smem[0].num_elements * NUM_WORKERS; 

    // dsmem works at threadblock level (not using warp)
    dsmem::init_barrier(k_dsmem_barrier[0], block.size());
    dsmem::set_barrier_bytes(k_dsmem_barrier[0], size_bytes);

    dsmem::init_barrier(v_dsmem_barrier[0], block.size());
    dsmem::set_barrier_bytes(v_dsmem_barrier[0], size_bytes);

    block.sync(); 
    cluster.sync(); 

    constexpr int kPhaseBit_dsmem_k = 1; 
    constexpr int kPhaseBit_dsmem_v = 1;
    
    int qo_blocks = n / (q_reg.rows * NUM_WORKERS), kv_blocks = n / (q_reg.rows*NUM_WORKERS);

    // for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

    int q_idx_cluster   = (cluster_idx * NUM_WORKERS * qo_blocks); 
    int q_idx_warpid = q_idx_cluster + (block_idx * NUM_WORKERS) + warpid;

    int kv_idx_cluster = (cluster_idx * NUM_WORKERS * kv_blocks);
    int kv_idx_warp    = kv_idx_cluster + (block_idx * NUM_WORKERS) + warpid;

    load(q_reg, __q__ + (q_idx_warpid * q_reg.rows * d), d);
    mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);

    load(v_smem[warpid], __v__ + (kv_idx_warp)*q_reg.rows*d, d);
    load(k_smem[warpid], __k__ + (kv_idx_warp)*q_reg.rows*d, d);
    // __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     // print out kv_idx_warp
    //     printf("kv_idx_warp: %d\n", kv_idx_warp);
    //     for (int w = 0; w < NUM_WORKERS; w++) {
    //         printf("v_smem[%d]: \n", w);
    //         for (int r = 0; r < v_smem[w].rows; r++) {
    //             for (int c = 0; c < v_smem[w].cols; c++) {
    //                 printf("%f ", __bfloat162float(v_smem[w].data[c + r * v_smem[w].cols]));
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads(); 

    for(auto kv_itr = 0; kv_itr < kv_blocks; kv_itr++) {

        if (kv_itr > 0) {
            int neighbor_idx = (block_idx - 1) % cluster_size;

            dsmem::tile_distribute_smem(k_smem[0], k_smem[0], cluster_size, neighbor_idx, size_bytes, k_dsmem_barrier[0]); 
            dsmem::tile_distribute_smem(v_smem[0], v_smem[0], cluster_size, neighbor_idx, size_bytes, v_dsmem_barrier[0]);

            dsmem::distribution_wait(k_dsmem_barrier[0], kPhaseBit_dsmem_k);
            dsmem::distribution_wait(v_dsmem_barrier[0], kPhaseBit_dsmem_v);
        }

        for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {

            load(k_reg, k_smem[subtile]);

            zero(att_block);
            dot(att_block, q_reg, k_reg, att_block);

            copy(norm_vec_last, norm_vec);
            copy(max_vec_last,  max_vec);

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

            load(v_reg, v_smem[subtile]);
            rt_bf_1x4<rt_col_layout> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

            mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma'ing onto it
            mma(o_prev, att_block_mma, v_reg_col, o_prev);
        }
        __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
    }

    cluster.sync();

    int o_idx_warpid = q_idx_warpid; 
    store(__o__ + o_idx_warpid * q_reg.rows*d, o_prev, d);

    // store(_o + (q_blk*NUM_WORKERS + warpid)*q_reg.rows*d, o_prev, d); // write out o
    // }
}

#include "harness.impl"