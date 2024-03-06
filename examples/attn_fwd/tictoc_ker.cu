#include <cuda/pipeline>
#include <cooperative_groups.h>

#include "../../src/kittens.cuh"

#define NUM_WORKERS 8
#define NUM_WARPGROUPS (NUM_WORKERS/4)

using namespace kittens;

/**
 * @brief Kernel function to attend to the input queries, keys, and values.
 *
 * This function computes the attention mechanism in parallel using CUDA warps and shared memory.
 * It leverages Tensor Cores for efficient matrix operations on the BF16 data type.
 * The attention computation involves scaling the queries, computing the dot product with keys,
 * applying softmax, and then computing the weighted sum with values to produce the output.
 *
 * @param n Number of elements in the input arrays.
 * @param d Dimensionality of the input arrays.
 * @param __q__ Pointer to the input queries array.
 * @param __k__ Pointer to the input keys array.
 * @param __v__ Pointer to the input values array.
 * @param __o__ Pointer to the output array.
 */
__global__ void attend_ker(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__) {

    auto warpid        = threadIdx.x / 32;
    auto warpgroupid   = threadIdx.x / 128;
    auto lane          = threadIdx.x % 32;
    auto block_start   = blockIdx.x*(n*d);

    const bf16 *_q = __q__ + block_start; // packed type means more!
    const bf16 *_k = __k__ + block_start;
    const bf16 *_v = __v__ + block_start;
          bf16 *_o = __o__ + block_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    rt_bf_2x4<> q_reg;
    rt_fl_2x2<> att_block;
    rt_bf_2x2<> att_block_mma;
    rt_fl_2x4<> o_prev;
    rt_fl_2x2<>::col_vec max_vec_last, max_vec;
    rt_fl_2x2<>::col_vec norm_vec_last, norm_vec;

    using layout = st_wgmma_row_32b_layout;
    // Allocate shared memory for query, key, and value matrices using the shared_allocator
    // These shared memory tiles will be used to store the input queries, keys, and values
    // for processing by the Tensor Cores in a warp-specialized manner.
    st_bf<8,4,layout> (&q_smem)[NUM_WARPGROUPS] = al.allocate<st_bf<8,4,layout>, NUM_WARPGROUPS>();
    st_bf_2x4<layout> (&k_smem)[2][NUM_WORKERS] = al.allocate<st_bf_2x4<layout>, 2, NUM_WORKERS>();
    st_bf_2x4<layout> (&v_smem)[2][NUM_WORKERS] = al.allocate<st_bf_2x4<layout>, 2, NUM_WORKERS>();

    int qo_blocks = n / (q_smem[warpgroupid].rows*NUM_WARPGROUPS);
    int kv_blocks = n / (q_reg.rows*NUM_WORKERS);

    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> q_barrier;
    if (threadIdx.x == 0) {init(&q_barrier, block.size());}
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> kv_barrier;
    if (threadIdx.x == 0) {init(&kv_barrier, block.size());}
    block.sync();

    int tic = 0, toc = 1;

    // Asynchronously load the first block of queries into shared memory
    tma::load_async(q_smem[warpgroupid], _q + warpgroupid * q_smem[warpgroupid].rows*d, d, q_barrier);

    // Asynchronously load the first blocks of keys and values into shared memory
    tma::load_async(k_smem[tic][warpid], _k + warpid * q_reg.rows*d, d, kv_barrier);
    tma::load_async(v_smem[tic][warpid], _v + warpid * q_reg.rows*d, d, kv_barrier);

    // Main processing loop for the attention mechanism computation
    // This loop iterates over blocks of queries and computes the attention mechanism
    // in parallel using the loaded shared memory tiles for queries, keys, and values.
    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

        q_barrier.arrive_and_wait();
        warpgroup::load(q_reg, q_smem[warpgroupid]);
        mul(q_reg, q_reg, __float2bfloat16(0.125f)); // Scale the queries
        if(q_blk+1 < qo_blocks) {
            // Asynchronously load the next block of queries into shared memory
            tma::load_async(q_smem[warpgroupid], _q + ((q_blk+1)*NUM_WARPGROUPS + warpgroupid) * q_smem[warpgroupid].rows*d, d, q_barrier);
        }

        neg_infty(max_vec);
        zero(norm_vec);

        // Initialize the output register to zero
        zero(o_prev);

        // Loop over blocks of keys and values to compute the attention scores
        // This loop computes the dot product of queries and keys, applies softmax,
        // and computes the weighted sum with values to produce the output.
        for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

            kv_barrier.arrive_and_wait(); // wait for the k and v fragments to be loaded.

            if(kv_idx+1 < kv_blocks) {
                // Asynchronously load the next blocks of keys and values into shared memory
                tma::load_async(k_smem[toc][warpid], _k + ((kv_idx+1)*NUM_WORKERS + warpid) * q_reg.rows*d, d, kv_barrier);
                tma::load_async(v_smem[toc][warpid], _v + ((kv_idx+1)*NUM_WORKERS + warpid) * q_reg.rows*d, d, kv_barrier);
            }
            else if(q_blk+1 < qo_blocks) {
                // Asynchronously load the first blocks of keys and values for the next query block
                tma::load_async(k_smem[toc][warpid], _k + warpid * q_reg.rows*d, d, kv_barrier);
                tma::load_async(v_smem[toc][warpid], _v + warpid * q_reg.rows*d, d, kv_barrier);
            }

            for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {

                rt_bf_2x4 local_reg;

                load(local_reg, k_smem[tic][subtile]);

                zero(att_block);
                dot(att_block, q_reg, local_reg, att_block); // Compute the dot product of queries and keys

                copy(norm_vec_last, norm_vec);
                copy(max_vec_last,  max_vec);

                row_max(max_vec, att_block, max_vec); // Compute the row-wise max to use for numerical stability in softmax
                sub_row(att_block, att_block, max_vec); // Subtract the max from the attention block
                exp(att_block, att_block); // Apply the exponential function to the attention block

                sub(max_vec_last, max_vec_last, max_vec); // Update the max vector for the next iteration
                exp(max_vec_last, max_vec_last); // Apply the exponential function to the max vector
                mul(norm_vec, norm_vec, max_vec_last); // Update the normalization vector

                row_sum(norm_vec, att_block, norm_vec); // Compute the row-wise sum for normalization
                div_row(att_block, att_block, norm_vec); // Normalize the attention block

                mul(norm_vec_last, norm_vec_last, max_vec_last); // Update the normalization vector for the previous iteration
                div(norm_vec_last, norm_vec_last, norm_vec); // Normalize the previous normalization vector

                copy(att_block_mma, att_block); // Copy the attention block for matrix multiplication

                load(local_reg, v_smem[tic][subtile]);
                rt_bf_2x4<rt_col_layout> &v_reg_col = swap_layout_inplace(local_reg); // Swap the layout of the value register for matrix multiplication

                mul_row(o_prev, o_prev, norm_vec_last); // Normalize the previous output register
                mma(o_prev, att_block_mma, v_reg_col, o_prev); // Perform matrix multiplication and accumulate the result
            }

            tic ^= 1;
            toc ^= 1;
        }

        // Store the computed output block back to global memory
        store(_o + (q_blk*NUM_WORKERS + warpid) * q_reg.rows*d, o_prev, d);
    }
}

#include "harness.impl"
