#include "kittens.cuh"

 // this kernel is more of an example kernel to show some TK programming models, rather than a kernel we think you should put into production, though it is pretty fast!

#define NUM_WORKERS 8 // This kernel uses 16 workers in parallel per block, to help issue instructions more quickly.

using namespace kittens;

constexpr int H = 2, R=H*16; // height of tiles
using shared_tile = st_bf<H,4>;
using global_layout = gl<bf16, -1, -1, -1, 64>; // B, H, N specified at runtime, D=64 known at compile time for this kernel

__launch_bounds__(NUM_WORKERS*32, 1)
__global__ void attend_ker64(global_layout Qg, global_layout Kg, global_layout Vg, global_layout Og) {

    const int N = Qg.rows; // sequence length
    auto workerid = kittens::warpid(); // which worker am I?
    const int batch = blockIdx.z;
    const int head  = blockIdx.y;
    const int q_seq = blockIdx.x * NUM_WORKERS + workerid;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    // K and V live in shared memory -- this is about all that will fit.
    shared_tile (&k_smem)[NUM_WORKERS] = al.allocate<shared_tile, NUM_WORKERS>();
    shared_tile (&v_smem)[NUM_WORKERS] = al.allocate<shared_tile, NUM_WORKERS>();

    // Initialize all of the register tiles.
    rt_bf<H,4> q_reg, k_reg, v_reg; // v_reg need to be swapped into col_l
    rt_fl<H,H> att_block;
    rt_bf<H,H> att_block_mma;
    rt_fl<H,4> o_reg;
    rt_fl<H,H>::col_vec max_vec_last, max_vec; // these are column vectors for the attention block
    rt_fl<H,H>::col_vec norm_vec_last, norm_vec; // these are column vectors for the attention block

    // each warp loads its own Q tile of 16x64, and then multiplies by 1/sqrt(d)
    if (q_seq*R < N) load(q_reg, Qg, {batch, head, q_seq, 0});

    // zero flash attention L, M, and O registers.
    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_reg);

    mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment
    
    const int kv_blocks = (N + NUM_WORKERS*R - 1) / (NUM_WORKERS*R);

    // iterate over k, v for these q's that have been loaded
    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

        // each warp loads its own chunk of k, v into shared memory
        int load_idx = kv_idx*NUM_WORKERS + workerid;
        if (load_idx*R < N) {
            load(k_smem[workerid], Kg, {batch, head, load_idx, 0});
            load(v_smem[workerid], Vg, {batch, head, load_idx, 0});
        }
        __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

        // now each warp goes through all of the subtiles, loads them, and then does the flash attention internal alg.
        int max_subtile = min(NUM_WORKERS, N/R - kv_idx*NUM_WORKERS);
        for(int subtile = 0; subtile < max_subtile; subtile++) {

            load(k_reg, k_smem[subtile]); // load k from shared into registers

            zero(att_block); // zero 16x16 attention tile
            mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T

            copy(norm_vec_last, norm_vec);
            copy(max_vec_last,  max_vec);

            row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
            sub_row(att_block, att_block, max_vec); // subtract max from attention -- now all <=0
            exp(att_block, att_block); // exponentiate the block in-place.

            sub(max_vec_last, max_vec_last, max_vec); // subtract new max from old max to find the new normalization.
            exp(max_vec_last, max_vec_last); // exponentiate this vector -- this is what we need to normalize by.
            mul(norm_vec, norm_vec, max_vec_last); // and the norm vec is now normalized.

            row_sum(norm_vec, att_block, norm_vec); // accumulate the new attention block onto the now-rescaled norm_vec
            div_row(att_block, att_block, norm_vec); // now the attention block is correctly normalized

            mul(norm_vec_last, norm_vec_last, max_vec_last); // normalize the previous norm vec according to the new max
            div(norm_vec_last, norm_vec_last, norm_vec); // normalize the previous norm vec according to the new norm

            copy(att_block_mma, att_block); // convert to bf16 for mma_AB

            load(v_reg, v_smem[subtile]); // load v from shared into registers.
            rt_bf<H, 4, col_l> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

            mul_row(o_reg, o_reg, norm_vec_last); // normalize o_reg in advance of mma_AB'ing onto it
            mma_AB(o_reg, att_block_mma, v_reg_col, o_reg); // mfma onto o_reg with the local attention@V matmul.
        }
        __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
    }

    if (q_seq*R < N) store(Og, o_reg, {batch, head, q_seq, 0}); // write out o.
}

#include "harness.impl"
