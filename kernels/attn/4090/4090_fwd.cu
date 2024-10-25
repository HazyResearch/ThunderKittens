#include "kittens.cuh"

 // this kernel is more of an example kernel to show some TK programming models, rather than a kernel we think you should put into production, though it is pretty fast!

#define NUM_WORKERS 8 // This kernel uses 16 workers in parallel per block, to help issue instructions more quickly.
#define NUM_WARPS   (NUM_WORKERS) // This kernel uses 16 workers in parallel per block, to help issue instructions more quickly.

using namespace kittens;

constexpr int R = 32, D = 64; // height of tiles
template<typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, R, D, L>;
template<typename T=float> using attn_tile = rt<T, R, R>;
using shared_tile = st_bf<R, D>;
using global_layout = gl<bf16, -1, -1, -1, D>; // B, H, N specified at runtime, D=64 known at compile time for this kernel
struct globals {
    global_layout Qg, Kg, Vg, Og;
};

__launch_bounds__(NUM_WORKERS*32, 1)
__global__ void attend_ker64(const __grid_constant__ globals g) {
    
    const int N = g.Qg.rows; // sequence length
    int workerid = kittens::warpid(); // which worker am I?
    using load_group = kittens::group<2>;
    int loadid = load_group::groupid();
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;
    const int batch = blockIdx.z;
    const int head  = blockIdx.y;
    const int q_seq = blockIdx.x * NUM_WORKERS + workerid;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    // K and V live in shared memory -- this is about all that will fit.
    shared_tile (&k_smem)[LOAD_BLOCKS][2] = al.allocate<shared_tile, LOAD_BLOCKS, 2>();
    shared_tile (&v_smem)[LOAD_BLOCKS][2] = al.allocate<shared_tile, LOAD_BLOCKS, 2>();

    shared_tile (&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile(&)[NUM_WORKERS]>(k_smem);

    // Initialize all of the register tiles.
    qkvo_tile<bf16> q_reg, k_reg;
    qkvo_tile<bf16, col_l> v_reg;
    qkvo_tile<float> o_reg; // v_reg need to be swapped into col_l
    attn_tile<float> att_block;
    attn_tile<bf16> att_block_mma;
    attn_tile<float>::col_vec max_vec_last, max_vec; // these are column vectors for the attention block
    attn_tile<float>::col_vec norm_vec; // these are column vectors for the attention block

    // each warp loads its own Q tile of 16x64, and then multiplies by 1/sqrt(d)
    // if (q_seq* R< N) load(q_reg, g.Qg, {batch, head, q_seq, 0});
    if (q_seq* R< N) {
        load(qo_smem[workerid], g.Qg, {batch, head, q_seq, 0});  // going through shared memory improves coalescing of dram reads.
        __syncwarp();
        load(q_reg, qo_smem[workerid]);
    }
    __syncthreads();

    // zero flash attention L, M, and O registers.
    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_reg);

    if constexpr(D == 64) mul(q_reg, q_reg, __float2bfloat16(0.125f * 1.44269504089)); // temperature adjustment
    else if constexpr(D == 128) mul(q_reg, q_reg, __float2bfloat16(0.08838834764f * 1.44269504089)); // temperature adjustment

    const int kv_blocks = N / (LOAD_BLOCKS*R);

    int tic = 0;

    // launch the load of the first k, v tiles
    load_group::load_async(k_smem[loadid][tic], g.Kg, {batch, head, loadid, 0});
    load_group::load_async(v_smem[loadid][tic], g.Vg, {batch, head, loadid, 0});

    // iterate over k, v for these q's that have been loaded
    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic^=1) {

        if(kv_idx+1 < kv_blocks) {
            int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;
            load_group::load_async(k_smem[loadid][tic^1], g.Kg, {batch, head, next_load_idx, 0});
            load_group::load_async(v_smem[loadid][tic^1], g.Vg, {batch, head, next_load_idx, 0});
            load_async_wait<2>(); // next k, v can stay in flight.
        }
        else {
            load_async_wait();
        }
        __syncthreads();

        // now each warp goes through all of the subtiles, loads them, and then does the flash attention internal alg.
        for(int subtile = 0; subtile < LOAD_BLOCKS; subtile++) {
            load(k_reg, k_smem[subtile][tic]); // load k from shared into registers
            zero(att_block); // zero 16x16 attention tile
            mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T
            copy(max_vec_last,  max_vec);
            row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
            sub_row(att_block, att_block, max_vec); // subtract max from attention -- now all <=0
            exp2(att_block, att_block); // exponentiate the block in-place.
            sub(max_vec_last, max_vec_last, max_vec); // subtract new max from old max to find the new normalization.
            exp2(max_vec_last, max_vec_last); // exponentiate this vector -- this is what we need to normalize by.
            mul(norm_vec, norm_vec, max_vec_last); // and the norm vec is now normalized.
            row_sum(norm_vec, att_block, norm_vec); // accumulate the new attention block onto the now-rescaled norm_vec
            copy(att_block_mma, att_block); // convert to bf16 for mma_AB
            load(v_reg, v_smem[subtile][tic]); // load v from shared into registers.
            mul_row(o_reg, o_reg, max_vec_last); // normalize o_reg in advance of mma_AB'ing onto it
            mma_AB(o_reg, att_block_mma, v_reg, o_reg); // mfma onto o_reg with the local attention@V matmul.
        }
        __syncthreads();
    }
    div_row(o_reg, o_reg, norm_vec);

    if (q_seq*R<N) { // write out o.
        store(qo_smem[workerid], o_reg); // going through shared memory improves coalescing of dram writes.
        __syncwarp();
        store(g.Og, qo_smem[workerid], {batch, head, q_seq, 0});
    }
}

#include "harness.impl"
