#include "../../../src/kittens.cuh"
#include <cooperative_groups.h>
using namespace kittens;

constexpr int NUM_WORKERS = 8;
constexpr int NUM_WARPGROUPS = (NUM_WORKERS/WARPGROUP_WARPS);

using layout_q = ducks::st_layout::wgmma_swizzle; // need to make this 128b
using layout_k = ducks::st_layout::wgmma_swizzle; // need to make this 128b
using layout_v = ducks::st_layout::wgmma_interleave; // need to make this 128b
using layout_o = ducks::st_layout::swizzle;

template<int D> struct ker_tile_dims {
    static_assert(D==64 || D==128);
    constexpr static int tile_width = D/kittens::TILE_DIM;
    constexpr static int qo_height  = 4;
    constexpr static int kv_height  = 512/D;
};
template<int N, int D> __global__  __launch_bounds__(NUM_WORKERS*kittens::WARP_THREADS, 2)
void attend_ker(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o) {
    extern __shared__ int __shm[]; // dynamic shared memory
    tma_swizzle_allocator al((int*)&__shm[0]); // lightweight allocator enforces alignments.

    constexpr int tile_width = ker_tile_dims<D>::tile_width; // constants
    constexpr int qo_height  = ker_tile_dims<D>::qo_height;
    constexpr int kv_height  = ker_tile_dims<D>::kv_height;
    constexpr int kv_blocks  = N / (kv_height*TILE_DIM);

    st_bf<qo_height, tile_width, layout_q> (&q_smem)[NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_q>, NUM_WARPGROUPS>(); // shared tiles
    st_bf<kv_height, tile_width, layout_k> (&k_smem)[2]              = al.allocate<st_bf<kv_height, tile_width, layout_k>, 2>();
    st_bf<kv_height, tile_width, layout_v> (&v_smem)[2]              = al.allocate<st_bf<kv_height, tile_width, layout_v>, 2>();

    rt_fl<1, kv_height> att_block; // declare registers
    rt_bf<1, kv_height> att_block_mma;
    rt_fl<1, tile_width> o_prev;
    col_vec<rt_fl<1, kv_height>> max_vec_last, max_vec;
    col_vec<rt_fl<1, kv_height>> norm_vec_last, norm_vec;

    int warpid      = kittens::warpid(); // who am i? when am i?
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS; 
    int tic = 0, toc = 1;

    __shared__ uint64_t qsmem_barrier, kvsmem_barrier; // initialize barriers
    if (warpid == 0) {
        tma::init_barrier<typeof(q_smem[0]), NUM_WARPGROUPS>(qsmem_barrier, 1);
        tma::init_barrier<typeof(k_smem[0]), 2             >(kvsmem_barrier, 1); 
    }
    __syncthreads();

    if (warpid%4 == 0) { // load q from HBM
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid;
        tma::load_async(q_smem[warpgroupid], tma_q, qsmem_barrier, tile_idx);
    }
    if (warpid == 0) { // load initial k, v from HBM
        tma::load_async(k_smem[tic], tma_k, kvsmem_barrier, blockIdx.y*kv_blocks);
        tma::load_async(v_smem[tic], tma_v, kvsmem_barrier, blockIdx.y*kv_blocks);
    }

    neg_infty(max_vec); // zero registers, while we wait
    zero(norm_vec);
    zero(o_prev);

    tma::arrive_and_wait(qsmem_barrier, 0); // wait for memory to arrive
    if constexpr (D==64)  warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f)); // temperature adjustment
    if constexpr (D==128) warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.08838834764f)); // temperature adjustment

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic ^= 1, toc ^= 1) {

        tma::arrive_and_wait(kvsmem_barrier, tic); // wait for kv memory to arrive
        __syncthreads(); // everybody on the same page?
        if (warpid == 0) { // go get the K, V from HBM
            tma::set_bytes(kvsmem_barrier, detail::transfer_bytes<typeof(k_smem[0]), 2>::bytes);
            if (kv_idx + 1 < kv_blocks) {    
                int tile_idx = (blockIdx.y * kv_blocks) + kv_idx + 1; 
                tma::load_async((k_smem[toc]), tma_k, kvsmem_barrier, tile_idx);
                tma::load_async((v_smem[toc]), tma_v, kvsmem_barrier, tile_idx);
            }
        }

        warpgroup::mma_fence(att_block); // qk matmul fence
        warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[tic]); // clear registers -- note mm_ABt, not mma_ABt.
        warpgroup::mma_commit_group(); // dew it

        copy(norm_vec_last, norm_vec); // copy registers, while we wait
        copy(max_vec_last,  max_vec);

        warpgroup::mma_async_wait(); // ding dong! matmuls arrived.

        row_max(max_vec, att_block, max_vec); // accumulate new max onto the max_vec
        sub_row(att_block, att_block, max_vec); // ensure all <=0 for numerics
        exp(att_block, att_block); // exponentiate attention block for softmax in FP32
        sub(max_vec_last, max_vec_last, max_vec); // how do we need to normalize previous O's due to new max, in log?
        exp(max_vec_last, max_vec_last); // how do we need to norm previous O's due to new max, actually?
        mul(norm_vec, norm_vec, max_vec_last); // norm previous exp sum using new max
        row_sum(norm_vec, att_block, norm_vec); // accumulate new exp sum onto the norm_vec
        div_row(att_block, att_block, norm_vec); // softmax normalization of existing attention block
        mul(norm_vec_last, norm_vec_last, max_vec_last); // incorporate previous max into norm for o
        div(norm_vec_last, norm_vec_last, norm_vec); // incorporate current norm into new norm for o
        copy(att_block_mma, att_block); // convert to bf16 for mma
        mul_row(o_prev, o_prev, norm_vec_last); // normalize o in advance of mma'ing onto it

        warpgroup::mma_fence(o_prev);  // av matmul fence
        warpgroup::mma_AB(o_prev, att_block_mma, v_smem[tic]); // mm accumulate next attention chunk onto o
        warpgroup::mma_commit_group(); // dew it.
    }

    auto *o_smem = reinterpret_cast<st_bf<qo_height, tile_width, layout_o>*>(&q_smem[0].data[0]); // reuse q memory for store
    warpgroup::store(o_smem[warpgroupid], o_prev); // store from registers to shared mem
    __syncthreads(); // everyone done?
    if (warpid%4 == 0) { // store o to HBM
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid; 
        tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); // dew it
    }
    tma::store_async_wait(); // done it.
}

#include "harness.impl"