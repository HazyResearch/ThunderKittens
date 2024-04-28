#include "../../../src/kittens.cuh"
#include <cooperative_groups.h>
using namespace kittens;

constexpr int NUM_WORKERS = 8, NUM_WARPGROUPS = (NUM_WORKERS/WARPGROUP_WARPS);
using layout_q = wgmma_swizzle_l;
using layout_k = wgmma_swizzle_l;
using layout_v = wgmma_interleave_l; // must support imm-trans-b
using layout_o = swizzle_l; // fastest write out
template<int D> struct fwd_attend_ker_tile_dims {
    constexpr static int tile_width = D/kittens::TILE_DIM;
    constexpr static int qo_height  = 4;
    constexpr static int kv_height  = 512/D;
};

// the two cases (D=64, D=128) basically identical, but the barriers have been slightly tuned to eke out that extra few percent.
// current benchmarks (N=4096) are 460 TFLOPs at D=64 and 517 TFLOPs at D=128.
template<int D, int N> __global__  __launch_bounds__(NUM_WORKERS*kittens::WARP_THREADS, 2)
void fwd_attend_ker(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o) {
    extern __shared__ int __shm[]; // dynamic shared memory
    tma_swizzle_allocator al((int*)&__shm[0]); // lightweight allocator enforces alignments.

    constexpr int tile_width = fwd_attend_ker_tile_dims<D>::tile_width; // constants
    constexpr int qo_height  = fwd_attend_ker_tile_dims<D>::qo_height;
    constexpr int kv_height  = fwd_attend_ker_tile_dims<D>::kv_height;
    constexpr int kv_blocks  = N / (kv_height*TILE_DIM);

    auto (&q_smem)[NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_q>, NUM_WARPGROUPS>(); // shared tiles
    auto (&k_smem)[2]              = al.allocate<st_bf<kv_height, tile_width, layout_k>, 2>();
    auto (&v_smem)[2]              = al.allocate<st_bf<kv_height, tile_width, layout_v>, 2>();

    rt_fl<1, kv_height> att_block; // declare registers
    rt_bf<1, kv_height> att_block_mma;
    rt_fl<1, tile_width> o_accum;
    col_vec<rt_fl<1, kv_height>> max_vec_last, max_vec;
    col_vec<rt_fl<1, kv_height>> norm_vec_last, norm_vec;

    int warpid      = kittens::warpid(); // who am i? when am i?
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS; 
    int tic = 0, toc = 1, v_phase = 0; // since we have two barriers for v, we need a half-rate tic as the v phase bit

    __shared__ uint64_t qsmem_barrier, kv_smem_barriers[3]; // init barriers
    // D=64 and D=128 meaningfully (2% perf) prefer different barrier, so this is my hack to prevent splitting into separate kernels.
    uint64_t *kbar, *vbar[2];
    if constexpr (D==64)  { kbar = &kv_smem_barriers[0]; vbar[0] = &kv_smem_barriers[0]; vbar[1] = &kv_smem_barriers[0]; } // set all as aliases
    if constexpr (D==128) { kbar = &kv_smem_barriers[0]; vbar[0] = &kv_smem_barriers[1]; vbar[1] = &kv_smem_barriers[2]; } // separate barriers
    if      (warpid == 0) tma::init_barrier<typeof(q_smem[0]), NUM_WARPGROUPS>(qsmem_barrier);
    else if (warpid == 1) tma::init_barrier<typeof(k_smem[0]), 128/D         >(*kbar);
    if constexpr(D==128) {
        if      (warpid == 2) tma::init_barrier<typeof(v_smem[0])>(*vbar[tic]);
        else if (warpid == 3) tma::init_barrier                   (*vbar[toc]); // will set bytes later anyways.
    }
    __syncthreads();

    if (warpid%4 == 0) { // load q from HBM
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid;
        tma::load_async(q_smem[warpgroupid], tma_q, qsmem_barrier, tile_idx);
    }
    if      (warpid == 0) tma::load_async(k_smem[tic], tma_k,* kbar     , blockIdx.y*kv_blocks); // load initial k, v from HBM
    else if (warpid == 1) tma::load_async(v_smem[tic], tma_v, *vbar[tic], blockIdx.y*kv_blocks);

    neg_infty(max_vec); // zero registers, while we wait
    zero(norm_vec);
    zero(o_accum);

    tma::arrive_and_wait(qsmem_barrier, 0); // wait for memory to arrive
    if constexpr (D==64)  warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f)); // temperature adjustment
    if constexpr (D==128) warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.08838834764f)); // temperature adjustment

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic ^= 1, toc ^= 1) {

        tma::arrive_and_wait(*kbar, tic); // wait for k memory to arrive (and also v, too, if D=64)
        __syncthreads(); // everybody on the same page?
        if (warpid == 0) { // go get the next K from HBM
            tma::set_bytes(*kbar, size_bytes<typeof(k_smem[0]), 128/D>); // depending on how many real barriers, adjust
            if constexpr (D==128) tma::set_bytes(*vbar[toc], size_bytes<typeof(v_smem[0])>);
            if (kv_idx+1 < kv_blocks) {
                tma::load_async(k_smem[toc], tma_k, *kbar     , (blockIdx.y * kv_blocks) + kv_idx + 1);
                tma::load_async(v_smem[toc], tma_v, *vbar[toc], (blockIdx.y * kv_blocks) + kv_idx + 1);
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
        mul_row(o_accum, o_accum, norm_vec_last); // normalize o in advance of mma'ing onto it

        if constexpr (D==128) tma::arrive_and_wait(*vbar[tic], v_phase); // wait for v memory to arrive, if this is a real barrier

        warpgroup::mma_fence(o_accum);  // av matmul fence
        warpgroup::mma_AB(o_accum, att_block_mma, v_smem[tic]); // mm accumulate next attention chunk onto o
        warpgroup::mma_commit_group(); // dew it.

        if(tic) v_phase^=1;
    }

    auto *o_smem = reinterpret_cast<st_bf<qo_height, tile_width, layout_o>*>(&q_smem[0].data[0]); // reuse q memory for store
    warpgroup::store(o_smem[warpgroupid], o_accum); // store from registers to shared mem
    __syncthreads(); // everyone done?
    if (warpid%4 == 0) { // store o to HBM
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid; 
        tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); // dew it
    }
    tma::store_async_wait(); // done it.
}

#include "harness.impl"