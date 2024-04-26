#include "../../../src/kittens.cuh"
#include <cooperative_groups.h>
using namespace kittens;

constexpr int NUM_WORKERS = 8;
constexpr int NUM_WARPGROUPS = (NUM_WORKERS/WARPGROUP_WARPS);

constexpr int qo_height = 4, kv_height = 8; // 4
constexpr int tile_width = 64/16;

using layout_q = ducks::st_layout::wgmma_swizzle; // need to make this 128b
using layout_k = ducks::st_layout::wgmma_swizzle; // need to make this 128b
using layout_v = ducks::st_layout::wgmma_interleave; // need to make this 128b
using layout_o = ducks::st_layout::swizzle;

template<int N> __global__  __launch_bounds__(NUM_WORKERS*kittens::WARP_THREADS, 2)
void attend_ker(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    st_bf<qo_height, tile_width, layout_q> (&q_smem)[NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_q>, NUM_WARPGROUPS>();
    st_bf<kv_height, tile_width, layout_k> (&k_smem)[2]              = al.allocate<st_bf<kv_height, tile_width, layout_k>, 2>();
    st_bf<kv_height, tile_width, layout_v> (&v_smem)[2]              = al.allocate<st_bf<kv_height, tile_width, layout_v>, 2>();

    int tic = 0, toc = 1;
 
    rt_fl<1, kv_height> att_block;
    rt_bf<1, kv_height> att_block_mma;
    rt_fl<1, tile_width> o_prev;
    rt_fl<1, kv_height>::col_vec max_vec_last, max_vec;
    rt_fl<1, kv_height>::col_vec norm_vec_last, norm_vec;

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS; 

    constexpr int kv_blocks = N / k_smem[0].rows;

    __shared__ uint64_t qsmem_barrier, kvsmem_barrier;//, vsmem_barrier;
    int kv_phasebit = 0;

    if (threadIdx.x == 0) {
        tma::init_barrier<st_bf<qo_height, tile_width, layout_q>, NUM_WARPGROUPS>(qsmem_barrier, 1);
        tma::init_barrier<st_bf<kv_height, tile_width, layout_k>, 2>(kvsmem_barrier, 1); 
    }
    __syncthreads();

    if (warpid == 0) {
        for (int wg = 0; wg < NUM_WORKERS/kittens::WARPGROUP_WARPS; wg++) { // load q
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + wg;
            tma::load_async((q_smem[wg]), tma_q, qsmem_barrier, tile_idx); 
        }  
        int tile_idx = (blockIdx.y * kv_blocks);
        tma::load_async((k_smem[tic]), tma_k, kvsmem_barrier, tile_idx);
        tma::load_async((v_smem[tic]), tma_v, kvsmem_barrier, tile_idx);
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);

    tma::arrive_and_wait(qsmem_barrier, 0);
    __syncthreads();

    warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f));

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic ^= 1, toc ^= 1) {

        tma::arrive_and_wait(kvsmem_barrier, kv_phasebit);
        kv_phasebit ^= 1;

        __syncthreads();
        if (warpid == 0) {
            tma::set_bytes(kvsmem_barrier, detail::transfer_bytes<typeof(k_smem[0]), 2>::bytes);

            if (kv_idx + 1 < kv_blocks) {    
                int tile_idx = (blockIdx.y * kv_blocks) + kv_idx + 1; 
                tma::load_async((k_smem[toc]), tma_k, kvsmem_barrier, tile_idx);
                tma::load_async((v_smem[toc]), tma_v, kvsmem_barrier, tile_idx);
            }
        }

        warpgroup::mma_fence(att_block);
        warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[tic]);
        warpgroup::mma_commit_group();

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

        warpgroup::mma_fence(o_prev);
        warpgroup::mma_AB(o_prev, att_block_mma, v_smem[tic]);
        warpgroup::mma_commit_group();
    }

    auto *o_smem = reinterpret_cast<st_bf<qo_height, tile_width, layout_o>*>(&q_smem[0].data[0]); // reuse q memory
    warpgroup::store(o_smem[warpgroupid], o_prev); 
    __syncthreads();
    if (warpid % 4 == 0) { // store o
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid; 
        tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    tma::store_async_wait();
}

#include "harness.impl"