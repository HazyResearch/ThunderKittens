#define KITTENS_HOPPER // we are on an H100
#include "../../src/kittens.cuh"
#include <cooperative_groups.h>

constexpr int NUM_WORKERS = 16;
constexpr int NUM_WARPGROUPS = (NUM_WORKERS/(kittens::WARPGROUP_WARPS));

constexpr int qo_height = 4, kv_height = 4;
constexpr int NUM_WORKERS_KV = 4;
constexpr int tile_width = 64/16;

using namespace kittens;

using layout_q = ducks::st_layout::wgmma_row_0b;
using layout_k = ducks::st_layout::wgmma_row_0b;
using layout_v = ducks::st_layout::wgmma_col_t_0b;
using layout_o = ducks::st_layout::tma_swizzle; 

template<int N> __global__  __launch_bounds__(NUM_WORKERS*kittens::WARP_THREADS, 1)
void attend_ker_fwd_train(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o, CUtensorMap* tma_l) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    st_bf<qo_height, tile_width, layout_q>           (&q_smem)   [NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_q>,          NUM_WARPGROUPS>();
    st_bf<kv_height, tile_width, layout_k>           (&k_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, tile_width, layout_k>, 2,       NUM_WORKERS_KV>();
    st_bf<kv_height, tile_width, layout_v>           (&v_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, tile_width, layout_v>, 2,       NUM_WORKERS_KV>();
    st_bf<qo_height, tile_width, layout_q>::col_vec  (&l_smem)   [NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_q>::col_vec, NUM_WARPGROUPS>();

    int tic = 0, toc = 1;
 
    rt_fl<1, kv_height> att_block;
    rt_bf<1, kv_height> att_block_mma;
    rt_fl<1, tile_width> o_prev;
    rt_fl<1, kv_height>::col_vec max_vec_last, max_vec;
    rt_fl<1, kv_height>::col_vec norm_vec_last, norm_vec;

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS; 

    auto block = cooperative_groups::this_thread_block();

    constexpr int qo_tiles  = N / q_smem[0].rows; 
    constexpr int kv_blocks = N / (NUM_WORKERS_KV*k_smem[0][0].rows);

    __shared__ uint64_t qsmem_barrier, ksmem_barrier, vsmem_barrier;

    constexpr int kPhaseBit = 1;

    if (warpid == 0) {
        tma::init_barrier<st_bf<qo_height, tile_width, layout_q>, NUM_WARPGROUPS>(qsmem_barrier, block.size());
        tma::init_barrier<st_bf<kv_height, tile_width, layout_k>, NUM_WORKERS_KV>(ksmem_barrier, block.size()); 
        tma::init_barrier<st_bf<kv_height, tile_width, layout_v>, NUM_WORKERS_KV>(vsmem_barrier, block.size());
    }
    __syncthreads();

    if (warpid == 0) {
        for (int wg = 0; wg < NUM_WORKERS/kittens::WARPGROUP_WARPS; wg++) { // load q
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + wg;
            tma::load_async((q_smem[wg]), tma_q, qsmem_barrier, tile_idx); 
        }
        for (int w = 0; w < NUM_WORKERS_KV; w++) { // load k, v      
            int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + (0 * NUM_WORKERS_KV) + w; 
            tma::load_async((k_smem[tic][w]), tma_k, ksmem_barrier, tile_idx); 
            tma::load_async((v_smem[tic][w]), tma_v, vsmem_barrier, tile_idx); 
        }
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);

    tma::arrive_and_wait(qsmem_barrier, kPhaseBit); 
    __syncthreads();
    warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f));

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic ^= 1, toc ^= 1) {

        tma::arrive_and_wait(ksmem_barrier, kPhaseBit);
        tma::arrive_and_wait(vsmem_barrier, kPhaseBit);

        if ((threadIdx.x == 0)) {
            tma::init_barrier<st_bf<kv_height, tile_width, layout_k>, NUM_WORKERS_KV>(ksmem_barrier, block.size());
            tma::init_barrier<st_bf<kv_height, tile_width, layout_v>, NUM_WORKERS_KV>(vsmem_barrier, block.size());
        }
        __syncthreads();

        if ((kv_idx + 1 < kv_blocks) && (warpid == 0)) {
            for (int w = 0; w < NUM_WORKERS_KV; w++) {        
                int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS_KV) + w; 
                tma::load_async((k_smem[toc][w]), tma_k, ksmem_barrier, tile_idx); 
                tma::load_async((v_smem[toc][w]), tma_v, vsmem_barrier, tile_idx); 
            }
        }

        for(int subtile = 0; subtile < NUM_WORKERS_KV; subtile++) {
            warpgroup::mma_fence(att_block);
            warpgroup::dot_reset(att_block, q_smem[warpgroupid], k_smem[tic][subtile]);
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
            warpgroup::mma_accum(o_prev, att_block_mma, v_smem[tic][subtile]);
            warpgroup::mma_commit_group();
        }
    }

    auto *o_smem = reinterpret_cast<st_bf<qo_height, tile_width, layout_o>*>(&q_smem[0].data[0]); // reuse q memory
    warpgroup::store(o_smem[warpgroupid], o_prev); 
    __syncthreads();
    if (warpid % 4 == 0) { // store o
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid; 
        tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    __syncthreads();

    warpgroup::store(l_smem[warpgroupid], norm_vec);
    __syncthreads();
    if (warpid % 4 == 0) { // store l
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid; 
        tma::store_async(tma_l, (l_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    tma::store_async_wait();
}

constexpr int NUM_WORKERS_BWD = 16;
constexpr int NUM_WARPGROUPS_BWD = (NUM_WORKERS_BWD/(kittens::WARPGROUP_WARPS));

constexpr int qo_height_bwd = 4, kv_height_bwd = 4;
constexpr int NUM_WORKERS_QO_BWD = 4;
constexpr int tile_width_bwd = 64/16;

template<int N> __global__  __launch_bounds__(NUM_WORKERS_BWD*kittens::WARP_THREADS, 1)
void attend_ker_bwd(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o, CUtensorMap* tma_l, 
                            CUtensorMap* tma_o_grad, CUtensorMap* tma_q_grad, CUtensorMap* tma_k_grad, CUtensorMap* tma_v_grad) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    st_bf<qo_height_bwd, tile_width_bwd, layout_q>           (&q_smem)   [NUM_WARPGROUPS_BWD] = al.allocate<st_bf<qo_height_bwd, tile_width_bwd, layout_q>,          NUM_WARPGROUPS_BWD>();
    st_bf<kv_height_bwd, tile_width_bwd, layout_k>           (&k_smem)[2][NUM_WORKERS_QO_BWD] = al.allocate<st_bf<kv_height_bwd, tile_width_bwd, layout_k>, 2,       NUM_WORKERS_QO_BWD>();
    st_bf<kv_height_bwd, tile_width_bwd, layout_v>           (&v_smem)[2][NUM_WORKERS_QO_BWD] = al.allocate<st_bf<kv_height_bwd, tile_width_bwd, layout_v>, 2,       NUM_WORKERS_QO_BWD>();
    st_bf<qo_height_bwd, tile_width_bwd, layout_q>::col_vec  (&l_smem)   [NUM_WARPGROUPS_BWD] = al.allocate<st_bf<qo_height_bwd, tile_width_bwd, layout_q>::col_vec, NUM_WARPGROUPS_BWD>();

    int tic = 0, toc = 1;
 
    rt_fl<1, kv_height_bwd> att_block;
    rt_bf<1, kv_height_bwd> att_block_mma;
    rt_fl<1, tile_width_bwd> o_prev;
    rt_fl<1, kv_height_bwd>::col_vec max_vec_last, max_vec;
    rt_fl<1, kv_height_bwd>::col_vec norm_vec_last, norm_vec;

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS; 

    auto block = cooperative_groups::this_thread_block();

    constexpr int qo_tiles  = N / q_smem[0].rows; 
    constexpr int kv_blocks = N / (NUM_WORKERS_QO_BWD*k_smem[0][0].rows);

    __shared__ uint64_t qsmem_barrier, ksmem_barrier, vsmem_barrier;
    constexpr int kPhaseBit = 1;

    if (warpid == 0) {
        tma::init_barrier<st_bf<qo_height_bwd, tile_width_bwd, layout_q>, NUM_WARPGROUPS_BWD>(qsmem_barrier, block.size());
        tma::init_barrier<st_bf<kv_height_bwd, tile_width_bwd, layout_k>, NUM_WORKERS_QO_BWD>(ksmem_barrier, block.size()); 
        tma::init_barrier<st_bf<kv_height_bwd, tile_width_bwd, layout_v>, NUM_WORKERS_QO_BWD>(vsmem_barrier, block.size());
    }
    __syncthreads();

    if (warpid == 0) {
        for (int wg = 0; wg < NUM_WORKERS_BWD/kittens::WARPGROUP_WARPS; wg++) { // load q
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS_BWD) + wg;
            tma::load_async((q_smem[wg]), tma_q, qsmem_barrier, tile_idx); 
        }
        for (int w = 0; w < NUM_WORKERS_QO_BWD; w++) { // load k, v      
            int tile_idx = (blockIdx.y * NUM_WORKERS_QO_BWD * kv_blocks) + (0 * NUM_WORKERS_QO_BWD) + w; 
            tma::load_async((k_smem[tic][w]), tma_k, ksmem_barrier, tile_idx); 
            tma::load_async((v_smem[tic][w]), tma_v, vsmem_barrier, tile_idx); 
        }
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);

    tma::arrive_and_wait(qsmem_barrier, kPhaseBit); 
    __syncthreads();
    warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f));

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic ^= 1, toc ^= 1) {

        tma::arrive_and_wait(ksmem_barrier, kPhaseBit);
        tma::arrive_and_wait(vsmem_barrier, kPhaseBit);

        if ((threadIdx.x == 0)) {
            tma::init_barrier<st_bf<kv_height_bwd, tile_width_bwd, layout_k>, NUM_WORKERS_QO_BWD>(ksmem_barrier, block.size());
            tma::init_barrier<st_bf<kv_height_bwd, tile_width_bwd, layout_v>, NUM_WORKERS_QO_BWD>(vsmem_barrier, block.size());
        }
        __syncthreads();

        if ((kv_idx + 1 < kv_blocks) && (warpid == 0)) {
            for (int w = 0; w < NUM_WORKERS_QO_BWD; w++) {        
                int tile_idx = (blockIdx.y * NUM_WORKERS_QO_BWD * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS_QO_BWD) + w; 
                tma::load_async((k_smem[toc][w]), tma_k, ksmem_barrier, tile_idx); 
                tma::load_async((v_smem[toc][w]), tma_v, vsmem_barrier, tile_idx); 
            }
        }

        for(int subtile = 0; subtile < NUM_WORKERS_QO_BWD; subtile++) {
            warpgroup::mma_fence(att_block);
            warpgroup::dot_reset(att_block, q_smem[warpgroupid], k_smem[tic][subtile]);
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
            warpgroup::mma_accum(o_prev, att_block_mma, v_smem[tic][subtile]);
            warpgroup::mma_commit_group();
        }
    }

    auto *o_smem = reinterpret_cast<st_bf<qo_height_bwd, tile_width_bwd, layout_o>*>(&q_smem[0].data[0]); // reuse q memory
    warpgroup::store(o_smem[warpgroupid], o_prev); 
    __syncthreads();
    if (warpid % 4 == 0) { // store o
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS_BWD) + warpgroupid; 
        tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);

    warpgroup::store(l_smem[warpgroupid], norm_vec);
    __syncthreads();
    if (warpid % 4 == 0) { // store l
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS_BWD) + warpgroupid; 
        tma::store_async(tma_l, (l_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    tma::store_async_wait();
}

#include "harness_t.impl"