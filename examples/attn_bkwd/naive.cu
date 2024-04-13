#define KITTENS_HOPPER // we are on an H100
#include "../../src/kittens.cuh"

constexpr int NUM_WORKERS = 16;
constexpr int NUM_WARPGROUPS = (NUM_WORKERS/(kittens::WARPGROUP_WARPS));

constexpr int qo_height = 4, kv_height = 4;
constexpr int NUM_WORKERS_KV = 4;
constexpr int tile_width = 64/16;

using namespace kittens;

using layout_q = ducks::st_layout::wgmma_row_0b;
using layout_k = ducks::st_layout::wgmma_row_0b;
using layout_v = ducks::st_layout::wgmma_row_0b;
using layout_o = ducks::st_layout::wgmma_row_0b;

using layout_col = ducks::st_layout::wgmma_col_t_0b; 

template<int N> __global__  __launch_bounds__(NUM_WORKERS*kittens::WARP_THREADS, 1)
void attend_ker_bwd(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_og, CUtensorMap* tma_q_grad, CUtensorMap* tma_k_grad, CUtensorMap* tma_v_grad) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_allocator al((int*)&__shm[0]);

    st_bf<qo_height, tile_width, layout_q> (&q_smem)   [NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_q>, NUM_WARPGROUPS>();
    st_bf<kv_height, tile_width, layout_k> (&k_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, tile_width, layout_k>, 2, NUM_WORKERS_KV>();
    st_bf<kv_height, tile_width, layout_v> (&v_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, tile_width, layout_v>, 2, NUM_WORKERS_KV>();
    st_bf<qo_height, tile_width, layout_o> (&og_smem)   [NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_o>, NUM_WARPGROUPS>();

    int tic = 0, toc = 1;
 
    rt_fl<1, kv_height> att_block;
    rt_bf<1, kv_height> att_block_mma;

    rt_fl<1, kv_height> dO_v_block;
    rt_fl<1, kv_height> imd_block;

    rt_fl<1, kv_height>::col_vec max_vec_last, max_vec;
    rt_fl<1, kv_height>::col_vec norm_vec_last, norm_vec;

    rt_fl<1, kv_height>::col_vec sm_vec; 

    rt_fl<1, tile_width> q_grad, k_grad, v_grad; 

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS; 

    constexpr int qo_tiles  = N / q_smem[0].rows; 
    constexpr int kv_blocks = N / (NUM_WORKERS_KV*k_smem[0][0].rows);

    __shared__ uint64_t qsmem_barrier, ksmem_barrier, vsmem_barrier, ogsmem_barrier;

    constexpr int tile_bytes = sizeof(bf16) * k_smem[0][0].num_elements * NUM_WORKERS_KV, kPhaseBit = 1;

    if (warpid == 0) {
        tma::init_barrier(qsmem_barrier, NUM_WORKERS*WARP_THREADS); 
        tma::set_barrier_bytes(qsmem_barrier, tile_bytes); 

        tma::init_barrier(ksmem_barrier, NUM_WORKERS*WARP_THREADS);
        tma::set_barrier_bytes(ksmem_barrier, tile_bytes);

        tma::init_barrier(vsmem_barrier, NUM_WORKERS*WARP_THREADS);
        tma::set_barrier_bytes(vsmem_barrier, tile_bytes);

        tma::init_barrier(ogsmem_barrier, NUM_WORKERS*WARP_THREADS);
        tma::set_barrier_bytes(ogsmem_barrier, tile_bytes);
    }
    __syncthreads();

    if (warpid == 0) {
        for (int wg = 0; wg < NUM_WORKERS/kittens::WARPGROUP_WARPS; wg++) { // load q
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + wg;
            tma::load_async((q_smem[wg]), tma_q, tile_idx, qsmem_barrier); 
            tma::load_async((og_smem[wg]), tma_og, tile_idx, ogsmem_barrier);
        }
        for (int w = 0; w < NUM_WORKERS_KV; w++) { // load k, v      
            int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + (0 * NUM_WORKERS_KV) + w; 
            tma::load_async((k_smem[tic][w]), tma_k, tile_idx, ksmem_barrier); 
            tma::load_async((v_smem[tic][w]), tma_v, tile_idx, vsmem_barrier); 
        }
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);

    tma::arrive_and_wait(qsmem_barrier, kPhaseBit); 
    __syncthreads();
    warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f));
    tma::arrive_and_wait(ogsmem_barrier, kPhaseBit);

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic ^= 1, toc ^= 1) {

        tma::arrive_and_wait(ksmem_barrier, kPhaseBit); 
        tma::arrive_and_wait(vsmem_barrier, kPhaseBit); 

        if ((threadIdx.x == 0)) {
            tma::init_barrier(ksmem_barrier, NUM_WORKERS*WARP_THREADS); 
            tma::set_barrier_bytes(ksmem_barrier, tile_bytes); 

            tma::init_barrier(vsmem_barrier, NUM_WORKERS*WARP_THREADS); 
            tma::set_barrier_bytes(vsmem_barrier, tile_bytes); 
        }
        __syncthreads();

        if ((kv_idx + 1 < kv_blocks) && (warpid == 0)) {
            for (int w = 0; w < NUM_WORKERS_KV; w++) {        
                int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS_KV) + w; 
                tma::load_async((k_smem[toc][w]), tma_k, tile_idx, ksmem_barrier); 
                tma::load_async((v_smem[toc][w]), tma_v, tile_idx, vsmem_barrier); 
            }
        }

        for(int subtile = 0; subtile < NUM_WORKERS_KV; subtile++) {
            warpgroup::fence(att_block);
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
            warpgroup::fence(v_grad);
            // need to tranpose on store
            warpgroup::dot_accum(v_grad, att_block_mma, og_smem[warpgroupid]);
            warpgroup::mma_commit_group();

            warpgroup::fence(dO_v_block);
            warpgroup::dot_accum(dO_v_block, og_smem[warpgroupid], v_smem[tic][subtile]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            mul(imd_block, dO_v_block, att_block);
            zero(sm_vec);
            row_sum(sm_vec, imd_block, sm_vec);
            sub_row(dO_v_block, dO_v_block, sm_vec);
            mul(dO_v_block, dO_v_block, att_block);
            copy(att_block_mma, dO_v_block); 

            warpgroup::fence(k_grad);
            // need to tranpose on store
            warpgroup::dot_accum(k_grad, att_block_mma, q_smem[warpgroupid]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            rt_bf<1, kv_height, ducks::rt_layout::col> &att_block_mma_col = swap_layout_inplace(att_block_mma);
            auto *att_block_mma_row_cast = reinterpret_cast<rt_bf<1, kv_height>*>(&att_block_mma_col.tiles[0][0].data[0]);

            warpgroup::fence(q_grad);
            // need to tranpose on store
            warpgroup::dot_accum(q_grad, *att_block_mma_row_cast, k_smem[tic][subtile]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
        }

        auto *temp_k_smem_tic = reinterpret_cast<st_bf<kv_height, tile_width, layout_col>*>(&k_smem[tic][0].data[0]);
        auto *temp_v_smem_tic = reinterpret_cast<st_bf<kv_height, tile_width, layout_col>*>(&v_smem[tic][0].data[0]);

        warpgroup::store(temp_k_smem_tic[warpgroupid], k_grad);
        warpgroup::store(temp_v_smem_tic[warpgroupid], v_grad);
        __syncthreads();
        if (warpid == 0) { // store k, v
            for (int w = 0; w < NUM_WORKERS_KV; w++) {        
                int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + ((kv_idx) * NUM_WORKERS_KV) + w; 
                tma::store_async(tma_k_grad, temp_k_smem_tic[w], tile_idx);
                tma::store_async(tma_v_grad, temp_v_smem_tic[w], tile_idx);
            }
            tma::store_commit_group();
        }
    }

    auto *temp_q_smem = reinterpret_cast<st_bf<qo_height, tile_width, layout_col>*>(&q_smem[0].data[0]);
    warpgroup::store(temp_q_smem[warpgroupid], q_grad);
    __syncthreads();
    if (warpid == 0) { // store q
        for (int wg = 0; wg < NUM_WORKERS/kittens::WARPGROUP_WARPS; wg++) { // load q
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + wg;
            tma::store_async(tma_q_grad, temp_q_smem[wg], tile_idx);
        }
        tma::store_commit_group();
    }
    tma::store_async_wait();

    // auto *o_smem = reinterpret_cast<st_bf<qo_height, tile_width, layout_o>*>(&q_smem[0].data[0]); // reuse q memory
    // warpgroup::store(o_smem[warpgroupid], o_prev); 
    // __syncthreads();
    // if (warpid % 4 == 0) { // store o
    //     int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid; 
    //     tma::store_async(tma_q_grad, (o_smem[warpgroupid]), tile_idx); 
    //     tma::store_commit_group(); 
    // }
    // tma::store_async_wait();
}

#include "harness.impl"