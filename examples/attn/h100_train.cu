
// #include "src/kittens.cuh"
#include "../../src/kittens.cuh" // for harness_h100_fwd.impl
#include <cuda/pipeline>
#include <cooperative_groups.h>

#define ATTN_B 16
#define ATTN_H 16
#define ATTN_N 4096
#define ATTN_D 64

#define NUM_WORKERS 8
#define NUM_WARPGROUPS (NUM_WORKERS/(kittens::WARPGROUP_WARPS))

#define qo_height 4
#define kv_height 8
#define NUM_WORKERS_KV 1
#define tile_width 64/16

using namespace kittens;

using layout_q = ducks::st_layout::wgmma_swizzle; // need to make this 128b
using layout_k = ducks::st_layout::wgmma_swizzle; // need to make this 128b
using layout_v = ducks::st_layout::wgmma_interleave; // need to make this 128b
using layout_o = ducks::st_layout::swizzle;

__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 2)
void attend_ker_fwd_train(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o, CUtensorMap* tma_l) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    st_bf<qo_height, tile_width, layout_q>          (&q_smem)   [NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_q>,          NUM_WARPGROUPS>();
    st_bf<kv_height, tile_width, layout_k>          (&k_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, tile_width, layout_k>, 2,       NUM_WORKERS_KV>();
    st_bf<kv_height, tile_width, layout_v>          (&v_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, tile_width, layout_v>, 2,       NUM_WORKERS_KV>();
    st_bf<qo_height, tile_width, layout_o>::col_vec (&l_smem)   [NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_o>::col_vec, NUM_WARPGROUPS>();

    int tic = 0, toc = 1;
 
    rt_fl<1, kv_height> att_block;
    rt_bf<1, kv_height> att_block_mma;
    rt_fl<1, tile_width> o_prev;
    rt_fl<1, kv_height>::col_vec max_vec_last, max_vec;
    rt_fl<1, kv_height>::col_vec norm_vec_last, norm_vec;

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    constexpr int kv_blocks = ATTN_N / (NUM_WORKERS_KV*k_smem[0][0].rows);

    __shared__ uint64_t qsmem_barrier, kvsmem_barrier;//, vsmem_barrier;

    int q_phasebit = 0;
    int kv_phasebit = 0;

    if (threadIdx.x == 0) {
        tma::init_barrier<st_bf<qo_height, tile_width, layout_q>, NUM_WARPGROUPS>(qsmem_barrier, 1);
        tma::init_barrier<st_bf<kv_height, tile_width, layout_k>, NUM_WORKERS_KV*2>(kvsmem_barrier, 1); 
    }

    if (warpid == 0) {
        for (int wg = 0; wg < NUM_WORKERS/kittens::WARPGROUP_WARPS; wg++) { // load q
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + wg;
            tma::load_async((q_smem[wg]), tma_q, qsmem_barrier, tile_idx); 
        }
        for (int w = 0; w < NUM_WORKERS_KV; w++) { // load k, v      
            int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + (0 * NUM_WORKERS_KV) + w; 
            tma::load_async((k_smem[tic][w]), tma_k, kvsmem_barrier, tile_idx); 
            tma::load_async((v_smem[tic][w]), tma_v, kvsmem_barrier, tile_idx); 
        }
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);
    __syncthreads();

    tma::arrive_and_wait(qsmem_barrier, q_phasebit);
    q_phasebit ^= 1;

    warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f));

    for (auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic ^= 1, toc ^= 1) {
        tma::arrive_and_wait(kvsmem_barrier, kv_phasebit);
        kv_phasebit ^= 1;

        __syncthreads();
        if (warpid == 0) {
            tma::set_bytes(kvsmem_barrier, 2 * NUM_WORKERS_KV * k_smem[0][0].num_elements * sizeof(bf16));

            if (kv_idx + 1 < kv_blocks) {
                for (int w = 0; w < NUM_WORKERS_KV; w++) {        
                    int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS_KV) + w; 
                    tma::load_async((k_smem[toc][w]), tma_k, kvsmem_barrier, tile_idx); 
                    tma::load_async((v_smem[toc][w]), tma_v, kvsmem_barrier, tile_idx);
                }
            }
        }

        warpgroup::mma_fence(att_block);
        warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[tic][0]);
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
        warpgroup::mma_AB(o_prev, att_block_mma, v_smem[tic][0]);
        warpgroup::mma_commit_group();
    }

    auto (*o_smem) = reinterpret_cast<st_bf<qo_height, tile_width, layout_o>(*)>(q_smem); // reuse q memory
    warpgroup::store(o_smem[warpgroupid], o_prev); 
    __syncthreads();
    
    if (warpid % 4 == 0) { // store o
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid;
        tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    __syncthreads();

    warpgroup::store(l_smem[warpgroupid], norm_vec);
    __syncthreads();
    if (warpid % 4 == 0) {
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid; 
        tma::store_async(tma_l, (l_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    tma::store_async_wait();
}

#define WORKERS 4
using layout_nrow = ducks::st_layout::swizzle;

__global__  __launch_bounds__(WORKERS*kittens::WARP_THREADS, 2)
void attend_ker_prep_train(CUtensorMap* tma_o, CUtensorMap* tma_d, CUtensorMap* tma_o_grad) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();

    st_bf<4, 4, layout_nrow>          (&og_smem)[WORKERS] = al.allocate<st_bf<4, 4, layout_nrow>, WORKERS>();
    st_bf<4, 4, layout_nrow>          (&o_smem) [WORKERS] = al.allocate<st_bf<4, 4, layout_nrow>, WORKERS>();
    st_bf<4, 4, layout_nrow>::col_vec (&d_smem) [WORKERS] = al.allocate<st_bf<4, 4, layout_nrow>::col_vec, WORKERS>();

    rt_fl<4, 4> og_reg;
    rt_fl<4, 4> o_reg; 
    rt_fl<4, 4>::col_vec d_reg;

    __shared__ uint64_t smem_barrier;
    int o_phasebit = 0; 

    if (threadIdx.x == 0) {
        tma::init_barrier<st_bf<4, 4, layout_o>, WORKERS * 2>(smem_barrier, 1);
    }

    if (warpid == 0) {
        for (int w = 0; w < WORKERS; w++) { // load o, o_grad
            int tile_idx = (blockIdx.y * WORKERS * gridDim.x) + (blockIdx.x * WORKERS) + w; 
            tma::load_async((o_smem[w]),  tma_o,      smem_barrier, tile_idx); 
            tma::load_async((og_smem[w]), tma_o_grad, smem_barrier, tile_idx); 
        }
    }
    __syncthreads();

    tma::arrive_and_wait(smem_barrier, o_phasebit);
    o_phasebit ^= 1;

    load(o_reg, o_smem[warpid]);
    load(og_reg, og_smem[warpid]);

    mul(og_reg, og_reg, o_reg);
    row_sum(d_reg, og_reg);
    
    store(d_smem[warpid], d_reg);
    __syncthreads(); 

    if (warpid == 0) {
        for (int w = 0; w < WORKERS; w++) {
            int tile_idx = (blockIdx.y * WORKERS * gridDim.x) + (blockIdx.x * WORKERS) + w; 
            tma::store_async(tma_d, (d_smem[w]), tile_idx); 
        }
        tma::store_commit_group();
    }

    tma::store_async_wait();
}

#define WORKERS_BWD 8
#define WORKERS_BWD_QO 8 

#define NUM_WARPGROUPS_BWD    (WORKERS_BWD/(kittens::WARPGROUP_WARPS))
#define NUM_WARPGROUPS_BWD_QO (WORKERS_BWD_QO/(kittens::WARPGROUP_WARPS))

#define tile_h 4
#define tile_h_qo 4
#define tile_w 64/16

using layout_wgmma     = ducks::st_layout::wgmma_swizzle;
using layout_wgmma_itl = ducks::st_layout::wgmma_interleave;
using layout_tma_swi   = ducks::st_layout::swizzle; 

#define k_smem_tile  st_bf<tile_h, tile_w, layout_wgmma_itl>
#define v_smem_tile  st_bf<tile_h, tile_w, layout_wgmma>

#define q_smem_tile  st_bf<tile_h_qo, tile_w, layout_wgmma_itl>
#define og_smem_tile st_bf<tile_h_qo, tile_w, layout_wgmma_itl>
#define qg_smem_tile st_bf<tile_h_qo, tile_w, layout_tma_swi>
#define l_smem_tile  st_bf<tile_h_qo, tile_w, layout_tma_swi>::col_vec
#define d_smem_tile  st_bf<tile_h_qo, tile_w, layout_tma_swi>::col_vec

using namespace cooperative_groups;
namespace cg = cooperative_groups;

#define KV_BLOCKS 2

__global__ __launch_bounds__(WORKERS_BWD*kittens::WARP_THREADS, 1)
void attend_ker_bwd_train(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, 
                            CUtensorMap* tma_l_vec, CUtensorMap* tma_d_vec, 
                            CUtensorMap* tma_og, CUtensorMap* tma_qg, CUtensorMap* tma_kg, CUtensorMap* tma_vg)
{
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    k_smem_tile  (&k_smem) [NUM_WARPGROUPS_BWD] = al.allocate<k_smem_tile, NUM_WARPGROUPS_BWD>();
    v_smem_tile  (&v_smem) [NUM_WARPGROUPS_BWD] = al.allocate<v_smem_tile, NUM_WARPGROUPS_BWD>();

    q_smem_tile  (&q_smem) [2][NUM_WARPGROUPS_BWD_QO]                     = al.allocate<q_smem_tile,  2, NUM_WARPGROUPS_BWD_QO>();
    og_smem_tile (&og_smem)[2][NUM_WARPGROUPS_BWD_QO]                     = al.allocate<og_smem_tile, 2, NUM_WARPGROUPS_BWD_QO>();
    qg_smem_tile (&qg_smem)[2][NUM_WARPGROUPS_BWD_QO][NUM_WARPGROUPS_BWD] = al.allocate<qg_smem_tile, 2, NUM_WARPGROUPS_BWD_QO, NUM_WARPGROUPS_BWD>();
    
    l_smem_tile  (&l_smem) [2][NUM_WARPGROUPS_BWD_QO]                     = al.allocate<l_smem_tile,  2, NUM_WARPGROUPS_BWD_QO>();
    d_smem_tile  (&d_smem) [2][NUM_WARPGROUPS_BWD_QO]                     = al.allocate<d_smem_tile,  2, NUM_WARPGROUPS_BWD_QO>();

    rt_fl<tile_h/kittens::WARPGROUP_WARPS, tile_w> kg_reg;
    rt_fl<tile_h/kittens::WARPGROUP_WARPS, tile_w> vg_reg;

    rt_fl<tile_h_qo/kittens::WARPGROUP_WARPS, tile_w> qg_reg;

    rt_bf<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h>::col_vec l_reg_bf; 
    rt_bf<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h>::col_vec d_reg_bf;
    rt_fl<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h>::col_vec l_reg_fl; 
    rt_fl<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h>::col_vec d_reg_fl;

    rt_fl<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h> att_block; 
    rt_bf<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h> att_block_mma;
    rt_fl<tile_h_qo/kittens::WARPGROUP_WARPS, tile_h> temp_block;

    int warpid = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    constexpr int qo_blocks = ATTN_N / (tile_h_qo * kittens::TILE_DIM * NUM_WARPGROUPS_BWD_QO);

    __shared__ uint64_t kv_b, qo_b, vec_b;

    int tic = 0, toc = 1;

    int kv_phasebit  = 0;
    int qo_phasebit  = 0;
    int vec_phasebit = 0;

    if (threadIdx.x == 0) {
        tma::init_barrier<q_smem_tile,  NUM_WARPGROUPS_BWD_QO * 2>(qo_b,  1); // q, og
        tma::init_barrier<k_smem_tile , NUM_WARPGROUPS_BWD    * 2>(kv_b,  1); // k, v
        tma::init_barrier<l_smem_tile , NUM_WARPGROUPS_BWD_QO * 2>(vec_b, 1); // l, d
    } 

    if (warpid == 0) {
        for (int w = 0; w < NUM_WARPGROUPS_BWD_QO; w++) {
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD_QO * qo_blocks) + (0 * NUM_WARPGROUPS_BWD_QO) + w;

            tma::load_async((q_smem [tic][w]),    tma_q,     qo_b, tile_idx); 
            tma::load_async((og_smem[tic][w]),    tma_og,    qo_b, tile_idx);

            tma::load_async((l_smem[tic][w]),     tma_l_vec, vec_b, tile_idx);
            tma::load_async((d_smem[tic][w]),     tma_d_vec, vec_b, tile_idx);
        } 
    }

    for (int kv_idx = 0; kv_idx < KV_BLOCKS; kv_idx++) {
        
        if (warpid == 0) {
            // load k and v
            for (int w = 0; w < NUM_WARPGROUPS_BWD; w++) {
                int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD * KV_BLOCKS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS_BWD * KV_BLOCKS) + (kv_idx * NUM_WARPGROUPS_BWD) + w;
                tma::load_async((k_smem[w]), tma_k, kv_b, tile_idx); 
                tma::load_async((v_smem[w]), tma_v, kv_b, tile_idx); 
            }
        }

        zero(kg_reg);
        zero(vg_reg);

        for (int qo_idx = 0; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
            
            tma::arrive_and_wait(vec_b, vec_phasebit);
            tma::arrive_and_wait(qo_b,  qo_phasebit);
            vec_phasebit ^= 1;
            qo_phasebit  ^= 1;

            if (qo_idx + 1 < qo_blocks) {
                if (threadIdx.x == 0) {
                    tma::set_bytes(qo_b,  NUM_WARPGROUPS_BWD_QO * sizeof(bf16) * q_smem[0][0].num_elements * 2); 
                    tma::set_bytes(vec_b, NUM_WARPGROUPS_BWD_QO * sizeof(bf16) * l_smem[0][0].length       * 2);
                }

                if (warpid == 0) {
                    for (int w = 0; w < NUM_WARPGROUPS_BWD_QO; w++) {
                        int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD_QO * qo_blocks) + ((qo_idx + 1) * NUM_WARPGROUPS_BWD_QO) + w;
                        tma::load_async((q_smem [toc][w]),    tma_q,   qo_b, tile_idx); 
                        tma::load_async((og_smem[toc][w]),    tma_og,  qo_b, tile_idx);

                        tma::load_async((l_smem[toc][w]),     tma_l_vec, vec_b, tile_idx);
                        tma::load_async((d_smem[toc][w]),     tma_d_vec, vec_b, tile_idx);
                    }
                }
            }
            else if (kv_idx + 1 < KV_BLOCKS) {
                if (threadIdx.x == 0) {
                    tma::set_bytes(qo_b,  NUM_WARPGROUPS_BWD_QO * sizeof(bf16) * q_smem[0][0].num_elements * 2); 
                    tma::set_bytes(vec_b, NUM_WARPGROUPS_BWD_QO * sizeof(bf16) * l_smem[0][0].length       * 2);
                }

                if (warpid == 0) {
                    for (int w = 0; w < NUM_WARPGROUPS_BWD_QO; w++) {
                        int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD_QO * qo_blocks) + (0 * NUM_WARPGROUPS_BWD_QO) + w;
                        tma::load_async((q_smem [toc][w]),    tma_q,   qo_b, tile_idx); 
                        tma::load_async((og_smem[toc][w]),    tma_og,  qo_b, tile_idx); 

                        tma::load_async((l_smem[toc][w]),     tma_l_vec, vec_b, tile_idx);
                        tma::load_async((d_smem[toc][w]),     tma_d_vec, vec_b, tile_idx);
                    }
                }
            } 

            if (qo_idx == 0) {
                tma::arrive_and_wait(kv_b, kv_phasebit);
                kv_phasebit ^= 1;

                if (KV_BLOCKS > 1) {
                    if (threadIdx.x == 0) {
                        tma::set_bytes(kv_b, NUM_WARPGROUPS_BWD * sizeof(bf16) * k_smem[0].num_elements * 2);
                    }
                }
            }

            if (qo_idx > 0 || kv_idx > 0) {
                tma::store_async_wait(); 
            }


            for (int subtile = 0; subtile < NUM_WARPGROUPS_BWD_QO; subtile++) {
                warpgroup::mma_fence(att_block);
                warpgroup::mm_ABt(att_block, q_smem[tic][subtile], k_smem[warpgroupid]);
                warpgroup::mma_commit_group();

                warpgroup::load(l_reg_bf, l_smem[tic][subtile]);
                copy(l_reg_fl, l_reg_bf);
                
                warpgroup::mma_async_wait();
                mul(att_block, att_block, __bfloat162float(__float2bfloat16(0.125f)));
                sub_row(att_block, att_block, l_reg_fl);
                exp(att_block, att_block);
                copy(temp_block, att_block);
                copy(att_block_mma, att_block);

                auto (*att_smem)[NUM_WARPGROUPS_BWD_QO][NUM_WARPGROUPS_BWD] = reinterpret_cast<st_bf<tile_h_qo, tile_w, layout_wgmma_itl> (*)[NUM_WARPGROUPS_BWD_QO][NUM_WARPGROUPS_BWD]>(qg_smem); 

                warpgroup::store(att_smem[tic][subtile][warpgroupid], att_block_mma);
                __syncthreads(); 
        
                warpgroup::mma_fence(att_block);
                warpgroup::mm_ABt(att_block, og_smem[tic][subtile], v_smem[warpgroupid]);
                warpgroup::mma_commit_group();

                warpgroup::load(d_reg_bf, d_smem[tic][subtile]);
                copy(d_reg_fl, d_reg_bf);

                warpgroup::mma_fence(vg_reg);
                warpgroup::mma_AtB(vg_reg, att_smem[tic][subtile][warpgroupid], og_smem[tic][subtile]);
                warpgroup::mma_commit_group();

                warpgroup::mma_async_wait<1>();
                sub_row(att_block, att_block, d_reg_fl);
                mul(temp_block, temp_block, att_block);
                mul(temp_block, temp_block, __bfloat162float(__float2bfloat16(0.125f)));
                copy(att_block_mma, temp_block);

                warpgroup::mma_async_wait(); 
                warpgroup::store(att_smem[tic][subtile][warpgroupid], att_block_mma);
                __syncthreads();

                zero(qg_reg);
                warpgroup::mma_fence(qg_reg);
                warpgroup::mma_AB(qg_reg, att_block_mma, k_smem[warpgroupid]);
                warpgroup::mma_commit_group(); 

                warpgroup::mma_fence(kg_reg);
                warpgroup::mma_AtB(kg_reg, att_smem[tic][subtile][warpgroupid], q_smem[tic][subtile]);
                warpgroup::mma_commit_group();
                
                warpgroup::mma_async_wait();
                warpgroup::store(qg_smem[tic][subtile][warpgroupid], qg_reg);
            }

            if (warpid % 4 == 0) {
                int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD_QO * qo_blocks) + (qo_idx * NUM_WARPGROUPS_BWD_QO) + warpgroupid; 
                for (int idx = 0; idx < NUM_WARPGROUPS_BWD; idx++) {
                    tma::store_sum_async(tma_qg, (qg_smem[tic][warpgroupid][idx]), tile_idx); 
                }
                tma::store_commit_group();
            }
        }

        warpgroup::store(k_smem[warpgroupid], kg_reg);
        warpgroup::store(v_smem[warpgroupid], vg_reg);
        __syncthreads();

        if (warpid % 4 == 0) {
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD * KV_BLOCKS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS_BWD * KV_BLOCKS) + (kv_idx * NUM_WARPGROUPS_BWD) + warpgroupid; 
            tma::store_async(tma_kg, (k_smem[warpgroupid]), tile_idx);
            tma::store_async(tma_vg, (v_smem[warpgroupid]), tile_idx);
            tma::store_commit_group();
        }
    }
    tma::store_async_wait();
}

// #include "harness_h100_bwd.impl" // (comment out when using the code below)

#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>

void fwd_train_attend_ker_tk(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l) {
    std::cout << "Entered forward attention kernel handler" << std::endl;

    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    CHECK_INPUT(l);

    auto batch = q.size(0);
    auto heads = q.size(1);
    auto threads = NUM_WORKERS * kittens::WARP_THREADS;
    auto n     = q.size(2);
    auto d     = q.size(3);

    TORCH_CHECK(batch == ATTN_B, "Batch size is hard coded - if you change in PyTorch, change in h100_train.cu too");
    TORCH_CHECK(heads == ATTN_H, "Num heads is hard coded - if you change in PyTorch, change in h100_train.cu too");
    TORCH_CHECK(n == ATTN_N, "Num elements is hard coded - if you change in PyTorch, change in h100_train.cu too");
    TORCH_CHECK(d == ATTN_D, "Num elements is hard coded - if you change in PyTorch, change in h100_train.cu too");

    TORCH_CHECK(n % (NUM_WORKERS * kittens::TILE_DIM) == 0, "The number of elements should be divisible the number of workers times the tile dimension");

    // convert to bf16
    c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();
    c10::BFloat16 *l_ptr = l.data_ptr<c10::BFloat16>();

    bf16* q_bf = reinterpret_cast<bf16*>(q_ptr);
    bf16* k_bf = reinterpret_cast<bf16*>(k_ptr);
    bf16* v_bf = reinterpret_cast<bf16*>(v_ptr);
    bf16* o_bf = reinterpret_cast<bf16*>(o_ptr);
    bf16* l_bf = reinterpret_cast<bf16*>(l_ptr);

    CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<qo_height, tile_width, layout_q>,          (ATTN_B*ATTN_H*ATTN_N)/(qo_height * 16)>(q_bf);
    CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<kv_height, tile_width, layout_k>,          (ATTN_B*ATTN_H*ATTN_N)/(kv_height * 16)>(k_bf);
    CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<kv_height, tile_width, layout_v>,          (ATTN_B*ATTN_H*ATTN_N)/(kv_height * 16)>(v_bf);
    CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<qo_height, tile_width, layout_o>,          (ATTN_B*ATTN_H*ATTN_N)/(qo_height * 16)>(o_bf);
    CUtensorMap* tma_l_d = tma::allocate_and_create_tensor_map<kittens::st_bf<qo_height, tile_width, layout_q>::col_vec, (ATTN_B*ATTN_H*ATTN_N)/(qo_height * 16)>(l_bf);

    std::cout << "Check and casts" << std::endl;
    unsigned long mem_size = 227000;
    cudaFuncSetAttribute(attend_ker_fwd_train, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    std::cout << "Launching kernel" << std::endl;

    dim3 grid(n/(NUM_WORKERS*kittens::TILE_DIM), batch*heads, 1);
    attend_ker_fwd_train<<<grid, threads, mem_size>>>(tma_q_d, tma_k_d, tma_v_d, tma_o_d, tma_l_d);

    std::cout << "Kernel launched" << std::endl;
    CHECK_CUDA_ERROR(cudaGetLastError());
    std::cout << "Exiting forward train attention kernel handler" << std::endl;
}

void prep_train_attend_ker_tk(torch::Tensor o, torch::Tensor og, torch::Tensor d_vec) { 
    std::cout << "Entered prep train attention kernel handler" << std::endl;

    CHECK_INPUT(o);
    CHECK_INPUT(og);
    CHECK_INPUT(d_vec);

    auto batch = o.size(0);
    auto heads = o.size(1);
    auto n     = o.size(2);
    auto d     = o.size(3);

    auto threads = WORKERS * kittens::WARP_THREADS;

    TORCH_CHECK(batch == ATTN_B, "Batch size is hard coded - if you change in PyTorch, change in h100_train.cu too");
    TORCH_CHECK(heads == ATTN_H, "Num heads is hard coded - if you change in PyTorch, change in h100_train.cu too");
    TORCH_CHECK(n == ATTN_N, "Num elements is hard coded - if you change in PyTorch, change in h100_train.cu too");
    TORCH_CHECK(d == ATTN_D, "Num elements is hard coded - if you change in PyTorch, change in h100_train.cu too");

    TORCH_CHECK(n % (WORKERS * kittens::TILE_DIM * 4) == 0, "The number of elements should be divisible the number of workers times the tile dimension");

    // convert to bf16
    c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();
    c10::BFloat16 *og_ptr = og.data_ptr<c10::BFloat16>();
    c10::BFloat16 *d_vec_ptr = d_vec.data_ptr<c10::BFloat16>();

    bf16* o_bf = reinterpret_cast<bf16*>(o_ptr);
    bf16* og_bf = reinterpret_cast<bf16*>(og_ptr);
    bf16* d_vec_bf = reinterpret_cast<bf16*>(d_vec_ptr);

    CUtensorMap* tma_o_d_pre  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, layout_nrow>,          (ATTN_B*ATTN_H*ATTN_N)/(4*16)>(o_bf);
    CUtensorMap* tma_d_d_pre  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, layout_nrow>::col_vec, (ATTN_B*ATTN_H*ATTN_N)/(4*16)>(d_vec_bf);
    CUtensorMap* tma_og_d_pre = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, layout_nrow>,          (ATTN_B*ATTN_H*ATTN_N)/(4*16)>(og_bf);

    std::cout << "Check and casts" << std::endl;
    unsigned long mem_size = 227000;
    cudaFuncSetAttribute(attend_ker_prep_train, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    std::cout << "Launching kernel" << std::endl;

    dim3 grid_1(n/(WORKERS*kittens::TILE_DIM), batch*heads, 1);
    auto threads_1 = WORKERS * kittens::WARP_THREADS;

    attend_ker_prep_train<<<grid_1, threads_1, mem_size>>>(tma_o_d_pre, tma_d_d_pre, tma_og_d_pre);

    std::cout << "Kernel launched" << std::endl;
    CHECK_CUDA_ERROR(cudaGetLastError());
    std::cout << "Exiting prep train attention kernel handler" << std::endl;
}

void bwd_train_attend_ker_tk(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor l_vec, torch::Tensor d_vec, torch::Tensor og, torch::Tensor qg, torch::Tensor kg, torch::Tensor vg) {
    std::cout << "Entered backward train attention kernel handler" << std::endl;

    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(l_vec);
    CHECK_INPUT(d_vec);
    CHECK_INPUT(og);
    CHECK_INPUT(qg);
    CHECK_INPUT(kg);
    CHECK_INPUT(vg);

    auto batch = q.size(0);
    auto heads = q.size(1);
    auto n     = q.size(2);
    auto d     = q.size(3);

    auto threads = WORKERS_BWD * kittens::WARP_THREADS;

    TORCH_CHECK(batch == ATTN_B, "Batch size is hard coded - if you change in PyTorch, change in h100_train.cu too");
    TORCH_CHECK(heads == ATTN_H, "Num heads is hard coded - if you change in PyTorch, change in h100_train.cu too");
    TORCH_CHECK(n == ATTN_N, "Num elements is hard coded - if you change in PyTorch, change in h100_train.cu too");
    TORCH_CHECK(d == ATTN_D, "Num elements is hard coded - if you change in PyTorch, change in h100_train.cu too");

    TORCH_CHECK(n % (WORKERS_BWD * kittens::TILE_DIM * 4) == 0, "The number of elements should be divisible the number of workers times the tile dimension");

    // convert to bf16
    c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
    c10::BFloat16 *l_ptr = l_vec.data_ptr<c10::BFloat16>();
    c10::BFloat16 *d_ptr = d_vec.data_ptr<c10::BFloat16>();
    c10::BFloat16 *og_ptr = og.data_ptr<c10::BFloat16>();
    c10::BFloat16 *qg_ptr = qg.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kg_ptr = kg.data_ptr<c10::BFloat16>();
    c10::BFloat16 *vg_ptr = vg.data_ptr<c10::BFloat16>();

    bf16* q_bf = reinterpret_cast<bf16*>(q_ptr);
    bf16* k_bf = reinterpret_cast<bf16*>(k_ptr);
    bf16* v_bf = reinterpret_cast<bf16*>(v_ptr);
    bf16* l_bf = reinterpret_cast<bf16*>(l_ptr);
    bf16* d_bf = reinterpret_cast<bf16*>(d_ptr);
    bf16* og_bf = reinterpret_cast<bf16*>(og_ptr);
    bf16* qg_bf = reinterpret_cast<bf16*>(qg_ptr);
    bf16* kg_bf = reinterpret_cast<bf16*>(kg_ptr);
    bf16* vg_bf = reinterpret_cast<bf16*>(vg_ptr);

    CUtensorMap* tma_q_d_bwd = tma::allocate_and_create_tensor_map<q_smem_tile, (ATTN_B*ATTN_H*ATTN_N)/(tile_h_qo * 16)>(q_bf);
    CUtensorMap* tma_k_d_bwd = tma::allocate_and_create_tensor_map<k_smem_tile, (ATTN_B*ATTN_H*ATTN_N)/(tile_h * 16)>(k_bf);
    CUtensorMap* tma_v_d_bwd = tma::allocate_and_create_tensor_map<v_smem_tile, (ATTN_B*ATTN_H*ATTN_N)/(tile_h * 16)>(v_bf);

    CUtensorMap* tma_l_d_bwd = tma::allocate_and_create_tensor_map<l_smem_tile, (ATTN_B*ATTN_H*ATTN_N)/(tile_h_qo * 16)>(l_bf);
    CUtensorMap* tma_d_d_bwd = tma::allocate_and_create_tensor_map<d_smem_tile, (ATTN_B*ATTN_H*ATTN_N)/(tile_h_qo * 16)>(d_bf);

    CUtensorMap* tma_og_d_bwd = tma::allocate_and_create_tensor_map<og_smem_tile, (ATTN_B*ATTN_H*ATTN_N)/(tile_h_qo * 16)>(og_bf);
    CUtensorMap* tma_qg_d_bwd = tma::allocate_and_create_tensor_map<qg_smem_tile, (ATTN_B*ATTN_H*ATTN_N)/(tile_h_qo * 16)>(qg_bf);
    CUtensorMap* tma_kg_d_bwd = tma::allocate_and_create_tensor_map<k_smem_tile, (ATTN_B*ATTN_H*ATTN_N)/(tile_h * 16)>(kg_bf);
    CUtensorMap* tma_vg_d_bwd = tma::allocate_and_create_tensor_map<v_smem_tile, (ATTN_B*ATTN_H*ATTN_N)/(tile_h * 16)>(vg_bf);

    std::cout << "Check and casts" << std::endl;
    unsigned long mem_size = 227000;
    cudaFuncSetAttribute(attend_ker_bwd_train, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    std::cout << "Launching kernel" << std::endl;

    dim3 grid_2(n/(KV_BLOCKS*WORKERS_BWD*kittens::TILE_DIM), batch*heads, 1);
    auto threads_2 = WORKERS_BWD * kittens::WARP_THREADS;

    attend_ker_bwd_train<<<grid_2, threads_2, mem_size>>>(tma_q_d_bwd, tma_k_d_bwd, tma_v_d_bwd, tma_l_d_bwd, tma_d_d_bwd, tma_og_d_bwd, tma_qg_d_bwd, tma_kg_d_bwd, tma_vg_d_bwd);

    std::cout << "Kernel launched" << std::endl;
    CHECK_CUDA_ERROR(cudaGetLastError());
    std::cout << "Exiting backward train attention kernel handler" << std::endl;
}