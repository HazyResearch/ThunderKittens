#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

constexpr int CONSUMER_WARPGROUPS = (3); 
constexpr int PRODUCER_WARPGROUPS = (1); 
constexpr int NUM_WARPGROUPS      = (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS); 
constexpr int NUM_WORKERS         = (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 

using namespace kittens;
namespace cg = cooperative_groups;

template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64/kittens::TILE_DIM);
    constexpr static int qo_height  = (4);
    constexpr static int kv_height  = (12);
};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128/kittens::TILE_DIM);
    constexpr static int qo_height  = (4);
    constexpr static int kv_height  = (8);
};

template<int D, bool is_causal>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_attend_ker(int N, int heads_ratio, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, CUtensorMap* tma_o, CUtensorMap* tma_l) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile    = st_bf<K::qo_height, K::tile_width>;
    using k_tile    = st_bf<K::kv_height, K::tile_width>;
    using v_tile    = st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile    = st_bf<K::qo_height, K::tile_width>;
    
    q_tile    (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[2]                   = al.allocate<k_tile, 2                  >();
    v_tile    (&v_smem)[2]                   = al.allocate<v_tile, 2                  >();
    l_col_vec (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, CONSUMER_WARPGROUPS>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem); // reuse q memory
    
    int kv_blocks = N / (K::kv_height*TILE_DIM);

    __shared__ kittens::barrier qsmem_barrier, k_smem_arrived[2], v_smem_arrived[2], compute_done[2];
    if (threadIdx.x == 0) { // initialize barriers and initial loads
        init_barrier(qsmem_barrier, 0, 1); // no threads, one transaction
        for(int j = 0; j < 2; j++) {
            init_barrier(k_smem_arrived[j], 0, 1); // no threads, one transaction
            init_barrier(v_smem_arrived[j], 0, 1); // no threads, one transaction
            init_barrier(compute_done[j], CONSUMER_WARPGROUPS, 0); // all the consumer threads across both blocks, no transactions
        }
        
        tma::expect_bytes(qsmem_barrier, sizeof(q_smem));
        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) { // issue async loads for Q chunks
            int q_tile_idx = (blockIdx.y * CONSUMER_WARPGROUPS * gridDim.x) + (blockIdx.x * CONSUMER_WARPGROUPS) + wg;
            tma::load_async(q_smem[wg], tma_q, qsmem_barrier, q_tile_idx); 
        }
        
        int kv_tile_idx = ((blockIdx.y/heads_ratio) * kv_blocks) + 0; 
        tma::expect<typeof(k_smem[0])>(k_smem_arrived[0]);
        tma::expect<typeof(v_smem[0])>(v_smem_arrived[0]);
        tma::load_async(k_smem[0], tma_k, k_smem_arrived[0], kv_tile_idx); 
        tma::load_async(v_smem[0], tma_v, v_smem_arrived[0], kv_tile_idx);
    }

    __syncthreads(); 

    int tic = 0, toc = 1;
    if(warpgroupid == NUM_WARPGROUPS-1) { // producer warpgroup
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));           
        
        int kv_iters; 
        if constexpr (is_causal) {
            kv_iters = (blockIdx.x * CONSUMER_WARPGROUPS * K::qo_height) - 1 + (CONSUMER_WARPGROUPS * K::qo_height); 
            kv_iters = ((kv_iters / K::kv_height) == 0) ? (0) : ((kv_iters / K::kv_height) - 1);
        }
        else {
            kv_iters = kv_blocks-2; 
        }

        if(warpid == NUM_WORKERS-4) {
            for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++, tic^=1, toc^=1) {
                int kv_tile_idx = ((blockIdx.y/heads_ratio) * kv_blocks) + (kv_idx + 1);
                
                tma::expect<typeof(k_smem[0])>(k_smem_arrived[toc]); 
                tma::load_async(k_smem[toc], tma_k, k_smem_arrived[toc], kv_tile_idx, 0);
                
                tma::expect<typeof(v_smem[0])>(v_smem_arrived[toc]); 
                tma::load_async(v_smem[toc], tma_v, v_smem_arrived[toc], kv_tile_idx, 0);
                
                wait(compute_done[tic], (kv_idx/2)%2);
            }
        }
        __syncthreads();
    }
    else { // consumer warpgroup
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(160));

        // premultiply by temperature and lg(e)
        wait(qsmem_barrier, 0);
        if constexpr (D == 64) { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f)); }
        else                   { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.08838834764f)); }
    
        rt_fl<1, K::kv_height> att_block;
        rt_bf<1, K::kv_height> att_block_mma;
        rt_fl<1, K::tile_width> o_reg;
        col_vec<rt_fl<1, K::kv_height>> max_vec_last, max_vec;
        col_vec<rt_fl<1, K::kv_height>> norm_vec_last, norm_vec;
        
        neg_infty(max_vec); // clear registers for the Q chunk
        zero(norm_vec);
        zero(o_reg);

        int kv_iters; 
        if constexpr (is_causal) {
            kv_iters = (blockIdx.x * CONSUMER_WARPGROUPS * K::qo_height) - 1 + ((warpgroupid + 1) * K::qo_height);
            kv_iters = (kv_iters/K::kv_height);
        }
        else { 
            kv_iters = kv_blocks - 1; 
        }
        
        const int kv_do = (blockIdx.x * CONSUMER_WARPGROUPS)/(K::kv_height/K::qo_height);

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++, tic^=1, toc^=1) {
        
            wait(k_smem_arrived[tic], (kv_idx/2)%2); // wait on k memory
            
            warpgroup::mma_fence(att_block);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[tic]);
            warpgroup::mma_commit_group();
            
            wait(v_smem_arrived[tic], (kv_idx/2)%2); // wait on v memory, during the matmul
            
            copy(norm_vec_last, norm_vec);
            copy(max_vec_last,  max_vec);
            
            warpgroup::mma_async_wait();

            if constexpr (is_causal) {
                int q_blk = (blockIdx.x * CONSUMER_WARPGROUPS * K::qo_height) + warpid; 
                int k_blk = (kv_idx * K::kv_height); 

                #pragma unroll
                for (int j = 0; j < K::kv_height; j++) {
                    int k_idx = k_blk + j;
                    auto &attn_subtile = reinterpret_cast<rt_fl_1x1<>&>(att_block.tiles[0][j]);

                    if      (k_idx >  q_blk) { neg_infty  (attn_subtile); }
                    else if (k_idx == q_blk) { make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
                }
            }

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
            mul_row(o_reg, o_reg, norm_vec_last); // normalize o_prev in advance of mma'ing onto it

            warpgroup::mma_fence(o_reg);
            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[tic]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
            
            if(warpgroup::laneid() == 0) {
                int count = ((warpgroupid == CONSUMER_WARPGROUPS - 1) && (kv_idx > kv_do) && is_causal) ? (2 - (blockIdx.x % 2)) : 1;
                arrive(compute_done[tic], count); 
            }
        }

        warpgroup::store(o_smem[warpgroupid], o_reg); 
        __syncthreads();

        if (warpid % 4 == 0) { // store o
            int tile_idx = (blockIdx.y * CONSUMER_WARPGROUPS * gridDim.x) + (blockIdx.x * CONSUMER_WARPGROUPS) + warpgroupid;
            tma::store_async(tma_o, o_smem[warpgroupid], tile_idx); 
            tma::store_commit_group(); 
        }

        log(norm_vec, norm_vec);
        add(norm_vec, norm_vec, max_vec);
        __syncthreads();
    
        warpgroup::store(l_smem[warpgroupid], norm_vec);
        __syncthreads();
        
        if (warpid % 4 == 0) {
            int tile_idx = (blockIdx.y * CONSUMER_WARPGROUPS * gridDim.x) + (blockIdx.x * CONSUMER_WARPGROUPS) + warpgroupid; 
            tma::store_async(tma_l, l_smem[warpgroupid], tile_idx); 
            tma::store_commit_group(); 
        }
    
        tma::store_async_wait();
    }
}

template<int D>
__global__  __launch_bounds__(4*kittens::WARP_THREADS, (D == 64) ? 2 : 1)
void bwd_attend_prep_ker(CUtensorMap* tma_o, CUtensorMap* tma_d, CUtensorMap* tma_o_grad) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();

    // initialize shared memory
    using og_tile = st_bf<4, D/kittens::TILE_DIM>;
    using o_tile  = st_bf<4, D/kittens::TILE_DIM>;
    using d_tile  = col_vec<st_fl<4, D/kittens::TILE_DIM>>;

    og_tile (&og_smem)[4] = al.allocate<og_tile, 4>();
    o_tile  (&o_smem) [4] = al.allocate<o_tile , 4>();
    d_tile  (&d_smem) [4] = al.allocate<d_tile , 4>();
    //////////////////////

    // initialize registers
    using fl_reg_tile = rt_fl<4, D/kittens::TILE_DIM>;
    using fl_reg_col  = col_vec<rt_fl<4, D/kittens::TILE_DIM>>;
    
    fl_reg_tile og_reg;
    fl_reg_tile  o_reg; 
    fl_reg_col d_reg;
    //////////////////////

    __shared__ kittens::barrier smem_barrier;

    if (threadIdx.x == 0) {
        init_barrier(smem_barrier, 0, 1);
        tma::expect_bytes(smem_barrier, sizeof(og_smem[0]) * 4 * 2);
    }
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < 4; w++) { // load o, o_grad
            int tile_idx = (blockIdx.y * 4 * gridDim.x) + (blockIdx.x * 4) + w; 
            tma::load_async((o_smem[w]),  tma_o,      smem_barrier, tile_idx); 
            tma::load_async((og_smem[w]), tma_o_grad, smem_barrier, tile_idx); 
        }
    }

    wait(smem_barrier, 0);

    load(o_reg, o_smem[warpid]);
    load(og_reg, og_smem[warpid]);

    mul(og_reg, og_reg, o_reg);
    row_sum(d_reg, og_reg);
    
    store(d_smem[warpid], d_reg);
    __syncthreads(); 

    if (warpid == 0) {
        for (int w = 0; w < 4; w++) {
            int tile_idx = (blockIdx.y * 4 * gridDim.x) + (blockIdx.x * 4) + w; 
            tma::store_async(tma_d, (d_smem[w]), tile_idx); 
        }
        tma::store_commit_group();
    }

    tma::store_async_wait();
}

constexpr int WORKERS_BWD    = 4;  
constexpr int WORKERS_BWD_QO = 4; 

constexpr int NUM_WARPGROUPS_BWD    = (WORKERS_BWD/(kittens::WARPGROUP_WARPS));
constexpr int NUM_WARPGROUPS_BWD_QO = (WORKERS_BWD_QO/(kittens::WARPGROUP_WARPS));

template<int D> struct bwd_attend_ker_tile_dims {};
template<> struct bwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64/kittens::TILE_DIM);
    constexpr static int tile_h  = (4);
    constexpr static int tile_h_qo  = (4);
};
template<> struct bwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128/kittens::TILE_DIM);
    constexpr static int tile_h  = (4);
    constexpr static int tile_h_qo  = (4);
};

template<int D, bool is_causal>
__global__ __launch_bounds__(WORKERS_BWD*kittens::WARP_THREADS, 1)
void bwd_attend_ker(int N, int heads_ratio, CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, 
                        CUtensorMap* tma_l_vec, CUtensorMap* tma_d_vec, 
                        CUtensorMap* tma_og, CUtensorMap* tma_qg, CUtensorMap* tma_kg, CUtensorMap* tma_vg) {
                        
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    using G            = bwd_attend_ker_tile_dims<D>;
    const int kv_iters = (is_causal) ? (1) : (2); 

    // initialize kg and vg (outputs) shared memory (reused for k and v (inputs))
    using kg_tile = st_fl<G::tile_h, G::tile_width>;
    using vg_tile = st_fl<G::tile_h, G::tile_width>;

    using k_tile = st_bf<G::tile_h, G::tile_width>;
    using v_tile = st_bf<G::tile_h, G::tile_width>;

    kg_tile (&kg_smem) [NUM_WARPGROUPS_BWD] = al.allocate<kg_tile, NUM_WARPGROUPS_BWD>();
    vg_tile (&vg_smem) [NUM_WARPGROUPS_BWD] = al.allocate<vg_tile, NUM_WARPGROUPS_BWD>();
    auto     (*k_smem)                      = reinterpret_cast<k_tile(*)>(kg_smem);
    auto     (*v_smem)                      = reinterpret_cast<v_tile(*)>(vg_smem);

    // initialize q, og (inputs) and qg (output) shared memory
    using q_tile  = st_bf<G::tile_h_qo, G::tile_width>;
    using og_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile = st_fl<G::tile_h_qo, G::tile_width>;

    q_tile  (&q_smem) [2][NUM_WARPGROUPS_BWD_QO]                     = al.allocate<q_tile , 2, NUM_WARPGROUPS_BWD_QO>();
    og_tile (&og_smem)[2][NUM_WARPGROUPS_BWD_QO]                     = al.allocate<og_tile, 2, NUM_WARPGROUPS_BWD_QO>();
    qg_tile (&qg_smem)[2][NUM_WARPGROUPS_BWD_QO][NUM_WARPGROUPS_BWD] = al.allocate<qg_tile, 2, NUM_WARPGROUPS_BWD_QO, NUM_WARPGROUPS_BWD>();

    // initialize l and d (inputs) shared memory
    using l_tile = col_vec<st_fl<G::tile_h_qo, G::tile_width>>;
    using d_tile = col_vec<st_fl<G::tile_h_qo, G::tile_width>>;

    l_tile (&l_smem)[2][NUM_WARPGROUPS_BWD_QO] = al.allocate<l_tile, 2, NUM_WARPGROUPS_BWD_QO>();
    d_tile (&d_smem)[2][NUM_WARPGROUPS_BWD_QO] = al.allocate<d_tile, 2, NUM_WARPGROUPS_BWD_QO>();

    // initialize attention shared memory (temporary)
    using attn_tile = st_bf<G::tile_h_qo, G::tile_h>; 
    
    attn_tile (&att_smem)[2][NUM_WARPGROUPS_BWD_QO][NUM_WARPGROUPS_BWD] = al.allocate<attn_tile, 2, NUM_WARPGROUPS_BWD_QO, NUM_WARPGROUPS_BWD>();
    //////////////////////

    // initialize registers
    using kg_reg_tile = rt_fl<G::tile_h/kittens::WARPGROUP_WARPS, G::tile_width>;
    using vg_reg_tile = rt_fl<G::tile_h/kittens::WARPGROUP_WARPS, G::tile_width>;
    using qg_reg_tile = rt_fl<G::tile_h_qo/kittens::WARPGROUP_WARPS, G::tile_width>; 
    
    using l_reg_vec  = col_vec<rt_fl<G::tile_h_qo/kittens::WARPGROUP_WARPS, G::tile_h>>;
    using d_reg_vec  = col_vec<rt_fl<G::tile_h_qo/kittens::WARPGROUP_WARPS, G::tile_h>>;

    using attn_reg_tile = rt_fl<G::tile_h_qo/kittens::WARPGROUP_WARPS, G::tile_h>;
    using attn_mma_tile = rt_bf<G::tile_h_qo/kittens::WARPGROUP_WARPS, G::tile_h>;
    using temp_tile     = rt_fl<G::tile_h_qo/kittens::WARPGROUP_WARPS, G::tile_h>;

    kg_reg_tile kg_reg;
    vg_reg_tile vg_reg;
    qg_reg_tile qg_reg;

    l_reg_vec l_reg; 
    d_reg_vec d_reg;

    attn_reg_tile att_block; 
    attn_mma_tile att_block_mma;
    temp_tile     temp_block;
    //////////////////////

    int warpid = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    const int qo_blocks = N / (G::tile_h_qo * kittens::TILE_DIM * NUM_WARPGROUPS_BWD_QO);

    __shared__ kittens::barrier kv_b, qo_b, vec_b;

    int tic = 0, toc = 1;

    if (threadIdx.x == 0) {
        init_barrier(qo_b,  0, 1); // q, og
        init_barrier(kv_b,  0, 1); // k, v
        init_barrier(vec_b, 0, 1); // l, d
    }
    __syncwarp(); 

    int q_start = (is_causal) ? (blockIdx.x) : (0);

    if (warpid == 0) {
        tma::expect_bytes(qo_b, (sizeof(q_smem[0][0]) + sizeof(og_smem[0][0])) * NUM_WARPGROUPS_BWD_QO);
        tma::expect_bytes(vec_b, (sizeof(l_smem[0][0]) + sizeof(d_smem[0][0])) * NUM_WARPGROUPS_BWD_QO);
        for (int w = 0; w < NUM_WARPGROUPS_BWD_QO; w++) {
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD_QO * qo_blocks) + (q_start * NUM_WARPGROUPS_BWD_QO) + w;

            tma::load_async(q_smem [tic][w], tma_q,  qo_b, tile_idx); 
            tma::load_async(og_smem[tic][w], tma_og, qo_b, tile_idx);

            tma::load_async(l_smem[tic][w], tma_l_vec, vec_b, tile_idx);
            tma::load_async(d_smem[tic][w], tma_d_vec, vec_b, tile_idx);
        }
    }

    for (int kv_idx = 0; kv_idx < kv_iters; kv_idx++) {
        
        if (warpid == 0) {
            // load k and v
            tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * NUM_WARPGROUPS_BWD);

            for (int w = 0; w < NUM_WARPGROUPS_BWD; w++) {
                int tile_idx = ((blockIdx.y/heads_ratio) * NUM_WARPGROUPS_BWD * kv_iters * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS_BWD * kv_iters) + (kv_idx * NUM_WARPGROUPS_BWD) + w;
                tma::load_async((k_smem[w]), tma_k, kv_b, tile_idx); 
                tma::load_async((v_smem[w]), tma_v, kv_b, tile_idx); 
            }
        }

        zero(kg_reg);
        zero(vg_reg);

        for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {

            wait(vec_b, tic);
            wait(qo_b,  tic);

            if (qo_idx + 1 < qo_blocks) {
                if (warpid == 0) {
                    tma::expect_bytes(qo_b,  (sizeof(q_smem[0][0]) + sizeof(og_smem[0][0])) * NUM_WARPGROUPS_BWD_QO);
                    tma::expect_bytes(vec_b, (sizeof(l_smem[0][0]) + sizeof(d_smem[0][0]))  * NUM_WARPGROUPS_BWD_QO);
    
                    for (int w = 0; w < NUM_WARPGROUPS_BWD_QO; w++) {
                        int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD_QO * qo_blocks) + ((qo_idx + 1) * NUM_WARPGROUPS_BWD_QO) + w;
                        tma::load_async((q_smem [toc][w]),    tma_q,   qo_b, tile_idx); 
                        tma::load_async((og_smem[toc][w]),    tma_og,  qo_b, tile_idx);
    
                        tma::load_async((l_smem[toc][w]),     tma_l_vec, vec_b, tile_idx);
                        tma::load_async((d_smem[toc][w]),     tma_d_vec, vec_b, tile_idx);
                    }
                }
            }
            else if (kv_idx + 1 < kv_iters) {
                if (warpid == 0) {
                    tma::expect_bytes(qo_b,  (sizeof(q_smem[0][0]) + sizeof(og_smem[0][0])) * NUM_WARPGROUPS_BWD_QO);
                    tma::expect_bytes(vec_b, (sizeof(l_smem[0][0]) + sizeof(d_smem[0][0]))  * NUM_WARPGROUPS_BWD_QO);

                    for (int w = 0; w < NUM_WARPGROUPS_BWD_QO; w++) {
                        int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD_QO * qo_blocks) + (0 * NUM_WARPGROUPS_BWD_QO) + w;
                        tma::load_async((q_smem [toc][w]),    tma_q,   qo_b, tile_idx); 
                        tma::load_async((og_smem[toc][w]),    tma_og,  qo_b, tile_idx); 

                        tma::load_async((l_smem[toc][w]),     tma_l_vec, vec_b, tile_idx);
                        tma::load_async((d_smem[toc][w]),     tma_d_vec, vec_b, tile_idx);
                    }
                }
            }

            if (qo_idx == q_start) {
                wait(kv_b, kv_idx % 2);
            }

            if (qo_idx > 0 || kv_idx > 0) {
                tma::store_async_wait(); 
            }

            warpgroup::mma_fence(att_block);
            warpgroup::mm_ABt(att_block, q_smem[tic][0], k_smem[warpgroupid]);
            warpgroup::mma_commit_group();

            warpgroup::load(l_reg, l_smem[tic][0]);

            warpgroup::mma_async_wait();

            if constexpr (D == 64) { mul(att_block, att_block, 0.125f); }
            else                   { mul(att_block, att_block, 0.08838834764f); }

            if constexpr (is_causal) {
                if (blockIdx.x == qo_idx) {
                    #pragma unroll
                    for(int j = 0; j < G::tile_h; j++) {
                        auto &attn_subtile = reinterpret_cast<rt_fl_1x1<>&>(att_block.tiles[0][j]);
                        if      (j>warpid)  neg_infty(attn_subtile);
                        else if (j==warpid) make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty());
                    }
                }
            }

            sub_row(att_block, att_block, l_reg);
            exp(att_block, att_block);
            copy(temp_block, att_block);
            copy(att_block_mma, att_block);

            warpgroup::store(att_smem[tic][0][warpgroupid], att_block_mma); 

            warpgroup::mma_fence(att_block);
            warpgroup::mm_ABt(att_block, og_smem[tic][0], v_smem[warpgroupid]);
            warpgroup::mma_commit_group();

            warpgroup::load(d_reg, d_smem[tic][0]);

            warpgroup::mma_fence(vg_reg);
            warpgroup::mma_AtB(vg_reg, att_smem[tic][0][warpgroupid], og_smem[tic][0]);
            warpgroup::mma_commit_group();
            
            warpgroup::mma_async_wait<1>();

            sub_row(att_block, att_block, d_reg);
            mul(temp_block, temp_block, att_block);
            
            if constexpr (D == 64) { mul(temp_block, temp_block, 0.125f); }
            else                   { mul(temp_block, temp_block, 0.08838834764f); }
            
            copy(att_block_mma, temp_block);

            warpgroup::mma_async_wait(); 
            warpgroup::store(att_smem[tic][0][warpgroupid], att_block_mma);

            zero(qg_reg);
            warpgroup::mma_fence(qg_reg);
            warpgroup::mma_AB(qg_reg, att_block_mma, k_smem[warpgroupid]);
            warpgroup::mma_commit_group(); 

            warpgroup::mma_fence(kg_reg);
            warpgroup::mma_AtB(kg_reg, att_smem[tic][0][warpgroupid], q_smem[tic][0]);
            warpgroup::mma_commit_group();

            warpgroup::mma_async_wait();
            warpgroup::store(qg_smem[tic][0][warpgroupid], qg_reg);
            __syncthreads();

            if (warpid % 4 == 0) {
                int tile_idx = (blockIdx.y * NUM_WARPGROUPS_BWD_QO * qo_blocks) + (qo_idx * NUM_WARPGROUPS_BWD_QO) + warpgroupid; 
                for (int idx = 0; idx < NUM_WARPGROUPS_BWD; idx++) {
                    tma::store_add_async(tma_qg, (qg_smem[tic][warpgroupid][idx]), tile_idx); 
                }
                tma::store_commit_group();
            }
        }

        warpgroup::store(kg_smem[warpgroupid], kg_reg);
        warpgroup::store(vg_smem[warpgroupid], vg_reg);
        __syncthreads();

        if (warpid % 4 == 0) {
            int tile_idx = ((blockIdx.y/heads_ratio) * NUM_WARPGROUPS_BWD * kv_iters * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS_BWD * kv_iters) + (kv_idx * NUM_WARPGROUPS_BWD) + warpgroupid; 
            tma::store_add_async(tma_kg, (kg_smem[warpgroupid]), tile_idx);
            tma::store_add_async(tma_vg, (vg_smem[warpgroupid]), tile_idx);
            tma::store_commit_group();
        }
    }

    tma::store_async_wait();
}

#include "harness.impl"