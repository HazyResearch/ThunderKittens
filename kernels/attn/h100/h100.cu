#include "include/kittens.cuh"
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

template<int D> struct bwd_attend_ker_tile_dims {};
template<> struct bwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64/kittens::TILE_DIM);
    constexpr static int tile_h  = (4);
    constexpr static int tile_h_qo  = (4);

    constexpr static int blocks_sm = 1;
};
template<> struct bwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128/kittens::TILE_DIM);
    constexpr static int tile_h  = (4);
    constexpr static int tile_h_qo  = (4);

    constexpr static int blocks_sm = 1; 
};

constexpr int BWD_CONSUMER_WARPGROUPS = (2); 
constexpr int BWD_PRODUCER_WARPGROUPS = (1); 
constexpr int BWD_NUM_WARPGROUPS      = (BWD_CONSUMER_WARPGROUPS+BWD_PRODUCER_WARPGROUPS); 
constexpr int BWD_NUM_WORKERS         = (BWD_NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 

template<int D, bool is_causal>
__global__ __launch_bounds__(BWD_NUM_WORKERS*kittens::WARP_THREADS, bwd_attend_ker_tile_dims<D>::blocks_sm)
void bwd_attend_ker(const int N, const int heads_ratio,
                        CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, 
                        CUtensorMap* tma_l_vec, CUtensorMap* tma_d_vec, 
                        CUtensorMap* tma_og, CUtensorMap* tma_qg, CUtensorMap* tma_kg, CUtensorMap* tma_vg) {
                        
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    using G = bwd_attend_ker_tile_dims<D>;
    ////////////////////////////// SHARED MEMORY //////////////////////////////
    using kg_tile = st_fl<G::tile_h, G::tile_width>;
    using vg_tile = st_fl<G::tile_h, G::tile_width>;

    using k_tile = st_bf<G::tile_h, G::tile_width>;
    using v_tile = st_bf<G::tile_h, G::tile_width>;

    using q_tile  = st_bf<G::tile_h_qo, G::tile_width>;
    using og_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile = st_fl<G::tile_h_qo, G::tile_width>;

    using l_tile = col_vec<st_fl<G::tile_h_qo, G::tile_width>>;
    using d_tile = col_vec<st_fl<G::tile_h_qo, G::tile_width>>;

    using attn_tile = st_bf<G::tile_h_qo, G::tile_h>; 
    /////////////////////////////////////////////////////////////////////////////

    ////////////////////////////// REGISTERS ////////////////////////////////////
    using kg_reg_tile = rt_fl<G::tile_h/kittens::WARPGROUP_WARPS, G::tile_width>;
    using vg_reg_tile = rt_fl<G::tile_h/kittens::WARPGROUP_WARPS, G::tile_width>;
    using qg_reg_tile = rt_fl<G::tile_h_qo/kittens::WARPGROUP_WARPS, G::tile_width>; 

    using l_reg_vec  = col_vec<rt_fl<G::tile_h_qo/kittens::WARPGROUP_WARPS, G::tile_h>>;
    using d_reg_vec  = col_vec<rt_fl<G::tile_h_qo/kittens::WARPGROUP_WARPS, G::tile_h>>;

    using attn_reg_tile = rt_fl<G::tile_h_qo/kittens::WARPGROUP_WARPS, G::tile_h>;
    using attn_mma_tile = rt_bf<G::tile_h_qo/kittens::WARPGROUP_WARPS, G::tile_h>;
    using temp_tile     = rt_fl<G::tile_h_qo/kittens::WARPGROUP_WARPS, G::tile_h>;
    ///////////////////////////////////////////////////////////////////////////////

    // initialize shared memory
    k_tile  (&k_smem) [BWD_CONSUMER_WARPGROUPS]     = al.allocate<k_tile, BWD_CONSUMER_WARPGROUPS>(); 
    v_tile  (&v_smem) [BWD_CONSUMER_WARPGROUPS]     = al.allocate<v_tile, BWD_CONSUMER_WARPGROUPS>(); 
    kg_tile (*kg_smem)                              = reinterpret_cast<kg_tile(*)>(k_smem); 
    vg_tile (*vg_smem)                              = reinterpret_cast<vg_tile(*)>(v_smem); 

    q_tile  (&q_smem) [2]                       = al.allocate<q_tile , 2>(); 
    og_tile (&og_smem)[2]                       = al.allocate<og_tile, 2>(); 
    qg_tile (&qg_smem)[BWD_CONSUMER_WARPGROUPS] = al.allocate<qg_tile, BWD_CONSUMER_WARPGROUPS>(); 

    l_tile (&l_smem)[2] = al.allocate<l_tile, 2>();
    d_tile (&d_smem)[2] = al.allocate<d_tile, 2>();

    attn_tile (&att_smem)[BWD_CONSUMER_WARPGROUPS] = al.allocate<attn_tile, BWD_CONSUMER_WARPGROUPS>(); 
    //////////////////////

    int warpid = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    const int qo_blocks = N / (G::tile_h_qo * kittens::TILE_DIM);

    __shared__ kittens::barrier kv_b, q_b[2], o_b[2], vec_b[2];
    __shared__ kittens::barrier compute_done[2]; 

    int tic = 0, toc = 1;
    int q_start = (is_causal) ? (blockIdx.x) : (0);

    if (threadIdx.x == 0) {
        // initialize barriers
        init_barrier(kv_b,  0, 1); // k, v
        for (int s = 0; s < 2; s++) {
            init_barrier(q_b[s],  0, 1); // q
            init_barrier(o_b[s],  0, 1); // o
            init_barrier(vec_b[s], 0, 1); // l, d
            
            init_barrier(compute_done[s], BWD_CONSUMER_WARPGROUPS, 0);
        }

        // load k and v
        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * BWD_CONSUMER_WARPGROUPS);
        for (int w = 0; w < BWD_CONSUMER_WARPGROUPS; w++) {
            int tile_idx = ((blockIdx.y/heads_ratio) * BWD_CONSUMER_WARPGROUPS * gridDim.x) + (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + w;
            tma::load_async((k_smem[w]), tma_k, kv_b, tile_idx); 
            tma::load_async((v_smem[w]), tma_v, kv_b, tile_idx); 
        }

        // load q, og, l, d into tic
        tma::expect_bytes(q_b[tic],  sizeof(q_smem[0]));
        tma::expect_bytes(o_b[tic],  sizeof(og_smem[0]));
        tma::expect_bytes(vec_b[tic], sizeof(l_smem[0]) + sizeof(d_smem[0]));

        int tile_idx = (blockIdx.y * qo_blocks) + (q_start); 

        tma::load_async( q_smem[tic], tma_q,       q_b[tic], tile_idx); 
        tma::load_async(og_smem[tic], tma_og,      o_b[tic], tile_idx);
        tma::load_async( l_smem[tic], tma_l_vec, vec_b[tic], tile_idx);
        tma::load_async( d_smem[tic], tma_d_vec, vec_b[tic], tile_idx);
    }

    __syncthreads(); 

    // producer
    if (warpgroupid == BWD_NUM_WARPGROUPS - 1) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(24));

        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            for (auto qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                if (qo_idx + 1 < qo_blocks) {
                    tma::expect_bytes(q_b[toc],   sizeof(q_smem[0])); 
                    tma::expect_bytes(o_b[toc],   sizeof(og_smem[0]));
                    tma::expect_bytes(vec_b[toc], sizeof(l_smem[0]) + sizeof(d_smem[0]));
                    
                    int tile_idx = (blockIdx.y * qo_blocks) + (qo_idx + 1);
                    
                    tma::load_async(q_smem [toc], tma_q,      q_b[toc], tile_idx); 
                    tma::load_async(og_smem[toc], tma_og,     o_b[toc], tile_idx);
                    tma::load_async( l_smem[toc], tma_l_vec, vec_b[toc], tile_idx);
                    tma::load_async( d_smem[toc], tma_d_vec, vec_b[toc], tile_idx);
                }
                
                wait(compute_done[tic], ((qo_idx - q_start)/(2))%2);
            }
        }
    }
    // consumer
    else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(240));

        // initialize registers
        kg_reg_tile kg_reg;
        vg_reg_tile vg_reg;
        qg_reg_tile qg_reg;

        l_reg_vec l_reg; 
        d_reg_vec d_reg;

        attn_reg_tile att_block; 
        attn_mma_tile att_block_mma;
        temp_tile     temp_block;
        //////////////////////

        zero(kg_reg);
        zero(vg_reg);

        wait(kv_b, 0);

        for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {

            wait(q_b[tic], ((qo_idx - q_start)/2)%2);
            
            warpgroup::mma_fence(att_block);
            warpgroup::mm_ABt(att_block, q_smem[tic], k_smem[warpgroupid]);
            warpgroup::mma_commit_group();

            wait(vec_b[tic], ((qo_idx - q_start)/2)%2);
            warpgroup::load(l_reg, l_smem[tic]);

            wait(o_b[tic], ((qo_idx - q_start)/2)%2);

            warpgroup::mma_async_wait();

            if constexpr (D == 64) { mul(att_block, att_block, 0.125f); }
            else                   { mul(att_block, att_block, 0.08838834764f); }

            if constexpr (is_causal) {
                int q_blk = (qo_idx * G::tile_h_qo) + (warpid % kittens::WARPGROUP_WARPS);
                int k_blk = (blockIdx.x * BWD_CONSUMER_WARPGROUPS * G::tile_h) + (warpgroupid * G::tile_h);

                for (int j = 0; j < G::tile_h; j++) {
                    int k_idx = k_blk + j;
                    auto &attn_subtile = reinterpret_cast<rt_fl_1x1<>&>(att_block.tiles[0][j]);

                    if      (k_idx >  q_blk) { neg_infty  (attn_subtile); }
                    else if (k_idx == q_blk) { make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
                }
            }

            sub_row(att_block, att_block, l_reg);
            exp(att_block, att_block);
            copy(temp_block, att_block);
            copy(att_block_mma, att_block);

            warpgroup::store(att_smem[warpgroupid], att_block_mma);
            asm volatile("bar.sync %0, 128;\n" :: "r"(warpgroupid));

            warpgroup::mma_fence(att_block);
            warpgroup::mm_ABt(att_block, og_smem[tic], v_smem[warpgroupid]);
            warpgroup::mma_commit_group();

            warpgroup::load(d_reg, d_smem[tic]);

            warpgroup::mma_fence(vg_reg);
            warpgroup::mma_AtB(vg_reg, att_smem[warpgroupid], og_smem[tic]);
            warpgroup::mma_commit_group();
            
            warpgroup::mma_async_wait<1>();

            sub_row(att_block, att_block, d_reg);
            mul(temp_block, temp_block, att_block);
            
            if constexpr (D == 64) { mul(temp_block, temp_block, 0.125f); }
            else                   { mul(temp_block, temp_block, 0.08838834764f); }
            
            copy(att_block_mma, temp_block);

            warpgroup::mma_async_wait(); 
            
            warpgroup::store(att_smem[warpgroupid], att_block_mma);

            warpgroup::mma_fence(qg_reg);
            warpgroup::mm_AB(qg_reg, att_block_mma, k_smem[warpgroupid]);
            warpgroup::mma_commit_group(); 

            asm volatile("bar.sync %0, 128;\n" :: "r"(warpgroupid));

            warpgroup::mma_fence(kg_reg);
            warpgroup::mma_AtB(kg_reg, att_smem[warpgroupid], q_smem[tic]);
            warpgroup::mma_commit_group();

            if (qo_idx > 0) tma::store_async_wait();
            warpgroup::mma_async_wait(); 

            asm volatile("bar.sync 10, 256;\n");
            warpgroup::store(qg_smem[warpgroupid], qg_reg);
            asm volatile("bar.sync 10, 256;\n");

            if (warpgroup::laneid() == 0) arrive(compute_done[tic]); 
            wait(compute_done[tic], ((qo_idx - q_start)/(2))%2);
        
            if ((warpid % 4 == 0) && (warpgroupid == 0)) {
                for (int w = 0; w < BWD_CONSUMER_WARPGROUPS; w++) {
                    int tile_idx = (blockIdx.y * qo_blocks) + (qo_idx) + (w/BWD_CONSUMER_WARPGROUPS);
                    tma::store_add_async(tma_qg, qg_smem[w], tile_idx); 
                    tma::store_commit_group();
                }
            }
        }

        tma::store_async_wait();
        asm volatile("bar.sync 10, 256;\n");

        if (warpgroupid == 0) {
            warpgroup::store(*kg_smem, kg_reg);
            warpgroup::store(*vg_smem, vg_reg);
        }
        else if (warpgroupid == 1) {
            warpgroup::store(qg_smem[0], kg_reg);
            warpgroup::store(qg_smem[1], vg_reg);
        }

        asm volatile("bar.sync 10, 256;\n");

        if (warpid == 0) {
            int tile_idx = ((blockIdx.y/heads_ratio) * BWD_CONSUMER_WARPGROUPS * gridDim.x) + (blockIdx.x * BWD_CONSUMER_WARPGROUPS); 

            tma::store_add_async(tma_kg, *kg_smem, tile_idx);
            tma::store_add_async(tma_vg, *vg_smem, tile_idx);
            tma::store_commit_group();
            
            tma::store_add_async(tma_kg, qg_smem[0], tile_idx+1);
            tma::store_add_async(tma_vg, qg_smem[1], tile_idx+1);
            tma::store_commit_group();
        }

        tma::store_async_wait();
    }
}

// #include "harness.impl"

#include "include/common/pyutils/torch_helpers.cuh"
#include <iostream>
void attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l, bool causal)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(l);
    CHECK_INPUT(o);

    auto batch    = q.size(0);
    auto seq_len  = q.size(2); 
    auto head_dim = q.size(3); 

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(l.size(0) == batch, "L batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(o.size(0) == batch, "O batch dimension - idx 0 - must match for all inputs");
    
    TORCH_CHECK(q.size(2) == seq_len, "Q sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(k.size(2) == seq_len, "K sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(v.size(2) == seq_len, "V sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(l.size(2) == seq_len, "L sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(o.size(2) == seq_len, "O sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(o.size(3) == head_dim, "O head dimension - idx 3 - must match for all non-vector inputs");

    // check if GQA
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    TORCH_CHECK(k.size(1) == v.size(1), "k and v must have the same number of heads");
    TORCH_CHECK(q.size(1) == o.size(1), "q and o must have the same number of heads");

    TORCH_CHECK(qo_heads >= kv_heads, "qo_heads must be greater than or equal to kv_heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "qo_heads must be divisible by kv_heads");

    auto heads_ratio = qo_heads / kv_heads;
    auto is_causal = causal; 

    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();
    c10::BFloat16* o_ptr = o.data_ptr<c10::BFloat16>();
    float         *l_ptr = l.data_ptr<float>();

    bf16* d_q  = reinterpret_cast<bf16*>(q_ptr);
    bf16* d_k  = reinterpret_cast<bf16*>(k_ptr);
    bf16* d_v  = reinterpret_cast<bf16*>(v_ptr);
    bf16* d_o  = reinterpret_cast<bf16*>(o_ptr);
    float* d_l = reinterpret_cast<float*>(l_ptr);

    CUtensorMap* tma_q_d; 
    CUtensorMap* tma_k_d; 
    CUtensorMap* tma_v_d; 
    CUtensorMap* tma_o_d; 
    CUtensorMap* tma_l_d; 

    if (head_dim == 64) {
        tma_q_d = tma::allocate_and_create_tensor_map<        st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width> >(d_q, batch*qo_heads*seq_len/(fwd_attend_ker_tile_dims<64>::qo_height * kittens::TILE_DIM));
        tma_k_d = tma::allocate_and_create_tensor_map<        st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width> >(d_k, batch*kv_heads*seq_len/(fwd_attend_ker_tile_dims<64>::kv_height * kittens::TILE_DIM));
        tma_v_d = tma::allocate_and_create_tensor_map<        st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width> >(d_v, batch*kv_heads*seq_len/(fwd_attend_ker_tile_dims<64>::kv_height * kittens::TILE_DIM));
        tma_o_d = tma::allocate_and_create_tensor_map<        st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width> >(d_o, batch*qo_heads*seq_len/(fwd_attend_ker_tile_dims<64>::qo_height * kittens::TILE_DIM));
        tma_l_d = tma::allocate_and_create_tensor_map<col_vec<st_fl<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width>>>(d_l, batch*qo_heads*seq_len/(fwd_attend_ker_tile_dims<64>::qo_height * kittens::TILE_DIM));
    }

    if (head_dim == 128) {
        tma_q_d = tma::allocate_and_create_tensor_map<        st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width> >(d_q, batch*qo_heads*seq_len/(fwd_attend_ker_tile_dims<128>::qo_height * kittens::TILE_DIM));
        tma_k_d = tma::allocate_and_create_tensor_map<        st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width> >(d_k, batch*kv_heads*seq_len/(fwd_attend_ker_tile_dims<128>::kv_height * kittens::TILE_DIM));
        tma_v_d = tma::allocate_and_create_tensor_map<        st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width> >(d_v, batch*kv_heads*seq_len/(fwd_attend_ker_tile_dims<128>::kv_height * kittens::TILE_DIM));
        tma_o_d = tma::allocate_and_create_tensor_map<        st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width> >(d_o, batch*qo_heads*seq_len/(fwd_attend_ker_tile_dims<128>::qo_height * kittens::TILE_DIM));
        tma_l_d = tma::allocate_and_create_tensor_map<col_vec<st_fl<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>>>(d_l, batch*qo_heads*seq_len/(fwd_attend_ker_tile_dims<128>::qo_height * kittens::TILE_DIM));
    }

    auto mem_size = kittens::MAX_SHARED_MEMORY;
    auto threads  = NUM_WORKERS * kittens::WARP_THREADS;

    TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 192");
    dim3 grid(seq_len/(CONSUMER_WARPGROUPS*kittens::TILE_DIM*4), batch*qo_heads, 1);

    if (is_causal && head_dim == 64) {
        cudaFuncSetAttribute(
            fwd_attend_ker<64, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        fwd_attend_ker<64, true><<<grid, threads, mem_size>>>(seq_len, heads_ratio, tma_q_d, tma_k_d, tma_v_d, tma_o_d, tma_l_d);
    }

    if (is_causal && head_dim == 128) {
        cudaFuncSetAttribute(
            fwd_attend_ker<128, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        fwd_attend_ker<128, true><<<grid, threads, mem_size>>>(seq_len, heads_ratio, tma_q_d, tma_k_d, tma_v_d, tma_o_d, tma_l_d);
    }

    if (!is_causal && head_dim == 64) {
        cudaFuncSetAttribute(
            fwd_attend_ker<64, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        fwd_attend_ker<64, false><<<grid, threads, mem_size>>>(seq_len, heads_ratio, tma_q_d, tma_k_d, tma_v_d, tma_o_d, tma_l_d);
    }

    if (!is_causal && head_dim == 128) {
        cudaFuncSetAttribute(
            fwd_attend_ker<128, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        fwd_attend_ker<128, false><<<grid, threads, mem_size>>>(seq_len, heads_ratio, tma_q_d, tma_k_d, tma_v_d, tma_o_d, tma_l_d);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

void attention_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor l_vec, torch::Tensor d_vec, torch::Tensor og, torch::Tensor qg, torch::Tensor kg, torch::Tensor vg, bool causal)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(l_vec);
    CHECK_INPUT(d_vec);
    CHECK_INPUT(o);
    CHECK_INPUT(og);
    CHECK_INPUT(qg);
    CHECK_INPUT(kg);
    CHECK_INPUT(vg);

    auto batch    = q.size(0);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(l_vec.size(0) == batch, "L batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(d_vec.size(0) == batch, "D batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(o.size(0) == batch, "O batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(og.size(0) == batch, "OG batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(qg.size(0) == batch, "QG batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(kg.size(0) == batch, "KG batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(vg.size(0) == batch, "VG batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(2) == seq_len, "Q sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(k.size(2) == seq_len, "K sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(v.size(2) == seq_len, "V sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(l_vec.size(2) == seq_len, "L sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(d_vec.size(2) == seq_len, "D sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(o.size(2) == seq_len, "O sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(og.size(2) == seq_len, "OG sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(qg.size(2) == seq_len, "QG sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(kg.size(2) == seq_len, "KG sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(vg.size(2) == seq_len, "VG sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(o.size(3) == head_dim, "O head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(og.size(3) == head_dim, "OG head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(qg.size(3) == head_dim, "QG head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(kg.size(3) == head_dim, "KG head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(vg.size(3) == head_dim, "VG head dimension - idx 3 - must match for all non-vector inputs");

    // check if GQA
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    TORCH_CHECK(k.size(1) == v.size(1), "k and v must have the same number of heads");
    TORCH_CHECK(q.size(1) == o.size(1), "q and o must have the same number of heads");

    TORCH_CHECK(qo_heads >= kv_heads, "qo_heads must be greater than or equal to kv_heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "qo_heads must be divisible by kv_heads");

    auto heads_ratio = qo_heads / kv_heads;

    // check if causal
    auto is_causal = causal;

    c10::BFloat16* q_ptr  = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr  = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr  = v.data_ptr<c10::BFloat16>();
    c10::BFloat16* o_ptr  = o.data_ptr<c10::BFloat16>();
    c10::BFloat16* og_ptr = og.data_ptr<c10::BFloat16>();
    float*         l_ptr  = l_vec.data_ptr<float>();
    float*         d_ptr  = d_vec.data_ptr<float>();
    float*         qg_ptr = qg.data_ptr<float>();
    float*         kg_ptr = kg.data_ptr<float>();
    float*         vg_ptr = vg.data_ptr<float>();

    bf16*  d_q  = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k  = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v  = reinterpret_cast<bf16*>(v_ptr);
    bf16*  d_o  = reinterpret_cast<bf16*>(o_ptr);
    bf16*  d_og = reinterpret_cast<bf16*>(og_ptr);
    float* d_l  = reinterpret_cast<float*>(l_ptr);
    float* d_d  = reinterpret_cast<float*>(d_ptr);
    float* d_qg = reinterpret_cast<float*>(qg_ptr);
    float* d_kg = reinterpret_cast<float*>(kg_ptr);
    float* d_vg = reinterpret_cast<float*>(vg_ptr);

    CUtensorMap* tma_q_b_d;
    CUtensorMap* tma_k_b_d;
    CUtensorMap* tma_v_b_d;
    CUtensorMap* tma_o_d;
    CUtensorMap* tma_og_d;
    CUtensorMap* tma_l_b_d;
    CUtensorMap* tma_d_b_d;
    CUtensorMap* tma_qg_d;
    CUtensorMap* tma_kg_d;
    CUtensorMap* tma_vg_d;

    CUtensorMap* tma_prep_o_d; 
    CUtensorMap* tma_prep_og_d;
    CUtensorMap* tma_prep_d_d;

    auto mem_size = kittens::MAX_SHARED_MEMORY; 
    auto threads  = 4 * kittens::WARP_THREADS;

    TORCH_CHECK(seq_len % (4*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 256");
    dim3 grid_bwd(seq_len/(4*kittens::TILE_DIM*4), batch*qo_heads, 1);

    if (head_dim == 64) {
        tma_prep_o_d  = tma::allocate_and_create_tensor_map<        st_bf<4, 64/kittens::TILE_DIM> >(d_o,  (batch*qo_heads*seq_len)/(4 * kittens::TILE_DIM));
        tma_prep_og_d = tma::allocate_and_create_tensor_map<        st_bf<4, 64/kittens::TILE_DIM> >(d_og, (batch*qo_heads*seq_len)/(4 * kittens::TILE_DIM));
        tma_prep_d_d  = tma::allocate_and_create_tensor_map<col_vec<st_fl<4, 64/kittens::TILE_DIM>>>(d_d,  (batch*qo_heads*seq_len)/(4 * kittens::TILE_DIM));

        cudaFuncSetAttribute(
            bwd_attend_prep_ker<64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        bwd_attend_prep_ker<64><<<grid_bwd, threads, mem_size>>>(tma_prep_o_d, tma_prep_d_d, tma_prep_og_d); 

        tma_q_b_d = tma::allocate_and_create_tensor_map<        st_bf<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>> (d_q,  (batch * qo_heads * seq_len)/(bwd_attend_ker_tile_dims<64>::tile_h_qo * kittens::TILE_DIM)); 
        tma_k_b_d = tma::allocate_and_create_tensor_map<        st_bf<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>> (d_k,  (batch * kv_heads * seq_len)/(bwd_attend_ker_tile_dims<64>::tile_h    * kittens::TILE_DIM));
        tma_v_b_d = tma::allocate_and_create_tensor_map<        st_bf<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>> (d_v,  (batch * kv_heads * seq_len)/(bwd_attend_ker_tile_dims<64>::tile_h    * kittens::TILE_DIM));
        tma_qg_d  = tma::allocate_and_create_tensor_map<        st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>> (d_qg, (batch * qo_heads * seq_len)/(bwd_attend_ker_tile_dims<64>::tile_h_qo * kittens::TILE_DIM));
        tma_kg_d  = tma::allocate_and_create_tensor_map<        st_fl<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>> (d_kg, (batch * kv_heads * seq_len)/(bwd_attend_ker_tile_dims<64>::tile_h    * kittens::TILE_DIM));
        tma_vg_d  = tma::allocate_and_create_tensor_map<        st_fl<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>> (d_vg, (batch * kv_heads * seq_len)/(bwd_attend_ker_tile_dims<64>::tile_h    * kittens::TILE_DIM));
        tma_og_d  = tma::allocate_and_create_tensor_map<        st_bf<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>> (d_og, (batch * qo_heads * seq_len)/(bwd_attend_ker_tile_dims<64>::tile_h_qo * kittens::TILE_DIM));
        tma_l_b_d = tma::allocate_and_create_tensor_map<col_vec<st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>>>(d_l,  (batch * qo_heads * seq_len)/(bwd_attend_ker_tile_dims<64>::tile_h_qo * kittens::TILE_DIM));
        tma_d_b_d = tma::allocate_and_create_tensor_map<col_vec<st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>>>(d_d,  (batch * qo_heads * seq_len)/(bwd_attend_ker_tile_dims<64>::tile_h_qo * kittens::TILE_DIM));
    }

    if (head_dim == 128) {
        tma_prep_o_d  = tma::allocate_and_create_tensor_map<        st_bf<4, 128/kittens::TILE_DIM> >(d_o,  (batch*qo_heads*seq_len)/(4 * kittens::TILE_DIM));
        tma_prep_og_d = tma::allocate_and_create_tensor_map<        st_bf<4, 128/kittens::TILE_DIM> >(d_og, (batch*qo_heads*seq_len)/(4 * kittens::TILE_DIM));
        tma_prep_d_d  = tma::allocate_and_create_tensor_map<col_vec<st_fl<4, 128/kittens::TILE_DIM>>>(d_d,  (batch*qo_heads*seq_len)/(4 * kittens::TILE_DIM));

        cudaFuncSetAttribute(
            bwd_attend_prep_ker<128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        bwd_attend_prep_ker<128><<<grid_bwd, threads, mem_size>>>(tma_prep_o_d, tma_prep_d_d, tma_prep_og_d);

        tma_q_b_d = tma::allocate_and_create_tensor_map<        st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>> (d_q,  (batch * qo_heads * seq_len)/(bwd_attend_ker_tile_dims<128>::tile_h_qo * kittens::TILE_DIM)); 
        tma_k_b_d = tma::allocate_and_create_tensor_map<        st_bf<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>> (d_k,  (batch * kv_heads * seq_len)/(bwd_attend_ker_tile_dims<128>::tile_h    * kittens::TILE_DIM));
        tma_v_b_d = tma::allocate_and_create_tensor_map<        st_bf<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>> (d_v,  (batch * kv_heads * seq_len)/(bwd_attend_ker_tile_dims<128>::tile_h    * kittens::TILE_DIM));
        tma_qg_d  = tma::allocate_and_create_tensor_map<        st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>> (d_qg, (batch * qo_heads * seq_len)/(bwd_attend_ker_tile_dims<128>::tile_h_qo * kittens::TILE_DIM));
        tma_kg_d  = tma::allocate_and_create_tensor_map<        st_fl<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>> (d_kg, (batch * kv_heads * seq_len)/(bwd_attend_ker_tile_dims<128>::tile_h    * kittens::TILE_DIM));
        tma_vg_d  = tma::allocate_and_create_tensor_map<        st_fl<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>> (d_vg, (batch * kv_heads * seq_len)/(bwd_attend_ker_tile_dims<128>::tile_h    * kittens::TILE_DIM));
        tma_og_d  = tma::allocate_and_create_tensor_map<        st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>> (d_og, (batch * qo_heads * seq_len)/(bwd_attend_ker_tile_dims<128>::tile_h_qo * kittens::TILE_DIM));
        tma_l_b_d = tma::allocate_and_create_tensor_map<col_vec<st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>>>(d_l,  (batch * qo_heads * seq_len)/(bwd_attend_ker_tile_dims<128>::tile_h_qo * kittens::TILE_DIM));
        tma_d_b_d = tma::allocate_and_create_tensor_map<col_vec<st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>>>(d_d,  (batch * qo_heads * seq_len)/(bwd_attend_ker_tile_dims<128>::tile_h_qo * kittens::TILE_DIM));
    }

    TORCH_CHECK(seq_len % (4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_DIM) == 0, "sequence length must be divisible by 128");
    dim3 grid_bwd_2(seq_len/(4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_DIM), batch*qo_heads, 1);

    threads = kittens::WARP_THREADS * BWD_NUM_WORKERS;

    if (is_causal && head_dim == 64) {

        mem_size = kittens::MAX_SHARED_MEMORY / bwd_attend_ker_tile_dims<64>::blocks_sm;

        cudaFuncSetAttribute(
            bwd_attend_ker<64, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        bwd_attend_ker<64, true><<<grid_bwd_2, threads, mem_size>>>(seq_len, heads_ratio, tma_q_b_d, tma_k_b_d, tma_v_b_d, tma_l_b_d, tma_d_b_d, tma_og_d, tma_qg_d, tma_kg_d, tma_vg_d);
    }

    if (is_causal && head_dim == 128) {

        mem_size = kittens::MAX_SHARED_MEMORY / bwd_attend_ker_tile_dims<128>::blocks_sm;

        cudaFuncSetAttribute(
            bwd_attend_ker<128, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        bwd_attend_ker<128, true><<<grid_bwd_2, threads, mem_size>>>(seq_len, heads_ratio, tma_q_b_d, tma_k_b_d, tma_v_b_d, tma_l_b_d, tma_d_b_d, tma_og_d, tma_qg_d, tma_kg_d, tma_vg_d);
    }

    if (!is_causal && head_dim == 64) {

        mem_size = kittens::MAX_SHARED_MEMORY / bwd_attend_ker_tile_dims<64>::blocks_sm;

        cudaFuncSetAttribute(
            bwd_attend_ker<64, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        bwd_attend_ker<64, false><<<grid_bwd_2, threads, mem_size>>>(seq_len, heads_ratio, tma_q_b_d, tma_k_b_d, tma_v_b_d, tma_l_b_d, tma_d_b_d, tma_og_d, tma_qg_d, tma_kg_d, tma_vg_d);
    }

    if (!is_causal && head_dim == 128) {

        mem_size = kittens::MAX_SHARED_MEMORY / bwd_attend_ker_tile_dims<128>::blocks_sm;

        cudaFuncSetAttribute(
            bwd_attend_ker<128, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        bwd_attend_ker<128, false><<<grid_bwd_2, threads, mem_size>>>(seq_len, heads_ratio, tma_q_b_d, tma_k_b_d, tma_v_b_d, tma_l_b_d, tma_d_b_d, tma_og_d, tma_qg_d, tma_kg_d, tma_vg_d);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}