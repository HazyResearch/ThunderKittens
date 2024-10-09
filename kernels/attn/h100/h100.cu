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
    constexpr static int tile_width = (64);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (8*16);
    constexpr static int stages     = (4); 
};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (8*16);
    constexpr static int stages     = (2); 
};

template<int D> struct fwd_globals {
    using q_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using k_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using v_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>>;
    using o_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;

    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    const int N; 
    const int hr;
};

template<int D, bool is_causal>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile    =         st_bf<K::qo_height, K::tile_width>;
    using k_tile    =         st_bf<K::kv_height, K::tile_width>;
    using v_tile    =         st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile    =         st_bf<K::qo_height, K::tile_width>;
    
    q_tile    (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[K::stages]           = al.allocate<k_tile, K::stages          >();
    v_tile    (&v_smem)[K::stages]           = al.allocate<v_tile, K::stages          >();
    l_col_vec (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, CONSUMER_WARPGROUPS>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);
    
    int kv_blocks = g.N / (K::kv_height);

    __shared__ kittens::barrier qsmem_barrier, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    if (threadIdx.x == 0) { 
        init_barrier(qsmem_barrier, 0, 1); 
        for(int j = 0; j < K::stages; j++) {
            init_barrier(k_smem_arrived[j], 0, 1); 
            init_barrier(v_smem_arrived[j], 0, 1); 
            init_barrier(compute_done[j], CONSUMER_WARPGROUPS, 0); 
        }
        
        tma::expect_bytes(qsmem_barrier, sizeof(q_smem));
        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            int4 q_tile_idx = {blockIdx.y / g.q.depth, blockIdx.y % g.q.depth, (blockIdx.x * CONSUMER_WARPGROUPS) + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_barrier);
        }

        for (int j = 0; j < K::stages - 1; j++) {
            int4 kv_tile_idx = {blockIdx.y / (g.q.depth), (blockIdx.y % g.q.depth)/(g.hr), j, 0};
            tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
            tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));

            tma::load_async(k_smem[j], g.k, kv_tile_idx, k_smem_arrived[j]);
            tma::load_async(v_smem[j], g.v, kv_tile_idx, v_smem_arrived[j]);
        }
    }

    __syncthreads(); 

    int pipe_idx = K::stages - 1; 
    
    if(warpgroupid == NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();      
        
        int kv_iters; 
        if constexpr (is_causal) {
            kv_iters = (blockIdx.x * CONSUMER_WARPGROUPS * (K::qo_height/kittens::TILE_DIM)) - 1 + (CONSUMER_WARPGROUPS * (K::qo_height/kittens::TILE_DIM)); 
            kv_iters = ((kv_iters / (K::kv_height/kittens::TILE_DIM)) == 0) ? (0) : ((kv_iters / (K::kv_height/kittens::TILE_DIM)) - 1);
        }
        else {
            kv_iters = kv_blocks-2; 
        }

        if(warpid == NUM_WORKERS-4) {
            for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                int4 kv_tile_idx = {blockIdx.y / (g.q.depth), (blockIdx.y % g.q.depth)/(g.hr), kv_idx + 1, 0};
                tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
                tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));

                tma::load_async(k_smem[(kv_idx+1)%K::stages], g.k, kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
                tma::load_async(v_smem[(kv_idx+1)%K::stages], g.v, kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);

                wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            }
        }
    }
    else {
        warpgroup::increase_registers<160>();

        wait(qsmem_barrier, 0);
        if constexpr (D == 64) { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f)); }
        else                   { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.08838834764f)); }
    
        rt_fl<16, K::kv_height>  att_block; 
        rt_fl<16, K::kv_height>  att_block_scaled;
        rt_bf<16, K::kv_height>  att_block_mma;
        rt_fl<16, K::tile_width> o_reg;
        
        col_vec<rt_fl<16, K::kv_height>> max_vec_last,        max_vec;
        col_vec<rt_fl<16, K::kv_height>> max_vec_last_scaled, max_vec_scaled;
        col_vec<rt_fl<16, K::kv_height>> norm_vec_last,       norm_vec;
        
        neg_infty(max_vec);
        zero(norm_vec);
        zero(o_reg);

        int kv_iters; 
        if constexpr (is_causal) {
            kv_iters = (blockIdx.x * CONSUMER_WARPGROUPS * (K::qo_height/kittens::TILE_DIM)) - 1 + ((warpgroupid + 1) * (K::qo_height/kittens::TILE_DIM));
            kv_iters = (kv_iters/(K::kv_height/kittens::TILE_DIM));
        }
        else { 
            kv_iters = kv_blocks - 1; 
        }
        
        const int kv_do = (blockIdx.x * CONSUMER_WARPGROUPS)/(K::kv_height/K::qo_height);

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
        
            wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            
            warpgroup::mma_fence(att_block);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(kv_idx)%K::stages]);
            warpgroup::mma_commit_group();
            
            copy(norm_vec_last, norm_vec);
            copy(max_vec_last,  max_vec);
            
            warpgroup::mma_async_wait();

            if constexpr (is_causal) {
                const int q_blk = (blockIdx.x * CONSUMER_WARPGROUPS * (K::qo_height/kittens::TILE_DIM)) + warpid; 
                      int k_blk = (kv_idx * (K::kv_height/kittens::TILE_DIM)); 

                #pragma unroll
                for(int _ = 0; k_blk == (kv_iters-1)*(K::kv_height/kittens::TILE_DIM) || k_blk == (kv_iters)*(K::kv_height/kittens::TILE_DIM); k_blk+=10000) {
                    #pragma unroll
                    for (auto j = 0; j < (K::kv_height/kittens::TILE_DIM); j++) {
                        auto k_idx = k_blk + j;
                        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(att_block.tiles[0][j]);

                        if      (k_idx >  q_blk) { neg_infty  (attn_subtile); }
                        else if (k_idx == q_blk) { make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
                    }
                }
            }

            row_max(max_vec, att_block, max_vec);
            
            mul(att_block_scaled, att_block, 1.44269504089f);
            mul(max_vec_scaled,   max_vec,   1.44269504089f);     
            sub_row(att_block_scaled, att_block_scaled, max_vec_scaled);
            exp2(att_block, att_block_scaled);

            mul(max_vec_last_scaled, max_vec_last, 1.44269504089f);
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last,       max_vec_last_scaled);
            mul(norm_vec,            norm_vec,     max_vec_last);

            row_sum(norm_vec,  att_block, norm_vec);
            div_row(att_block, att_block, norm_vec);
            
            mul(norm_vec_last, norm_vec_last, max_vec_last);
            div(norm_vec_last, norm_vec_last, norm_vec);
            
            copy(att_block_mma, att_block); 
            mul_row(o_reg, o_reg, norm_vec_last); 

            wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2); 

            warpgroup::mma_fence(o_reg);
            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(kv_idx)%K::stages]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            if(warpid % 4 == 0) {
                int count = ((warpgroupid == CONSUMER_WARPGROUPS - 1) && (kv_idx > kv_do) && is_causal) ? (2 - (blockIdx.x % 2)) : 1;
                arrive(compute_done[(kv_idx)%K::stages], count); 
            }
        }

        warpgroup::store(o_smem[warpgroupid], o_reg); 
        warpgroup::sync();

        if (warpid % 4 == 0) {
            int4 o_tile_idx = {blockIdx.y / g.o.depth, blockIdx.y % g.o.depth, (blockIdx.x * CONSUMER_WARPGROUPS) + warpgroupid, 0};
            tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
            tma::store_commit_group();
        }

        log(norm_vec, norm_vec);
        add(norm_vec, norm_vec, max_vec);

        if constexpr (D == 64) { mul(norm_vec, norm_vec, -8.0f); }
        else                   { mul(norm_vec, norm_vec, -11.3137085f); }

        warpgroup::sync(); 
    
        warpgroup::store(l_smem[warpgroupid], norm_vec);
        warpgroup::sync();
        
        if (warpid % 4 == 0) {
            int4 tile_idx = {blockIdx.y / g.l.depth, blockIdx.y % g.l.depth, 0, (blockIdx.x * CONSUMER_WARPGROUPS) + warpgroupid};
            tma::store_async(g.l, l_smem[warpgroupid], tile_idx);
            tma::store_commit_group();
        }
    
        tma::store_async_wait();
    }
}

template<int D>
struct bwd_prep_globals {
    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;

    og_gl og;
    o_gl  o;
    d_gl  d;
};

template<int D>
__global__  __launch_bounds__(4*kittens::WARP_THREADS, (D == 64) ? 2 : 1)
void bwd_attend_prep_ker(const __grid_constant__ bwd_prep_globals<D> g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();

    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    og_tile (&og_smem)[4] = al.allocate<og_tile, 4>();
    o_tile  (&o_smem) [4] = al.allocate<o_tile , 4>();
    d_tile  (&d_smem) [4] = al.allocate<d_tile , 4>();

    using fl_reg_tile = rt_fl<4*16, D>;
    using fl_reg_col  = col_vec<rt_fl<4*16, D>>;
    
    fl_reg_tile og_reg;
    fl_reg_tile  o_reg; 
    fl_reg_col   d_reg;

    __shared__ kittens::barrier smem_barrier;

    if (threadIdx.x == 0) {
        init_barrier(smem_barrier, 0, 1);
        tma::expect_bytes(smem_barrier, sizeof(og_smem[0]) * 4 * 2);
    }
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < 4; w++) { // load o, o_grad
            int4 tile_idx = {blockIdx.y / g.o.depth, blockIdx.y % g.o.depth, (blockIdx.x * 4) + w, 0};
            tma::load_async(o_smem[w],  g.o,  tile_idx, smem_barrier);
            tma::load_async(og_smem[w], g.og, tile_idx, smem_barrier);
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
            int4 tile_idx = {blockIdx.y / g.d.depth, blockIdx.y % g.d.depth, 0, (blockIdx.x * 4) + w};
            tma::store_async(g.d, d_smem[w], tile_idx);
        }
        tma::store_commit_group();
    }

    tma::store_async_wait();
}

template<int D> struct bwd_attend_ker_tile_dims {};
template<> struct bwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int tile_h     = (4*16);
    constexpr static int tile_h_qo  = (4*16);

    constexpr static int blocks_sm = 1;
};
template<> struct bwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int tile_h     = (4*16);
    constexpr static int tile_h_qo  = (4*16);

    constexpr static int blocks_sm = 1; 
};

constexpr int BWD_CONSUMER_WARPGROUPS = (2); 
constexpr int BWD_PRODUCER_WARPGROUPS = (1); 
constexpr int BWD_NUM_WARPGROUPS      = (BWD_CONSUMER_WARPGROUPS+BWD_PRODUCER_WARPGROUPS); 
constexpr int BWD_NUM_WORKERS         = (BWD_NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 

template<int D>
struct bwd_globals {
    using G = bwd_attend_ker_tile_dims<D>;

    using q_tile  =         st_bf<G::tile_h_qo, G::tile_width>;
    using k_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using v_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using og_tile =         st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile =         st_fl<G::tile_h_qo, G::tile_width>;
    using kg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using vg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using l_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

    using q_gl  = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl  = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl  = gl<bf16,  -1, -1, -1, -1, v_tile>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;

    using qg_gl = gl<float, -1, -1, -1, -1, qg_tile>;
    using kg_gl = gl<float, -1, -1, -1, -1, kg_tile>;
    using vg_gl = gl<float, -1, -1, -1, -1, vg_tile>;

    using l_gl  = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;

    q_gl  q;
    k_gl  k;
    v_gl  v;
    og_gl og;
    qg_gl qg;
    kg_gl kg;
    vg_gl vg;
    l_gl  l;
    d_gl  d;

    const int N;
    const int hr;
};

template<int D, bool is_causal>
__global__ __launch_bounds__(BWD_NUM_WORKERS*kittens::WARP_THREADS, bwd_attend_ker_tile_dims<D>::blocks_sm)
void bwd_attend_ker(const __grid_constant__ bwd_globals<D> g) {
                        
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int N = g.N, hr = g.hr;

    using G = bwd_attend_ker_tile_dims<D>;
    
    using kg_tile = st_fl<G::tile_h, G::tile_width>;
    using vg_tile = st_fl<G::tile_h, G::tile_width>;

    using k_tile = st_bf<G::tile_h, G::tile_width>;
    using v_tile = st_bf<G::tile_h, G::tile_width>;

    using q_tile  = st_bf<G::tile_h_qo, G::tile_width>;
    using og_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile = st_fl<G::tile_h_qo, G::tile_width>;

    using l_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

    using attn_tile = st_bf<G::tile_h_qo, G::tile_h>; 

    k_tile  (&k_smem) [BWD_CONSUMER_WARPGROUPS] = al.allocate<k_tile, BWD_CONSUMER_WARPGROUPS>();
    v_tile  (&v_smem) [BWD_CONSUMER_WARPGROUPS] = al.allocate<v_tile, BWD_CONSUMER_WARPGROUPS>();

    q_tile  (&q_smem) [2] = al.allocate<q_tile , 2>(); 
    og_tile (&og_smem)[2] = al.allocate<og_tile, 2>(); 
    qg_tile (&qg_smem)    = al.allocate<qg_tile>();

    l_tile   (&l_smem)[2] = al.allocate<l_tile, 2>();
    d_tile   (&d_smem)[2] = al.allocate<d_tile, 2>();
    kg_tile (*kg_smem)    = reinterpret_cast<kg_tile*>(&k_smem[0].data[0]); 
    vg_tile (*vg_smem)    = reinterpret_cast<vg_tile*>(&q_smem[0].data[0]); 

    attn_tile (&ds_smem)[BWD_CONSUMER_WARPGROUPS] = al.allocate<attn_tile, BWD_CONSUMER_WARPGROUPS>();

    int warpid = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    const int qo_blocks = N / (G::tile_h_qo);

    __shared__ kittens::barrier kv_b, q_b[2], o_b[2], vec_b[2];
    __shared__ kittens::barrier compute_done[2], qg_ready; 

    int tic = 0, toc = 1;
    const int q_start = (is_causal) ? (blockIdx.x) : (0);

    if (threadIdx.x == 0) {
        init_barrier(kv_b,  0, 1);
        init_barrier(qg_ready, 1, 0);
        
        for (int s = 0; s < 2; s++) {
            init_barrier(q_b[s],  0, 1);
            init_barrier(o_b[s],  0, 1); 
            init_barrier(vec_b[s], 0, 1);
            
            init_barrier(compute_done[s], 1, 0);
        }

        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * BWD_CONSUMER_WARPGROUPS);
        for (int w = 0; w < BWD_CONSUMER_WARPGROUPS; w++) {
            int4 tile_idx = {blockIdx.y / g.q.depth, (blockIdx.y % g.q.depth) / hr, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + w, 0};
            tma::load_async(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async(v_smem[w], g.v, tile_idx, kv_b);
        }

        tma::expect_bytes(q_b[tic],   sizeof(q_smem[0]));
        tma::expect_bytes(o_b[tic],   sizeof(og_smem[0]));
        tma::expect_bytes(vec_b[tic], sizeof(l_smem[0]) + sizeof(d_smem[0]));

        int4 tile_idx = {blockIdx.y / g.q.depth, blockIdx.y % g.q.depth, q_start, 0};
        tma::load_async(q_smem[tic],  g.q,  tile_idx, q_b[tic]);
        tma::load_async(og_smem[tic], g.og, tile_idx, o_b[tic]);

        int4 vec_idx = {blockIdx.y / g.q.depth, blockIdx.y % g.q.depth, 0, q_start};
        tma::load_async(l_smem[tic], g.l, vec_idx, vec_b[tic]);
        tma::load_async(d_smem[tic], g.d, vec_idx, vec_b[tic]);
    }

    __syncthreads(); 

    if (warpgroupid == BWD_NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<24>();

        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            for (auto qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                if (qo_idx + 1 < qo_blocks) {
                    tma::expect_bytes(q_b[toc],   sizeof(q_smem[0])); 
                    tma::expect_bytes(o_b[toc],   sizeof(og_smem[0]));
                    tma::expect_bytes(vec_b[toc], sizeof(l_smem[0]) + sizeof(d_smem[0]));
                    
                    int4 tile_idx = {blockIdx.y / g.q.depth, blockIdx.y % g.q.depth, qo_idx + 1, 0};
                    tma::load_async(q_smem[toc], g.q,  tile_idx, q_b[toc]);
                    tma::load_async(og_smem[toc], g.og, tile_idx, o_b[toc]);

                    int4 vec_idx = {blockIdx.y / g.q.depth, blockIdx.y % g.q.depth, 0, qo_idx + 1};
                    tma::load_async(l_smem[toc], g.l, vec_idx, vec_b[toc]);
                    tma::load_async(d_smem[toc], g.d, vec_idx, vec_b[toc]);
                }
                
                wait(compute_done[tic], ((qo_idx - q_start)/(2))%2);
            }
        }
        else if(warpid % WARPGROUP_WARPS == 1) {
            for (auto qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                wait(compute_done[tic], ((qo_idx - q_start)/(2))%2);
                
                int4 tile_idx = {blockIdx.y / g.q.depth, blockIdx.y % g.q.depth, qo_idx, 0};
                tma::store_add_async(g.qg, qg_smem, tile_idx);
                
                tma::store_commit_group();
                tma::store_async_wait();
                
                arrive(qg_ready);
            }
        }
    }
    else if(warpgroupid == 0) {
        warpgroup::increase_registers<256>();

        rt_fl<16, G::tile_width> kg_reg;
        rt_fl<16, G::tile_width> vg_reg;
        rt_fl<16, G::tile_width> qg_reg;

        row_vec<rt_fl<16, 4*16>> row_reg; 

        rt_fl<16, 4*16> s_block_t; 
        rt_fl<16, 4*16> dp_block_t;

        rt_fl<16, 4*16> p_block_t;
        rt_fl<16, 4*16> ds_block_t;

        rt_bf<16, 4*16> p_block_t_mma;
        rt_bf<16, 4*16> ds_block_t_mma;

        zero(kg_reg);
        zero(vg_reg);

        wait(kv_b, 0);

        for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {

            wait(vec_b[tic], ((qo_idx - q_start)/2)%2);

            #pragma unroll
            for(int i = 0; i < 4; i++) {
                int base_col = 16*i + 2*(laneid()%4);
                s_block_t.tiles[0][i].data[0] = *(float2*)&l_smem[tic][base_col + 0];
                s_block_t.tiles[0][i].data[1] = *(float2*)&l_smem[tic][base_col + 0];
                s_block_t.tiles[0][i].data[2] = *(float2*)&l_smem[tic][base_col + 8];
                s_block_t.tiles[0][i].data[3] = *(float2*)&l_smem[tic][base_col + 8];
            }

            wait(q_b[tic], ((qo_idx - q_start)/2)%2);

            warpgroup::mma_fence(s_block_t);
            warpgroup::mma_ABt(s_block_t, k_smem[warpgroupid], q_smem[tic]);
            warpgroup::mma_commit_group();

            wait(o_b[tic], ((qo_idx - q_start)/2)%2);

            warpgroup::mma_fence(dp_block_t);
            warpgroup::mm_ABt(dp_block_t, v_smem[warpgroupid], og_smem[tic]);
            warpgroup::mma_commit_group();

            warpgroup::mma_async_wait();

            if constexpr (D == 64) { mul(s_block_t, s_block_t, 1.44269504089f*0.125f); }
            else                   { mul(s_block_t, s_block_t, 1.44269504089f*0.08838834764f); }

            if constexpr (is_causal) {
                int q_blk = (qo_idx) * (G::tile_h_qo/kittens::TILE_DIM);
                int k_blk = (blockIdx.x * BWD_CONSUMER_WARPGROUPS * (G::tile_h/kittens::TILE_DIM)) + (warpgroupid * (G::tile_h/kittens::TILE_DIM)) + (warpid % kittens::WARPGROUP_WARPS);

                for (int j = 0; j < (G::tile_h_qo/kittens::TILE_DIM); j++) {
                    int q_idx = q_blk + j;
                    auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(s_block_t.tiles[0][j]);

                    if      (q_idx <  k_blk) { neg_infty(attn_subtile); }
                    else if (q_idx == k_blk) { make_causal_t(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
                }
            }

            exp2(s_block_t, s_block_t);
            copy(p_block_t, s_block_t);
            copy(p_block_t_mma, s_block_t);

            #pragma unroll
            for(int i = 0; i < 4; i++) {
                int base_col = 16*i + 2*(laneid()%4);
                dp_block_t.tiles[0][i].data[0] = base_ops::sub::template op<float2>(dp_block_t.tiles[0][i].data[0], *(float2*)&d_smem[tic][base_col + 0]);
                dp_block_t.tiles[0][i].data[1] = base_ops::sub::template op<float2>(dp_block_t.tiles[0][i].data[1], *(float2*)&d_smem[tic][base_col + 0]);
                dp_block_t.tiles[0][i].data[2] = base_ops::sub::template op<float2>(dp_block_t.tiles[0][i].data[2], *(float2*)&d_smem[tic][base_col + 8]);
                dp_block_t.tiles[0][i].data[3] = base_ops::sub::template op<float2>(dp_block_t.tiles[0][i].data[3], *(float2*)&d_smem[tic][base_col + 8]);
            }
            mul(ds_block_t, p_block_t, dp_block_t);

            if constexpr (D == 64) { mul(ds_block_t, ds_block_t, 0.125f); }
            else                   { mul(ds_block_t, ds_block_t, 0.08838834764f); }

            warpgroup::mma_fence(vg_reg);
            warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[tic]);
            warpgroup::mma_commit_group();

            copy(ds_block_t_mma, ds_block_t);
            
            warpgroup::store(ds_smem[warpgroupid], ds_block_t);

            warpgroup::mma_fence(kg_reg);
            warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[tic]);
            warpgroup::mma_commit_group();
            
            warpgroup::mma_async_wait();
            asm volatile("bar.sync 10, 256;\n");

            warpgroup::mma_fence(qg_reg);
            warpgroup::mm_AtB(qg_reg, ds_smem[0], k_smem[0]);
            warpgroup::mma_AtB(qg_reg, ds_smem[1], k_smem[1]);
            warpgroup::mma_commit_group(); 

            wait(qg_ready, toc);
            if (qo_idx > 0) tma::store_async_wait();

            warpgroup::mma_async_wait();
            warpgroup::store(qg_smem, qg_reg);

            asm volatile("bar.sync %0, 128;\n" :: "r"(warpgroup::groupid()+4));

            if(warpid % 4 == 0) arrive(compute_done[tic]);
        }

        asm volatile("bar.sync 10, 256;\n");

        if (warpgroupid == 0) warpgroup::store(kg_smem[0], kg_reg);
        if (warpgroupid == 1) warpgroup::store(kg_smem[1], kg_reg);

        asm volatile("bar.sync %0, 128;\n" :: "r"(warpgroup::groupid()+4));

        if (warpid == 0) {
            int4 tile_idx = {blockIdx.y / g.q.depth, (blockIdx.y % g.q.depth) / hr, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + 0, 0};
            tma::store_add_async(g.kg, kg_smem[0], tile_idx);
            tma::store_commit_group();
        }
        else if (warpid == 4) {
            int4 tile_idx = {blockIdx.y / g.q.depth, (blockIdx.y % g.q.depth) / hr, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + 1, 0};
            tma::store_add_async(g.kg, kg_smem[1], tile_idx);
            tma::store_commit_group();
        }

        wait(qg_ready, toc);
        if(warpgroupid == 0) warpgroup::store(vg_smem[0], vg_reg);
        if(warpgroupid == 1) warpgroup::store(vg_smem[1], vg_reg);
        
        asm volatile("bar.sync %0, 128;\n" :: "r"(warpgroup::groupid()+4));

        if (warpid == 0) {
            int4 tile_idx = {blockIdx.y / g.q.depth, (blockIdx.y % g.q.depth) / hr, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + 0, 0};
            tma::store_add_async(g.vg, vg_smem[0], tile_idx);
            tma::store_commit_group();
        }
        else if (warpid == 4) {
            int4 tile_idx = {blockIdx.y / g.q.depth, (blockIdx.y % g.q.depth) / hr, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + 1, 0};
            tma::store_add_async(g.vg, vg_smem[1], tile_idx);
            tma::store_commit_group();
        }

        tma::store_async_wait();
    }
    else {
        warpgroup::increase_registers<224>();

        rt_fl<16, G::tile_width> kg_reg;
        rt_fl<16, G::tile_width> vg_reg;

        row_vec<rt_fl<16, 64>> row_reg; 

        rt_fl<16, 64> s_block_t; 
        rt_fl<16, 64> dp_block_t;
        rt_fl<16, 64> p_block_t;
        rt_fl<16, 64> ds_block_t;
        rt_bf<16, 64> p_block_t_mma;
        rt_bf<16, 64> ds_block_t_mma;

        zero(kg_reg);
        zero(vg_reg);

        wait(kv_b, 0);

        for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {

            wait(vec_b[tic], ((qo_idx - q_start)/2)%2);

            #pragma unroll
            for(int i = 0; i < 4; i++) {
                int base_col = 16*i + 2*(laneid()%4);
                s_block_t.tiles[0][i].data[0] = *(float2*)&l_smem[tic][base_col + 0];
                s_block_t.tiles[0][i].data[1] = *(float2*)&l_smem[tic][base_col + 0];
                s_block_t.tiles[0][i].data[2] = *(float2*)&l_smem[tic][base_col + 8];
                s_block_t.tiles[0][i].data[3] = *(float2*)&l_smem[tic][base_col + 8];
            }

            wait(q_b[tic], ((qo_idx - q_start)/2)%2);

            warpgroup::mma_fence(s_block_t);
            warpgroup::mma_ABt(s_block_t, k_smem[warpgroupid], q_smem[tic]);
            warpgroup::mma_commit_group();

            wait(o_b[tic], ((qo_idx - q_start)/2)%2);

            warpgroup::mma_fence(dp_block_t);
            warpgroup::mm_ABt(dp_block_t, v_smem[warpgroupid], og_smem[tic]);
            warpgroup::mma_commit_group();

            warpgroup::mma_async_wait();

            if constexpr (D == 64) { mul(s_block_t, s_block_t, 1.44269504089f*0.125f); }
            else                   { mul(s_block_t, s_block_t, 1.44269504089f*0.08838834764f); }

            if constexpr (is_causal) {
                int q_blk = (qo_idx) * (G::tile_h_qo/kittens::TILE_DIM);
                int k_blk = (blockIdx.x * BWD_CONSUMER_WARPGROUPS * (G::tile_h/kittens::TILE_DIM)) + (warpgroupid * (G::tile_h/kittens::TILE_DIM)) + (warpid % kittens::WARPGROUP_WARPS);

                for (int j = 0; j < (G::tile_h_qo/kittens::TILE_DIM); j++) {
                    int q_idx = q_blk + j;
                    auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(s_block_t.tiles[0][j]);

                    if      (q_idx  < k_blk) { neg_infty(attn_subtile); }
                    else if (q_idx == k_blk) { make_causal_t(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
                }
            }

            exp2(s_block_t, s_block_t);
            copy(p_block_t, s_block_t);
            copy(p_block_t_mma, s_block_t);

            #pragma unroll
            for(int i = 0; i < 4; i++) {
                int base_col = 16*i + 2*(laneid()%4);
                dp_block_t.tiles[0][i].data[0] = base_ops::sub::template op<float2>(dp_block_t.tiles[0][i].data[0], *(float2*)&d_smem[tic][base_col + 0]);
                dp_block_t.tiles[0][i].data[1] = base_ops::sub::template op<float2>(dp_block_t.tiles[0][i].data[1], *(float2*)&d_smem[tic][base_col + 0]);
                dp_block_t.tiles[0][i].data[2] = base_ops::sub::template op<float2>(dp_block_t.tiles[0][i].data[2], *(float2*)&d_smem[tic][base_col + 8]);
                dp_block_t.tiles[0][i].data[3] = base_ops::sub::template op<float2>(dp_block_t.tiles[0][i].data[3], *(float2*)&d_smem[tic][base_col + 8]);
            }

            mul(ds_block_t, p_block_t, dp_block_t);

            if constexpr (D == 64) { mul(ds_block_t, ds_block_t, 0.125f); }
            else                   { mul(ds_block_t, ds_block_t, 0.08838834764f); }

            warpgroup::mma_fence(vg_reg);
            warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[tic]);
            warpgroup::mma_commit_group();

            copy(ds_block_t_mma, ds_block_t);
            
            warpgroup::store(ds_smem[warpgroupid], ds_block_t);

            warpgroup::mma_fence(kg_reg);
            warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[tic]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
            asm volatile("bar.sync 10, 256;\n");
        }

        asm volatile("bar.sync 10, 256;\n");

        if (warpgroupid == 0) warpgroup::store(kg_smem[0], kg_reg);
        if (warpgroupid == 1) warpgroup::store(kg_smem[1], kg_reg);

        asm volatile("bar.sync %0, 128;\n" :: "r"(warpgroup::groupid()+4));

        if (warpid == 0) {
            int4 tile_idx = {blockIdx.y / g.q.depth, (blockIdx.y % g.q.depth) / hr, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + 0, 0};
            tma::store_add_async(g.kg, kg_smem[0], tile_idx);
            tma::store_commit_group();
        }
        else if (warpid == 4) {
            int4 tile_idx = {blockIdx.y / g.q.depth, (blockIdx.y % g.q.depth) / hr, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + 1, 0};
            tma::store_add_async(g.kg, kg_smem[1], tile_idx);
            tma::store_commit_group();
        }

        wait(qg_ready, toc);
        if(warpgroupid == 0) warpgroup::store(vg_smem[0], vg_reg);
        if(warpgroupid == 1) warpgroup::store(vg_smem[1], vg_reg);
        
        asm volatile("bar.sync %0, 128;\n" :: "r"(warpgroup::groupid()+4));

        if (warpid == 0) {
            int4 tile_idx = {blockIdx.y / g.q.depth, (blockIdx.y % g.q.depth) / hr, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + 0, 0};
            tma::store_add_async(g.vg, vg_smem[0], tile_idx);
            tma::store_commit_group();
        }
        else if (warpid == 4) {
            int4 tile_idx = {blockIdx.y / g.q.depth, (blockIdx.y % g.q.depth) / hr, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + 1, 0};
            tma::store_add_async(g.vg, vg_smem[1], tile_idx);
            tma::store_commit_group();
        }

        tma::store_async_wait();
    }
}

// #include "harness.impl"

#include "common/pyutils/torch_helpers.cuh"
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

    auto is_causal = causal; 

    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    TORCH_CHECK(qo_heads >= kv_heads, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");

    TORCH_CHECK(q.size(1) == qo_heads, "QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(l.size(1) == qo_heads, "L head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(o.size(1) == qo_heads, "O head dimension - idx 1 - must match for all inputs");

    auto hr = qo_heads / kv_heads;

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

    if (head_dim == 64) {
        using q_tile    =         st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width>;
        using k_tile    =         st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width>;
        using v_tile    =         st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width>;
        using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width>>;
        using o_tile    =         st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width>;

        using q_global = gl<bf16,  -1, -1, -1, -1, q_tile>;
        using k_global = gl<bf16,  -1, -1, -1, -1, k_tile>;
        using v_global = gl<bf16,  -1, -1, -1, -1, v_tile>;
        using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
        using o_global = gl<bf16,  -1, -1, -1, -1, o_tile>;

        using globals      = fwd_globals<64>;

        q_global qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        k_global kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        v_global vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        l_global lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,  static_cast<unsigned int>(seq_len)};
        o_global og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};

        globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, static_cast<int>(seq_len), static_cast<int>(hr)};

        auto mem_size = kittens::MAX_SHARED_MEMORY;
        auto threads  = NUM_WORKERS * kittens::WARP_THREADS;

        TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 192");
        dim3 grid(seq_len/(CONSUMER_WARPGROUPS*kittens::TILE_DIM*4), batch*qo_heads, 1);

        if (is_causal) {
            cudaFuncSetAttribute(
                fwd_attend_ker<64, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<64, true><<<grid, (32*NUM_WORKERS), mem_size>>>(g);
        }
        else {
            cudaFuncSetAttribute(
                fwd_attend_ker<64, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<64, false><<<grid, (32*NUM_WORKERS), mem_size>>>(g);
        }
    }

    if (head_dim == 128) {
        using q_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>;
        using k_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width>;
        using v_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width>;
        using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>>;
        using o_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>;

        using q_global = gl<bf16,  -1, -1, -1, -1, q_tile>;
        using k_global = gl<bf16,  -1, -1, -1, -1, k_tile>;
        using v_global = gl<bf16,  -1, -1, -1, -1, v_tile>;
        using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
        using o_global = gl<bf16,  -1, -1, -1, -1, o_tile>;

        using globals      = fwd_globals<128>;

        q_global qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        k_global kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        v_global vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        l_global lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len)};
        o_global og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};

        globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, static_cast<int>(seq_len), static_cast<int>(hr)};

        auto mem_size = kittens::MAX_SHARED_MEMORY;
        auto threads  = NUM_WORKERS * kittens::WARP_THREADS;

        TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 192");
        dim3 grid(seq_len/(CONSUMER_WARPGROUPS*kittens::TILE_DIM*4), batch*qo_heads, 1);

        if (is_causal) {
            cudaFuncSetAttribute(
                fwd_attend_ker<128, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<128, true><<<grid, (32*NUM_WORKERS), mem_size>>>(g);
        }
        else {
            cudaFuncSetAttribute(
                fwd_attend_ker<128, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<128, false><<<grid, (32*NUM_WORKERS), mem_size>>>(g);
        }
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

    // check if causal
    auto is_causal = causal;

    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    TORCH_CHECK(qo_heads >= kv_heads, "Q heads must be greater than or equal to K and V heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "Q heads must be divisible by KV heads");

    TORCH_CHECK(q.size(1) == qo_heads, "Q heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(l_vec.size(1) == qo_heads, "L heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(d_vec.size(1) == qo_heads, "D heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(o.size(1) == qo_heads, "O heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(og.size(1) == qo_heads, "OG heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(qg.size(1) == qo_heads, "QG heads dimension - idx 1 - must match for all inputs");

    TORCH_CHECK(k.size(1) == kv_heads, "K heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_heads, "V heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(kg.size(1) == kv_heads, "KG heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(vg.size(1) == kv_heads, "VG heads dimension - idx 1 - must match for all inputs");

    auto hr = qo_heads / kv_heads;

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

    auto mem_size = kittens::MAX_SHARED_MEMORY; 
    auto threads  = 4 * kittens::WARP_THREADS;

    TORCH_CHECK(seq_len % (4*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 256");
    dim3 grid_bwd(seq_len/(4*kittens::TILE_DIM*4), batch*qo_heads, 1);

    if (head_dim == 64)  {
        using og_tile = st_bf<4*16, 64>;
        using o_tile  = st_bf<4*16, 64>;
        using d_tile  = col_vec<st_fl<4*16, 64>>;

        using og_global = gl<bf16,  -1, -1, -1, -1, og_tile>;
        using o_global  = gl<bf16,  -1, -1, -1, -1, o_tile>;
        using d_global  = gl<float, -1, -1, -1, -1, d_tile>;

        using bwd_prep_globals = bwd_prep_globals<64>;

        og_global prep_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        o_global  prep_o_arg {d_o,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        d_global  prep_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,  static_cast<unsigned int>(seq_len)};

        bwd_prep_globals bwd_g{prep_og_arg, prep_o_arg, prep_d_arg};

        cudaFuncSetAttribute(
            bwd_attend_prep_ker<64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        bwd_attend_prep_ker<64><<<grid_bwd, threads, mem_size>>>(bwd_g); 

        using bwd_q_tile    =         st_bf<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_k_tile    =         st_bf<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_v_tile    =         st_bf<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_og_tile   =         st_bf<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_qg_tile   =         st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_kg_tile   =         st_fl<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_vg_tile   =         st_fl<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_l_tile    = row_vec<st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_h>>;
        using bwd_d_tile    = row_vec<st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_h>>;

        using bwd_q_global  = gl<bf16,  -1, -1, -1, -1, bwd_q_tile>;
        using bwd_k_global  = gl<bf16,  -1, -1, -1, -1, bwd_k_tile>;
        using bwd_v_global  = gl<bf16,  -1, -1, -1, -1, bwd_v_tile>;

        using bwd_og_global = gl<bf16,  -1, -1, -1, -1, bwd_og_tile>;

        using bwd_qg_global = gl<float, -1, -1, -1, -1, bwd_qg_tile>;
        using bwd_kg_global = gl<float, -1, -1, -1, -1, bwd_kg_tile>;
        using bwd_vg_global = gl<float, -1, -1, -1, -1, bwd_vg_tile>;

        using bwd_l_global  = gl<float, -1, -1, -1, -1, bwd_l_tile>;
        using bwd_d_global  = gl<float, -1, -1, -1, -1, bwd_d_tile>;

        using bwd_global_args = bwd_globals<64>;

        bwd_q_global  bwd_q_arg {d_q,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_k_global  bwd_k_arg {d_k,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_v_global  bwd_v_arg {d_v,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_og_global bwd_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_qg_global bwd_qg_arg{d_qg, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_kg_global bwd_kg_arg{d_kg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_vg_global bwd_vg_arg{d_vg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_l_global  bwd_l_arg {d_l,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,  static_cast<unsigned int>(seq_len)};
        bwd_d_global  bwd_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,  static_cast<unsigned int>(seq_len)};

        bwd_global_args bwd_global{bwd_q_arg, 
                        bwd_k_arg, 
                        bwd_v_arg, 
                        bwd_og_arg, 
                        bwd_qg_arg, 
                        bwd_kg_arg, 
                        bwd_vg_arg, 
                        bwd_l_arg, 
                        bwd_d_arg, 
                        static_cast<int>(seq_len), 
                        static_cast<int>(hr)};

        TORCH_CHECK(seq_len % (4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_DIM) == 0, "sequence length must be divisible by 128");
        dim3 grid_bwd_2(seq_len/(4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_DIM), batch*qo_heads, 1);
        threads = kittens::WARP_THREADS * BWD_NUM_WORKERS;

        mem_size = kittens::MAX_SHARED_MEMORY / bwd_attend_ker_tile_dims<64>::blocks_sm;

        if (is_causal) {
            cudaFuncSetAttribute(
                bwd_attend_ker<64, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );
            
            bwd_attend_ker<64, true><<<grid_bwd_2, threads, mem_size>>>(bwd_global); 
        }
        else {
            cudaFuncSetAttribute(
                bwd_attend_ker<64, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );
            
            bwd_attend_ker<64, false><<<grid_bwd_2, threads, mem_size>>>(bwd_global); 
        }
    }

    if (head_dim == 128) {
        using og_tile = st_bf<4*16, 128>;
        using o_tile  = st_bf<4*16, 128>;
        using d_tile  = col_vec<st_fl<4*16, 128>>;

        using og_global = gl<bf16,  -1, -1, -1, -1, og_tile>;
        using o_global  = gl<bf16,  -1, -1, -1, -1, o_tile>;
        using d_global  = gl<float, -1, -1, -1, -1, d_tile>;

        using bwd_prep_globals = bwd_prep_globals<128>;

        og_global prep_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        o_global  prep_o_arg {d_o,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        d_global  prep_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len)};

        bwd_prep_globals bwd_g{prep_og_arg, prep_o_arg, prep_d_arg};

        cudaFuncSetAttribute(
            bwd_attend_prep_ker<128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        bwd_attend_prep_ker<128><<<grid_bwd, threads, mem_size>>>(bwd_g); 

        using bwd_q_tile    =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_k_tile    =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_v_tile    =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_og_tile   =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_qg_tile   =         st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_kg_tile   =         st_fl<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_vg_tile   =         st_fl<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_l_tile    = row_vec<st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_h>>;
        using bwd_d_tile    = row_vec<st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_h>>;

        using bwd_q_global  = gl<bf16,  -1, -1, -1, -1, bwd_q_tile>;
        using bwd_k_global  = gl<bf16,  -1, -1, -1, -1, bwd_k_tile>;
        using bwd_v_global  = gl<bf16,  -1, -1, -1, -1, bwd_v_tile>;

        using bwd_og_global = gl<bf16,  -1, -1, -1, -1, bwd_og_tile>;

        using bwd_qg_global = gl<float, -1, -1, -1, -1, bwd_qg_tile>;
        using bwd_kg_global = gl<float, -1, -1, -1, -1, bwd_kg_tile>;
        using bwd_vg_global = gl<float, -1, -1, -1, -1, bwd_vg_tile>;

        using bwd_l_global  = gl<float, -1, -1, -1, -1, bwd_l_tile>;
        using bwd_d_global  = gl<float, -1, -1, -1, -1, bwd_d_tile>;

        using bwd_global_args = bwd_globals<128>;

        bwd_q_global  bwd_q_arg {d_q,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_k_global  bwd_k_arg {d_k,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_v_global  bwd_v_arg {d_v,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_og_global bwd_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_qg_global bwd_qg_arg{d_qg, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_kg_global bwd_kg_arg{d_kg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_vg_global bwd_vg_arg{d_vg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_l_global  bwd_l_arg {d_l,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len)};
        bwd_d_global  bwd_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len)};

        bwd_global_args bwd_global{
                        bwd_q_arg, 
                        bwd_k_arg, 
                        bwd_v_arg, 
                        bwd_og_arg, 
                        bwd_qg_arg, 
                        bwd_kg_arg, 
                        bwd_vg_arg, 
                        bwd_l_arg, 
                        bwd_d_arg, 
                        static_cast<int>(seq_len), 
                        static_cast<int>(hr)};
        
        TORCH_CHECK(seq_len % (4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_DIM) == 0, "sequence length must be divisible by 128");
        dim3 grid_bwd_2(seq_len/(4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_DIM), batch*qo_heads, 1);
        threads = kittens::WARP_THREADS * BWD_NUM_WORKERS;

        mem_size = kittens::MAX_SHARED_MEMORY / bwd_attend_ker_tile_dims<128>::blocks_sm;

        if (is_causal) {
            cudaFuncSetAttribute(
                bwd_attend_ker<128, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );
            
            bwd_attend_ker<128, true><<<grid_bwd_2, threads, mem_size>>>(bwd_global); 
        }
        else {
            cudaFuncSetAttribute(
                bwd_attend_ker<128, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );
            
            bwd_attend_ker<128, false><<<grid_bwd_2, threads, mem_size>>>(bwd_global); 
        }
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}
