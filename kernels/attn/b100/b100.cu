#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

constexpr int CONSUMER_WARPGROUPS = (2); 
constexpr int PRODUCER_WARPGROUPS = (1); 
constexpr int NUM_WARPGROUPS      = (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS); 
constexpr int NUM_WORKERS         = (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 

using namespace kittens;
namespace cg = cooperative_groups;

template<int D> struct fwd_attend_ker_tile_dims {};
// template<> struct fwd_attend_ker_tile_dims<64> {
//     constexpr static int tile_width = (64);
//     constexpr static int qo_height  = (4*16);
//     constexpr static int kv_height  = (8*16);
//     constexpr static int stages     = (4); 
// };
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (128);
    constexpr static int kv_height  = (128);
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

    auto all_tmem = allocate_tmem();

    auto att_tm    = all_tmem.subtile<tmem<float, 128, 128>>(0, 0);
    auto o_tm      = att_tm.subtile<tmem<float, 128, 128>>(0, 128);
    auto att_bf_tm = o_tm.subtile<tmem<bf16, 128, 128>>(0, 128);

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile    =         st_bf<K::qo_height, K::tile_width>;
    using k_tile    =         st_bf<K::kv_height, K::tile_width>;
    using v_tile    =         st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile    =         st_bf<K::qo_height, K::tile_width>;
    
    q_tile    (&q_smem)[CONSUMER_WARPGROUPS/2] = al.allocate<q_tile, CONSUMER_WARPGROUPS/2>();
    k_tile    (&k_smem)[K::stages]             = al.allocate<k_tile, K::stages>();
    v_tile    (&v_smem)[K::stages]             = al.allocate<v_tile, K::stages>();
    l_col_vec (&l_smem)[CONSUMER_WARPGROUPS/2] = al.allocate<l_col_vec, CONSUMER_WARPGROUPS/2>();
    auto      (&o_smem)                        = reinterpret_cast<o_tile(&)>(q_smem);
    
    int kv_blocks   = g.N / (K::kv_height);
    int kv_head_idx = blockIdx.y / g.hr;
    int seq_idx     = blockIdx.x * (CONSUMER_WARPGROUPS/2); 

    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    __shared__ kittens::semaphore mma_semaphore;
    if (threadIdx.x == 0) { 
        init_semaphore(qsmem_semaphore, 0, 1); 
        for(int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1); 
            init_semaphore(v_smem_arrived[j], 0, 1); 
            init_semaphore(compute_done[j], CONSUMER_WARPGROUPS/2, 0); 
        }
        init_semaphore(mma_semaphore, 0, 1);

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < CONSUMER_WARPGROUPS/2; wg++) {
            coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }

        for (int j = 0; j < K::stages - 1; j++) {
            coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, j, 0};
            tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
            tma::load_async(k_smem[j], g.k, kv_tile_idx, k_smem_arrived[j]);
            tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
            tma::load_async(v_smem[j], g.v, kv_tile_idx, v_smem_arrived[j]);
        }
    }

    __syncthreads(); 

    int pipe_idx = K::stages - 1; 
    
    if(warpgroupid == NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();      
        
        int kv_iters = kv_blocks-2;

        if(warpid == NUM_WORKERS-4) {
            for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx + 1, 0};
                tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
                tma::load_async(k_smem[(kv_idx+1)%K::stages], g.k, kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
                tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));
                tma::load_async(v_smem[(kv_idx+1)%K::stages], g.v, kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);
                
                wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            }
        }
    }
    else {
        warpgroup::increase_registers<224>();

        rt_fl<16, K::kv_height>  att_block;
        rt_bf<16, K::kv_height>  att_block_mma;
        rt_fl<16, K::tile_width> o_reg;
        
        col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;
        
        neg_infty(max_vec);
        zero(norm_vec);
        zero(o_reg);

        int kv_iters = kv_blocks - 1;

        wait(qsmem_semaphore, 0);

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {

            wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            
            if (warpid % 8 == 0) mm_ABt(att_tm, q_smem[warpgroupid/2], k_smem[(kv_idx)%K::stages], mma_semaphore);
            
            copy(max_vec_last_scaled, max_vec);
            if constexpr (D == 64) { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
            else                   { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }
            
            wait(mma_semaphore, 0);
            
            group<8>::load_async(att_block, att_tm); 
            tm_load_wait();
            group<8>::sync(0 + (warpgroupid/2));

            row_max(max_vec, att_block, max_vec);
            
            if constexpr (D == 64) { 
                mul(att_block,    att_block, 1.44269504089f*0.125f); 
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
            }
            else                   { 
                mul(att_block,    att_block, 1.44269504089f*0.08838834764f); 
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
            }

            sub_row(att_block, att_block, max_vec_scaled);
            exp2(att_block, att_block);
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last_scaled,       max_vec_last_scaled);
            mul(norm_vec,            norm_vec,     max_vec_last_scaled);
            row_sum(norm_vec,  att_block, norm_vec);
            copy(att_block_mma, att_block); 

            wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            group<8>::sync(0 + (warpgroupid/2));

            if(kv_idx == 0) { zero(o_reg); }
            else {
                group<8>::load_async(o_reg, o_tm);
                tm_load_wait();
            }
            group<8>::sync(0 + (warpgroupid/2));
            mul_row(o_reg, o_reg, max_vec_last_scaled);
            group<8>::store_async(att_bf_tm, att_block_mma);
            group<8>::store_async(o_tm, o_reg);
            tm_store_wait();
            group<8>::sync(0 + (warpgroupid/2));
           
            if (warpid % 8 == 0) mma_AB(o_tm, att_bf_tm, v_smem[(kv_idx)%K::stages], mma_semaphore);
        
            wait(mma_semaphore, 1); 
            group<8>::sync(0 + (warpgroupid/2));

            if(group<8>::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
        }

        group<8>::load_async(o_reg, o_tm);
        tm_load_wait();
        div_row(o_reg, o_reg, norm_vec);
        group<8>::store(o_smem, o_reg); 
        group<8>::sync(0 + (warpgroupid/2));

        if (warpid % 8 == 0) {
            coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + (warpgroupid/2), 0};
            tma::store_async(g.o, o_smem, o_tile_idx);
        }
        
        tma::store_async_wait();
    }
}

#include "harness.impl"