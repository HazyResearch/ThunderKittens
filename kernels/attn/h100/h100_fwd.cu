#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

#define CONSUMER_WARPGROUPS (3) // hardcoded
#define PRODUCER_WARPGROUPS (1) // hardcoded
#define NUM_WARPGROUPS (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS)
#define NUM_WORKERS (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS)

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
    
    st_bf<K::qo_height, K::tile_width>          (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<st_bf<K::qo_height, K::tile_width>,          CONSUMER_WARPGROUPS>();
    st_bf<K::kv_height, K::tile_width>          (&k_smem)[2]                   = al.allocate<st_bf<K::kv_height, K::tile_width>,          2                  >();
    st_bf<K::kv_height, K::tile_width>          (&v_smem)[2]                   = al.allocate<st_bf<K::kv_height, K::tile_width>,          2                  >();
    col_vec<st_fl<K::qo_height, K::tile_width>> (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<col_vec<st_fl<K::qo_height, K::tile_width>>, CONSUMER_WARPGROUPS>();
    
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
        // asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));           
        
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
        // asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(160));

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

        auto (*o_smem) = reinterpret_cast<st_bf<K::qo_height, K::tile_width>(*)>(q_smem); // reuse q memory
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

    st_bf<4,         D/kittens::TILE_DIM>  (&og_smem)[4] = al.allocate<        st_bf<4, D/kittens::TILE_DIM>,  4>();
    st_bf<4,         D/kittens::TILE_DIM>  (&o_smem) [4] = al.allocate<        st_bf<4, D/kittens::TILE_DIM>,  4>();
    col_vec<st_fl<4, D/kittens::TILE_DIM>> (&d_smem) [4] = al.allocate<col_vec<st_fl<4, D/kittens::TILE_DIM>>, 4>();

            rt_fl<4, D/kittens::TILE_DIM> og_reg;
            rt_fl<4, D/kittens::TILE_DIM>  o_reg; 
    col_vec<rt_fl<4, D/kittens::TILE_DIM>> d_reg;

    __shared__ uint64_t smem_barrier;

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

#include "harness.impl"