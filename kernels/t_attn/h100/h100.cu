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
    constexpr static int kv_height  = (8); // 8 is better for 768, 1536
    constexpr static int stages     = (4); 
};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128/kittens::TILE_DIM);
    constexpr static int qo_height  = (4);
    constexpr static int kv_height  = (4);
    constexpr static int stages     = (2); 
};

template<int D>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void triangle_attention(int N, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, CUtensorMap* tma_b, CUtensorMap* tma_o) {

    // this is the CUDA shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile  = st_bf<K::qo_height, K::tile_width>; // 4 * 8 * 256 * 2 = 16KB -> 48
    using k_tile  = st_bf<K::kv_height, K::tile_width>; // 4 * 8 * 256 * 2 = 32KB -> 16
    using v_tile  = st_bf<K::kv_height, K::tile_width>; // 4 * 8 * 256 * 2 = 32KB -> 16
    using b_tile  = st_bf<K::qo_height, K::kv_height>;  // 4 * 4 * 256 * 2 = 8KB  -> 48
    using o_tile  = st_bf<K::qo_height, K::tile_width>;
    
    q_tile    (&q_smem)[CONSUMER_WARPGROUPS]             = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[K::stages]                       = al.allocate<k_tile, K::stages          >();
    v_tile    (&v_smem)[K::stages]                       = al.allocate<v_tile, K::stages          >();
    o_tile    (&b_smem)[K::stages][CONSUMER_WARPGROUPS]  = al.allocate<o_tile, K::stages, CONSUMER_WARPGROUPS>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem); // reuse q memory
    
    int kv_blocks = N / (K::kv_height*TILE_DIM);

    __shared__ kittens::barrier qsmem_barrier; 
    __shared__ kittens::barrier k_smem_arrived[K::stages]; 
    __shared__ kittens::barrier v_smem_arrived[K::stages]; 
    __shared__ kittens::barrier b_smem_arrived[K::stages];
    __shared__ kittens::barrier compute_done[K::stages];

    if (threadIdx.x == 0) { // initialize barriers and initial loads
        init_barrier(qsmem_barrier, 0, 1); // no threads, one transaction
        for(int j = 0; j < K::stages; j++) {
            init_barrier(k_smem_arrived[j], 0, 1); // no threads, one transaction
            init_barrier(v_smem_arrived[j], 0, 1); // no threads, one transaction
            init_barrier(b_smem_arrived[j], 0, 1); // no threads, one transaction
            init_barrier(compute_done[j], CONSUMER_WARPGROUPS, 0); // all the consumer threads across both blocks, no transactions
        }

        int q_tile_idx = (blockIdx.y * CONSUMER_WARPGROUPS * gridDim.x) + (blockIdx.x * CONSUMER_WARPGROUPS); 
        int kv_tile_idx = ((blockIdx.y) * kv_blocks);
        
        tma::expect_bytes(qsmem_barrier, sizeof(q_smem[0])*CONSUMER_WARPGROUPS);
        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            tma::load_async(q_smem[wg], tma_q, qsmem_barrier, q_tile_idx + wg);
        }

        for (int j = 0; j < K::stages - 1; j++) {
            tma::expect<typeof(k_smem[0])>(k_smem_arrived[j]);
            tma::expect<typeof(v_smem[0])>(v_smem_arrived[j]);
            tma::expect_bytes(b_smem_arrived[j], sizeof(b_smem[0][0])*CONSUMER_WARPGROUPS);

            tma::load_async(k_smem[j], tma_k, k_smem_arrived[j], kv_tile_idx + j); 
            tma::load_async(v_smem[j], tma_v, v_smem_arrived[j], kv_tile_idx + j);

            for (int g = 0; g < CONSUMER_WARPGROUPS; g++) {
                tma::load_async(b_smem[j][g], tma_b, b_smem_arrived[j], q_tile_idx + g, kv_tile_idx + j);
            }
        }
    }
    __syncthreads(); 

    int pipe_idx = K::stages - 1; 
    
    if(warpgroupid == NUM_WARPGROUPS-1) { // producer warpgroup
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));           
        
        int kv_iters = kv_blocks-2; 

        if(warpid == NUM_WORKERS-4) {
            for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                int q_tile_idx = (blockIdx.y * CONSUMER_WARPGROUPS * gridDim.x) + (blockIdx.x * CONSUMER_WARPGROUPS); 
                int kv_tile_idx = ((blockIdx.y) * kv_blocks) + (kv_idx + 1);
                
                tma::expect<typeof(k_smem[0])>(k_smem_arrived[(kv_idx+1)%K::stages]);
                tma::load_async(k_smem[(kv_idx+1)%K::stages], tma_k, k_smem_arrived[(kv_idx+1)%K::stages], kv_tile_idx, 0);
                
                tma::expect<typeof(v_smem[0])>(v_smem_arrived[(kv_idx+1)%K::stages]); 
                tma::load_async(v_smem[(kv_idx+1)%K::stages], tma_v, v_smem_arrived[(kv_idx+1)%K::stages], kv_tile_idx, 0);

                tma::expect_bytes(b_smem_arrived[(kv_idx+1)%K::stages], sizeof(b_smem[0][0])*CONSUMER_WARPGROUPS);
                for (int g = 0; g < CONSUMER_WARPGROUPS; g++) {
                    tma::load_async(b_smem[(kv_idx+1)%K::stages][g], tma_b, b_smem_arrived[(kv_idx+1)%K::stages], q_tile_idx + g, kv_tile_idx);
                }

                wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            }
        }
    }
    else { // consumer warpgroup
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(160));

        // premultiply by temperature and lg(e)
        wait(qsmem_barrier, 0);
        if constexpr (D == 64) { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f)); }
        else                   { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.08838834764f)); }
    
        rt_fl<1, K::kv_height>  att_block; 
        rt_fl<1, K::kv_height>  att_block_scaled;
        rt_bf<1, K::kv_height>  att_block_mma;
        rt_fl<1, K::tile_width> o_reg;
        col_vec<rt_fl<1, K::kv_height>> max_vec_last,        max_vec;
        col_vec<rt_fl<1, K::kv_height>> max_vec_last_scaled, max_vec_scaled;
        col_vec<rt_fl<1, K::kv_height>> norm_vec_last,       norm_vec;
        
        neg_infty(max_vec); // clear registers for the Q chunk
        zero(norm_vec);
        zero(o_reg);

        int kv_iters = kv_blocks - 1; 
        
        const int kv_do = (blockIdx.x * CONSUMER_WARPGROUPS)/(K::kv_height/K::qo_height);

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
        
            wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2); // wait on k memory
            
            warpgroup::mma_fence(att_block);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(kv_idx)%K::stages]);
            warpgroup::mma_commit_group();
            
            wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2); // wait on v memory, during the matmul
            
            copy(norm_vec_last, norm_vec);
            copy(max_vec_last,  max_vec);
            
            warpgroup::mma_async_wait();

            row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
            
            mul(att_block_scaled, att_block, 1.44269504089f);
            mul(max_vec_scaled,   max_vec,   1.44269504089f);     
            sub_row(att_block_scaled, att_block_scaled, max_vec_scaled);
            exp2(att_block, att_block_scaled);

            mul(max_vec_last_scaled, max_vec_last, 1.44269504089f);
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last,       max_vec_last_scaled);
            mul(norm_vec,            norm_vec,     max_vec_last);

            row_sum(norm_vec,  att_block, norm_vec); // accumulate onto the norm_vec
            div_row(att_block, att_block, norm_vec);
            
            mul(norm_vec_last, norm_vec_last, max_vec_last);
            div(norm_vec_last, norm_vec_last, norm_vec);
            
            copy(att_block_mma, att_block); // convert to bf16 for mma
            mul_row(o_reg, o_reg, norm_vec_last); // normalize o_prev in advance of mma'ing onto it

            warpgroup::mma_fence(o_reg);
            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(kv_idx)%K::stages]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
            
            if(warpgroup::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
        }

        warpgroup::store(o_smem[warpgroupid], o_reg); 
        warpgroup::sync(); 

        if (warpid % 4 == 0) { // store o
            int tile_idx = (blockIdx.y * CONSUMER_WARPGROUPS * gridDim.x) + (blockIdx.x * CONSUMER_WARPGROUPS) + warpgroupid;
            tma::store_async(tma_o, o_smem[warpgroupid], tile_idx); 
            tma::store_commit_group(); 
        }
    
        tma::store_async_wait();
    }
}

#include "harness.impl"