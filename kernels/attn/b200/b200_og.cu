#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

constexpr int CONSUMER_WARPGROUPS = (4); 
constexpr int PRODUCER_WARPGROUPS = (1); 
constexpr int NUM_WARPGROUPS      = (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS); 
constexpr int NUM_WORKERS         = (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 
constexpr int NUM_CONSUMERS       = CONSUMER_WARPGROUPS/2;

constexpr bool causal = false;

using namespace kittens;
namespace cg = cooperative_groups;

struct rescale_add {
    template<typename T> static __device__ inline T op(const T &a, const T &b) {
        if constexpr (std::is_same_v<T, float2>) {
            constexpr float2 scale = {1.44269504089f*0.08838834764f, 1.44269504089f*0.08838834764f};
            float2 c;
            asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(*(uint64_t*)&c) : "l"(*(uint64_t*)&a), "l"(*(uint64_t*)&scale), "l"(*(uint64_t*)&b));
            return c;
        }
        else {
            static_assert(sizeof(T) == 999, "Currently unsupported type");
        }
    }
};
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void rescale_add_row(T &dst, const T &src, const V &row_values) {
    row_map<rescale_add, T, V>(dst, src, row_values);
}

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

    // Allocate smem
    using consumer = group<8>;
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpgroup::groupid(), consumerid = consumer::groupid();

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile    =         st_bf<K::qo_height, K::tile_width>;
    using k_tile    =         st_bf<K::kv_height, K::tile_width>;
    using v_tile    =         st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile    =         st_bf<K::qo_height, K::tile_width>;
    
    q_tile    (&q_smem)[NUM_CONSUMERS] = al.allocate<q_tile, NUM_CONSUMERS>();
    k_tile    (&k_smem)[K::stages]     = al.allocate<k_tile, K::stages>();
    v_tile    (&v_smem)[K::stages]     = al.allocate<v_tile, K::stages>();
    l_col_vec (&l_smem)[NUM_CONSUMERS] = al.allocate<l_col_vec, NUM_CONSUMERS>();
    auto      (*o_smem)                = reinterpret_cast<o_tile(*)>(&q_smem);

    // Allocate tmem
    auto all_tmem = allocate_tmem();

    using att_tm_fl = tmem<float, K::qo_height, K::kv_height>;
    using att_tm_bf = tmem<bf16,  K::qo_height, K::kv_height>;
    using o_tm_fl   = tmem<float, K::qo_height, K::tile_width>;

    att_tm_fl att_tm    = all_tmem.subtile<att_tm_fl>(0, consumerid*K::kv_height);
    o_tm_fl   o_tm      = all_tmem.subtile<o_tm_fl>  (0, (NUM_CONSUMERS*K::kv_height) + consumerid*K::tile_width);
    att_tm_bf att_bf_tm = reinterpret_cast<att_tm_bf&>(att_tm);
    // att_tm_bf att_bf_tm = all_tmem.subtile<att_tm_bf>(0, (NUM_CONSUMERS*(K::kv_height+K::tile_width)) + consumerid*K::kv_height/2);
    
    int kv_blocks   = g.N / (K::kv_height);
    int kv_head_idx = blockIdx.y / g.hr;
    int seq_idx     = blockIdx.x * (NUM_CONSUMERS); 

    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    __shared__ kittens::semaphore mma_semaphore[NUM_CONSUMERS];
    if (threadIdx.x == 0) { 
        init_semaphore(qsmem_semaphore, 0, 1); 
        for(int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1); 
            init_semaphore(v_smem_arrived[j], 0, 1); 
            init_semaphore(compute_done[j], NUM_CONSUMERS, 0); 
        }
        for(int j = 0; j < NUM_CONSUMERS; j++) {
            init_semaphore(mma_semaphore[j], 0, 1);
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < NUM_CONSUMERS; wg++) {
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
        warpgroup::decrease_registers<24>();      
        
        int kv_iters = kv_blocks-2;

        if(warpid == NUM_WORKERS-4) {
            for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx + 1, 0};
                tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
                tma::load_async(k_smem[(kv_idx+1)%K::stages], g.k, kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
                tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));
                tma::load_async(v_smem[(kv_idx+1)%K::stages], g.v, kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);
                
                kittens::wait(compute_done[(kv_idx+K::stages)%K::stages], ((kv_idx)/K::stages)%2);
            }
        }
    }
    else {
        warpgroup::increase_registers<112>();

        rt_fl<16, K::kv_height>  att_block;
        rt_bf<16, K::kv_height>  att_block_mma;
        rt_fl<16, K::tile_width> o_reg;
        
        col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;
        
        neg_infty(max_vec);
        zero(norm_vec);

        int kv_iters = kv_blocks - 1;

        kittens::wait(qsmem_semaphore, 0);

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {

            kittens::wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            
            if (consumer::warpid() == 0) mm_ABt(att_tm, q_smem[consumerid], k_smem[(kv_idx)%K::stages], mma_semaphore[consumerid]);
            
            copy(max_vec_last_scaled, max_vec);
            mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f);
            
            kittens::wait(mma_semaphore[consumerid], 0);
            
            consumer::load_async(att_block, att_tm); 
            tm_load_wait();

            row_max(max_vec, att_block, max_vec);
            
            // mul(att_block,    att_block, 1.44269504089f*0.08838834764f); 
            mul(max_vec_scaled, max_vec, -1.44269504089f*0.08838834764f);

            rescale_add_row(att_block, att_block, max_vec_scaled);
            // sub_row(att_block, att_block, max_vec_scaled);
            exp2(att_block, att_block);
            add(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last_scaled, max_vec_last_scaled);
            mul(norm_vec, norm_vec, max_vec_last_scaled);
            row_sum(norm_vec,  att_block, norm_vec);
            copy(att_block_mma, att_block); 
            consumer::store_async(att_bf_tm, att_block_mma);

            kittens::wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            tm_store_wait();

            if(kv_idx == 0) { zero(o_reg); }
            else {
                consumer::load_async(o_reg, o_tm);
                tm_load_wait();
            }
            mul_row(o_reg, o_reg, max_vec_last_scaled);
            consumer::store_async(o_tm, o_reg);
            tm_store_wait();
            consumer::sync(consumerid);
           
            if (consumer::warpid() == 0) mma_AB(o_tm, att_bf_tm, v_smem[(kv_idx)%K::stages], mma_semaphore[consumerid]);
        
            kittens::wait(mma_semaphore[consumerid], 1); 
            consumer::sync(consumerid);

            if(consumer::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
        }

        consumer::load_async(o_reg, o_tm);
        tm_load_wait();
        div_row(o_reg, o_reg, norm_vec);
        consumer::store(o_smem[consumerid], o_reg); 
        consumer::sync(consumerid);

        if (consumer::warpid() == 0) {
            coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + (consumerid), 0};
            tma::store_async(g.o, o_smem[consumerid], o_tile_idx);
        }
        
        tma::store_async_wait();
    }
}

#include "harness.impl"