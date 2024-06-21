// #define TORCH_COMPILE // defined by default for PyTorch bindings - to use cpp harness, comment this out

#ifdef TORCH_COMPILE
#include "src/kittens.cuh"
#else
#include "../../../../src/kittens.cuh"
#endif

#include <cooperative_groups.h>
#include <cuda/pipeline>

#define NUM_WORKERS (8)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define NUM_WARPGROUPS (NUM_WORKERS/kittens::WARPGROUP_WARPS)

using namespace kittens;

template<ducks::sv::all SV, ducks::st::all ST>
__device__ inline void cumulative_add(SV &dst, const ST &src) {
    // this is called along a warpgroup
    static_assert(ST::cols <= 128);
    static_assert(ST::cols == SV::length);
    int lane = threadIdx.x % 128;
    if(lane < ST::cols) {
        float f = dst[lane];
        // acc equal to the last row of dst
        for (auto i = 0; i < ST::rows; i++) {
            f += __bfloat162float(src[{i, lane}]);
        }
        dst[lane] = f;
    }
}

template<ducks::rt::all RT>
__device__ inline void softmax_featuremap_inplace(RT &dst) {
    col_vec<RT> max_vec, sum_vec;
    row_max(max_vec, dst);
    sub_row(dst, dst, max_vec); // now in range (-infty, 0) for numerical stability
    exp(dst, dst);
    row_sum(sum_vec, dst);
    div_row(dst, dst, sum_vec);
}

#define ATTN_D 128
#define ATTN_F 128

using q_tile = st_bf<4, 8, wgmma_swizzle_l>;
using k_tile = st_bf<4, 8, wgmma_swizzle_l>;
using v_tile = st_bf<4, 8, wgmma_interleave_l>;
using o_tile = st_bf<4, 8, wgmma_swizzle_l>;

using qk_map_tile = st_bf<8, 4, wgmma_interleave_l>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void hedgehog_linear_attention_smd(int n, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, 
                                                CUtensorMap* tma_o,
                                                const CUtensorMap* tma_qmap, const CUtensorMap* tma_kmap,
                                                float alpha, float beta)  { // alpha is for linear component, beta is for sliding window component

    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    q_tile (&q_smem) [2] = al.allocate<q_tile, 2>(); // 32k, (tic/toc)*16k
    k_tile (&k_smem) [3] = al.allocate<k_tile, 3>(); // 48k, (3-ring)*(64x128)
    v_tile (&v_smem) [3] = al.allocate<v_tile, 3>(); // 48k, (3-ring)*(64x128)
    o_tile (&o_smem)     = al.allocate<o_tile>   (); // 16k

    qk_map_tile (&qf_map) = al.allocate<qk_map_tile>(); // 16k, for fusing featuremap computation
    qk_map_tile (&kf_map) = al.allocate<qk_map_tile>(); // 16k, for fusing featuremap computation

    st_bf<4, 8, wgmma_interleave_l> (&kv_smem)        [2] = al.allocate<st_bf<4, 8, wgmma_interleave_l>, 2>(); // 32k, 64x128 featurized 
    
    row_vec<st_bf<4,4>> (&cumsum_k_smem)              [2] = al.allocate<row_vec<st_bf<4,4>>, 2>(); // smol
    col_vec<st_bf<4,4>> (&norm_exchange)              [2] = al.allocate<col_vec<st_bf<4,4>>, 2>(); // smol

    st_bf<4, 4, wgmma_interleave_l> (*k_scratch_smem)     = reinterpret_cast<st_bf<4, 4, wgmma_interleave_l>*>(&kv_smem[0].data[0]);


    int warpid = kittens::warpid();
    int warpgroupid = warpid >= 4;

    int tic = 0, toc = 1;
    int ring_id = 0;
    __shared__ uint64_t qkv_barrier;

    int blocks = n / (q_tile::rows);

    if (warpid == 0) {
        tma::init_barrier(qkv_barrier, 1);
        tma::set_bytes(qkv_barrier, 
            size_bytes<typeof(q_smem[0])> + 
            size_bytes<typeof(k_smem[0])> + 
            size_bytes<typeof(v_smem[0])> +
            // we need qk maps to be loaded on this first iter, too.
            size_bytes<typeof(qf_map)> +
            size_bytes<typeof(kf_map)>
        );
        int tile_idx = (blockIdx.x * blocks) + 0;
        // first thing we need to do is load the QK map
        tma::load_async(qf_map, tma_qmap, qkv_barrier, 0);
        tma::load_async(kf_map, tma_kmap, qkv_barrier, 0);
        // now we also load the first data we need
        tma::load_async(q_smem[tic],       tma_q, qkv_barrier, tile_idx);
        tma::load_async(k_smem[ring_id+1], tma_k, qkv_barrier, tile_idx);
        tma::load_async(v_smem[ring_id+1], tma_v, qkv_barrier, tile_idx);
    }
    __syncthreads();

    rt_fl<1, 8> local_kv; // this is going to be split across the two warpgroups involved.

    zero(local_kv);
    warpgroup::zero(cumsum_k_smem[warpgroupid]);

    for (int block = 0; block < blocks; block++, tic^=1, toc^=1, ring_id=(ring_id+1)%3) {

        tma::arrive_and_wait(qkv_barrier, tic);  // ding! memory arrived
        __syncthreads();

        if (warpid == 0 && block < blocks-1) {
            tma::set_bytes(qkv_barrier,
                size_bytes<typeof(q_smem[0])> + 
                size_bytes<typeof(k_smem[0])> + 
                size_bytes<typeof(v_smem[0])>
            );

            int tile_idx = (blockIdx.x * blocks) + block + 1;
            tma::load_async(q_smem[toc],           tma_q, qkv_barrier, tile_idx); 
            tma::load_async(k_smem[(ring_id+2)%3], tma_k, qkv_barrier, tile_idx); 
            tma::load_async(v_smem[(ring_id+2)%3], tma_v, qkv_barrier, tile_idx);
        }

        // ----- let's do sliding window first -----
        // only warps 0-4 need to be involved in this

        rt_fl<1, 8> sliding_o;
        rt_fl<1, 4>::col_vec sliding_norm_vec;
        zero(sliding_o);
        zero(sliding_norm_vec);
        if(warpgroupid == 0) {

            rt_fl<1, 4> att_block[2];
            rt_bf<1, 4> att_block_bf[2];
            rt_fl<1, 4>::col_vec max_vec;

            neg_infty(max_vec); // zero registers for the Q chunk

            for(int subtile = 0; subtile < 2; subtile++) {
                if (block + subtile >= 1) { // ensure tile has been loaded by now.
                    warpgroup::mma_fence(att_block[subtile]);
                    warpgroup::mm_ABt(att_block[subtile], q_smem[tic], k_smem[(ring_id+subtile)%3]);
                    warpgroup::mma_commit_group();
                }
                else {
                    neg_infty(att_block[subtile]); // initial blocks must be zero
                }
            }
            warpgroup::mma_async_wait();
            // make last block causal
            #pragma unroll
            for(int j = 0; j < 4; j++) {
                auto &attn_subtile = reinterpret_cast<rt_fl_1x1<>&>(att_block[1].tiles[0][j]);
                if (j>warpid) neg_infty(attn_subtile);
                else if (j==warpid) make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty());
            }
            // now do the softmax. first we subtract max for numerical stability. then exp.
            #pragma unroll
            for(int subtile = 0; subtile < 2; subtile++) {
                mul(att_block[subtile], att_block[subtile], 0.125); // temperature adjustment.
                row_max(max_vec, att_block[subtile], max_vec); // accumulate onto the max_vec
            }
            #pragma unroll
            for(int subtile = 0; subtile < 2; subtile++) {
                sub_row(att_block[subtile], att_block[subtile], max_vec);
                exp(att_block[subtile], att_block[subtile]);
                mul(att_block[subtile], att_block[subtile], beta);
            }
            // now we sum so that we can divide (normalize) later
            #pragma unroll
            for(int subtile = 0; subtile < 2; subtile++) {
                row_sum(sliding_norm_vec, att_block[subtile], sliding_norm_vec); // incorporates beta
                copy(att_block_bf[subtile], att_block[subtile]); // cast to bf16 for next matmul
            }
            for(int subtile = 0; subtile < 2; subtile++) {
                warpgroup::mma_fence(sliding_o);
                warpgroup::mma_AB(sliding_o, att_block_bf[subtile], v_smem[(ring_id+subtile)%3]);
                warpgroup::mma_commit_group();
            }
            warpgroup::mma_async_wait();
        }
        __syncthreads();

        rt_fl<1, 8> linear_o; // this is partitioned across the two warpgroups.
        rt_fl<1, 4>::col_vec linear_norm_vec;
        zero(linear_o);
        zero(linear_norm_vec);
        // if(block >= 2) { // if not in at least the third block, no need for linear attention.

        //     // ******* linear attn ******** // 

        //     // matmul to generate linear_q before softmax

        //     rt_fl<1, 4> linear_q;
        //     rt_bf<1, 4> linear_q_bf;

        //     warpgroup::mma_fence(linear_q);
        //     warpgroup::mm_AB(linear_q, q_smem[tic], qf_map); // reset
        //     warpgroup::mma_commit_group();
        //     warpgroup::mma_async_wait(); // q is now projected-
        //     mul(linear_q, linear_q, warpgroupid ? alpha : -1.f*alpha);
        //     // now we need to run q through a local softmax to featurize
        //     softmax_featuremap_inplace(linear_q);
        //     copy(linear_q_bf, linear_q); // now to bf16

        //     // copy the local KV cache into shared memory to shared memory and do matmul
        //     warpgroup::store(kv_smem[warpgroupid], local_kv);
        //     __syncthreads(); // this should probably be a cooperative group of just the 4 warps
        //     warpgroup::mma_fence(linear_o);
        //     warpgroup::mm_AB(linear_o, linear_q_bf, kv_smem[warpgroupid]);
        //     warpgroup::mma_commit_group();
        //     warpgroup::mma_async_wait();

        //     // next we need to go figure out the norm.
        //     // first we load sum(k) from smem to registers.
        //     row_vec<rt_bf<1,4>> cumsum_k_reg;
        //     load(cumsum_k_reg, cumsum_k_smem[warpgroupid]);
        //     // now we can project this up into a register tile
        //     // we're broadcasting along the column axis (filling all rows with the same value)
        //     rt_bf<1,4> cumsum_k_reg_tile;
        //     broadcast_col(cumsum_k_reg_tile, cumsum_k_reg);
        //     // next we matmul! this gives us a tile.
        //     rt_fl_1x1<> norm_tile;
        //     zero(norm_tile);
        //     mma_ABt(norm_tile, linear_q_bf, cumsum_k_reg_tile, norm_tile);
        //     row_max(linear_norm_vec, norm_tile); // technically any column slice would work but this is EZ
        //     // ^ note this incorporates alpha since it was premultiplied onto linear_q!
            
        //     // now accumulate KV onto the matmul for the future.
        //     rt_fl<1, 4> linear_k;

        //     // matmul to generate linear_k before softmax
        //     warpgroup::mma_fence(linear_k);
        //     warpgroup::mm_AB(linear_k, k_smem[ring_id], kf_map); // reset
        //     warpgroup::mma_commit_group();
        //     warpgroup::mma_async_wait(); // q is now projected
        //     if(warpgroupid) {
        //         mul(linear_k, linear_k, -1.f);
        //     }
        //     // now we need to run q through a local softmax to featurize
        //     softmax_featuremap_inplace(linear_k);

        //     // copy the local KV cache into shared memory & do matmul
        //     warpgroup::store(k_scratch_smem[warpgroupid], linear_k); // screw it, this is now just a scratchpad.
        //     __syncthreads();
        //     warpgroup::mma_fence(local_kv);
        //     warpgroup::mma_AtB(local_kv, k_scratch_smem[warpgroupid], v_smem[tic]);
        //     warpgroup::mma_commit_group();
        //     warpgroup::mma_async_wait();

        //     cumulative_add(cumsum_k_smem[warpgroupid], k_scratch_smem[warpgroupid]);
        // }

        // next step is to sum two norm vecs
        add(sliding_norm_vec, sliding_norm_vec, linear_norm_vec);
        warpgroup::store(norm_exchange[warpgroupid], sliding_norm_vec);
        __syncthreads();
        col_vec<rt_fl_1x1<>> total_norm;
        warpgroup::load(total_norm, norm_exchange[warpgroupid^1]);
        add(total_norm, total_norm, sliding_norm_vec);
        // we have now finally accumulated the total norm for everything
        add(sliding_o, sliding_o, linear_o); // local o
        div_row(sliding_o, sliding_o, total_norm); // this half is now normalized
        if(warpgroupid == 1) {
            warpgroup::store(o_smem, sliding_o);
        }
        __syncthreads();
        if(warpgroupid == 0) {
            warpgroup::load(linear_o, o_smem);
            add(sliding_o, sliding_o, linear_o);
            warpgroup::store(o_smem, sliding_o);
        }
        __syncthreads();
        if(warpid == 0) {
            tma::store_async(tma_o, o_smem, blockIdx.x*blocks + block);
            tma::store_commit_group();
        }
        tma::store_async_wait();
    }
}

#ifdef TORCH_COMPILE
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>

void hh_lin_tk_smd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {

    CHECK_INPUT(q); 
    CHECK_INPUT(k); 
    CHECK_INPUT(v); 
    CHECK_INPUT(o); 

    auto batch = q.size(0); 
    auto heads = q.size(1); 
    auto N  = q.size(2); 

    // N must be >= 64 and a multiple of 64
    TORCH_CHECK(N >= 64, "N must be >= 64");
    TORCH_CHECK(N % 64 == 0, "N must be a multiple of 64");

    auto q_d  = q.size(3);
    auto k_d  = k.size(3);
    auto v_d  = v.size(3);
    auto o_d  = o.size(3);

    // all must be == 128
    TORCH_CHECK(q_d == k_d, "q and k must have the same dimension");
    TORCH_CHECK(q_d == v_d, "q and v must have the same dimension");
    TORCH_CHECK(q_d == o_d, "q and o must have the same dimension");
    TORCH_CHECK(q_d == 128, "q, k, v must have dimension 128");

    // auto kv_d_1 = kv.size(2);
    // auto kv_d_2 = kv.size(3);

    // // kv must be 256x128
    // TORCH_CHECK(kv_d_1 == 256, "kv must have dimension 256");
    // TORCH_CHECK(kv_d_2 == 128, "kv must have dimension 128");

    // // k must be 1 x 256
    // TORCH_CHECK(ks.size(2) == 1, "ks must have sequence length 1");
    // TORCH_CHECK(ks.size(3) == 256, "ks must have dimension 256");

    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "k must be bf16");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "v must be bf16");
    // TORCH_CHECK(kv.scalar_type() == c10::ScalarType::BFloat16, "kv must be bf16");
    // TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "ks must be bf16");
    TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "o must be bf16");

    c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
    // c10::BFloat16 *kv_ptr = kv.data_ptr<c10::BFloat16>();
    // c10::BFloat16 *ks_ptr = ks.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();

    const bf16* d_q = reinterpret_cast<const bf16*>(q_ptr); 
    const bf16* d_k = reinterpret_cast<const bf16*>(k_ptr);  
    const bf16* d_v = reinterpret_cast<const bf16*>(v_ptr);  
    // bf16* d_kv_state = reinterpret_cast<bf16*>(kv_ptr);  
    // bf16* d_k_state  = reinterpret_cast<bf16*>(ks_ptr);
    bf16* d_o = reinterpret_cast<bf16*>(o_ptr);

    CUtensorMap* tma_q_d  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>>   (d_q,        (batch*heads*N/(16 * 4)),    128/(16 * 4) ); 
    CUtensorMap* tma_k_d  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave>>(d_k,        (batch*heads*N/(16 * 4)),    128/(16 * 4) );
    CUtensorMap* tma_v_d  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave>>(d_v,        (batch*heads*N/(16 * 4)),    128/(16 * 8) );
    CUtensorMap* tma_o_d  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 8, kittens::ducks::st_layout::wgmma_swizzle>>   (d_o,        (batch*heads*N/(16 * 4)),    128/(16 * 8) );
    // CUtensorMap* tma_kv_d = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 8, kittens::ducks::st_layout::wgmma_swizzle>>   (d_kv_state, (batch*heads*256/(16 * 4)),  128/(16 * 8) );
    // CUtensorMap* tma_ks_d = tma::allocate_and_create_tensor_map<row_vec<st_bf<4, 4*4>>>                                           (d_k_state,  (batch*heads*  1/(   1  )));  

    unsigned long mem_size = kittens::MAX_SHARED_MEMORY;

    cudaFuncSetAttribute(
        hedgehog_linear_attention_smd,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    dim3 grid(batch*heads, 1, 1);

    hedgehog_linear_attention_smd<<<grid, 128, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d); 

    CHECK_CUDA_ERROR(cudaGetLastError());
}

#else
#include "harness.impl"
#endif