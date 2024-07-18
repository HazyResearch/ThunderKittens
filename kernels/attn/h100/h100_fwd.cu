#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

#define NUM_WORKERS (12)
#define NUM_WARPGROUPS (NUM_WORKERS/(kittens::WARPGROUP_WARPS))
#define NUM_BLOCKS (1)

using namespace kittens;

using layout_q = wgmma_swizzle_l; 
using layout_k = wgmma_swizzle_l;
using layout_v = wgmma_interleave_l;
using layout_o = swizzle_l;

template<int D> struct fwd_attend_ker_tile_dims {
    constexpr static int tile_width = D/kittens::TILE_DIM;
    constexpr static int qo_height  = 4;
    constexpr static int kv_height  = 1024/D;
};

#define RED  "\033[91m" 
#define GREEN  "\033[92m" 
#define YELLOW  "\033[93m" 
#define BLUE  "\033[94m" 
#define MAGENTA  "\033[95m" 
#define CYAN  "\033[96m" 
#define WHITE  "\033[97m" 
#define RESET  "\033[0m" 

template<typename... Args> __device__ void gprintf(Args... args) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x % 32 == 0) {
        printf(args...);
    }
}

template<int D>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, NUM_BLOCKS)
void fwd_attend_ker_dim(int N, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, CUtensorMap* tma_o) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    constexpr int tile_width = fwd_attend_ker_tile_dims<D>::tile_width; // constants
    constexpr int qo_height  = fwd_attend_ker_tile_dims<D>::qo_height;
    constexpr int kv_height  = fwd_attend_ker_tile_dims<D>::kv_height;

    using q_tile = st_bf<qo_height, tile_width, layout_q>; // 64 * (64 | 128) * 2 = (8192 | 16384) per warpgroup
    using k_tile = st_bf<kv_height, tile_width, layout_k>; // (256 | 128) * (64 | 128) * 2 = 32768 per pipeline stage
    using v_tile = st_bf<kv_height, tile_width, layout_v>; // (256 | 128) * (64 | 128) * 2 = 32768 per pipeline stage
    using o_tile = st_bf<qo_height, tile_width, layout_o>; // overwrites existing memory so irrelevant

    q_tile (&q_smem)[NUM_WARPGROUPS] = al.allocate<q_tile, NUM_WARPGROUPS>();
    k_tile (&k_smem)[2]              = al.allocate<k_tile, 2             >();
    v_tile (&v_smem)[2]              = al.allocate<v_tile, 2             >();

    int tic = 0, toc = 1;
 
    rt_fl<1, kv_height> att_block;
    rt_bf<1, kv_height> att_block_mma;
    rt_fl<1, tile_width> o_prev;
    col_vec<rt_fl<1, kv_height>> max_vec_last, max_vec;
    col_vec<rt_fl<1, kv_height>> norm_vec_last, norm_vec;

    int kv_blocks = N / (k_tile::rows);

    __shared__ uint64_t qsmem_barrier, kvsmem_barrier;
    __shared__ uint32_t producer_barrier, consumer_barriers[2];

    if (threadIdx.x == 0) {
        tma::init_barrier<q_tile, NUM_WARPGROUPS>(qsmem_barrier, 1);
        tma::init_barrier<k_tile, 2>(kvsmem_barrier, 1);
    }

    if (threadIdx.x == 0) {
        for (int wg = 0; wg < NUM_WARPGROUPS; wg++) { // load q
            int q_tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + wg;
            tma::load_async((q_smem[wg]), tma_q, qsmem_barrier, q_tile_idx); 
        }
        int kv_tile_idx = (blockIdx.y * kv_blocks) + 0; 
        tma::load_async((k_smem[0]), tma_k, kvsmem_barrier, kv_tile_idx); 
        tma::load_async((v_smem[0]), tma_v, kvsmem_barrier, kv_tile_idx);
        producer_barrier = 0; // nobody is done with them yet
        consumer_barriers[0] = 0; // read to use on kv iter 0 (launch has been issued)
        consumer_barriers[1] = 0; // not ready to use -- launch has not been issued.
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);
    __syncthreads();

    tma::arrive_and_wait(qsmem_barrier, 0);

    // premultiply by tempreature
    if constexpr (D == 64) { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f)); }
    else { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.08838834764f)); }

    for (auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic=tic^1, toc=toc^1) {

        while(consumer_barriers[tic] != kv_idx) __nanosleep(10); // has not been reset
        
        tma::arrive_and_wait(kvsmem_barrier, tic);
        
        if(threadIdx.x%32 == 0) {
            int producer_barrier_old = atomicInc(&producer_barrier, NUM_WORKERS-1);
            if (producer_barrier_old == NUM_WORKERS-1) { // last one to do it?
                tma::set_bytes(kvsmem_barrier, 2  * size_bytes<k_tile>);
                consumer_barriers[toc] = kv_idx+1; // also reset the tma barrier, we are now allowed to hit the next one
                
                if (kv_idx + 1 < kv_blocks) {    
                    int tile_idx = (blockIdx.y * kv_blocks) + (kv_idx + 1);
                    tma::load_async(k_smem[toc], tma_k, kvsmem_barrier, tile_idx); 
                    tma::load_async(v_smem[toc], tma_v, kvsmem_barrier, tile_idx);
                }
            }
        }
        
        warpgroup::mma_fence(att_block);
        warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[tic]);
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
        warpgroup::mma_AB(o_prev, att_block_mma, v_smem[tic]);
        warpgroup::mma_commit_group();
        warpgroup::mma_async_wait();
    }

    auto (*o_smem) = reinterpret_cast<o_tile(*)>(q_smem); // reuse q memory
    warpgroup::store(o_smem[warpgroupid], o_prev); 
    __syncthreads();
    
    if (warpid % 4 == 0) { // store o
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid;
        tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    tma::store_async_wait();
}

// #include "common/pyutils/torch_helpers.cuh"
// void attention_inference_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {

//     // get general parameters to check
//     TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q,k,v must have 4 dimensions (B,H,N,D)");
//     auto batch = q.size(0);
//     auto heads = q.size(1);
//     auto N = q.size(2);
//     TORCH_CHECK(N>0 && N%128 == 0, "N must be a multiple of 128");
//     auto D = q.size(3);
//     TORCH_CHECK(D == 128 || D == 64, "Only head dims of 64 or 128 are supported");

//     // check K, V, O dimensions, too.
//     TORCH_CHECK(k.size(0) == batch && k.size(1) == heads && v.size(2) == N && k.size(3) == D, "k must be (B,H,N,128)");
//     TORCH_CHECK(v.size(0) == batch && v.size(1) == heads && v.size(2) == N && v.size(3) == D, "v must be (B,H,N,128)");
//     TORCH_CHECK(o.size(0) == batch && o.size(1) == heads && o.size(2) == N && o.size(3) == D, "o must be (B,H,N,128)");

//     CHECK_INPUT(q);
//     CHECK_INPUT(k);
//     CHECK_INPUT(v);
//     CHECK_INPUT(o);
//     TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
//     TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bf16");
//     TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bf16");
//     TORCH_CHECK(o.scalar_type() == torch::kBFloat16, "o must be bf16");

//     // make sure D = 64 or 128
//     TORCH_CHECK(D == 64 | D == 128, "Currently, only D = 64 or 128 is supported");

//     // convert to bf16
//     c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();

//     const bf16* q_bf = reinterpret_cast<const bf16*>(q_ptr);
//     const bf16* k_bf = reinterpret_cast<const bf16*>(k_ptr);
//     const bf16* v_bf = reinterpret_cast<const bf16*>(v_ptr);
//     bf16* o_bf = reinterpret_cast<bf16*>(o_ptr);

//     auto threads = NUM_WORKERS * kittens::WARP_THREADS;
//     if (D == 64) {

//         CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width, layout_q>>(q_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<64>::qo_height * 16));
//         CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width, layout_k>>(k_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<64>::kv_height * 16));
//         CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width, layout_v>>(v_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<64>::kv_height * 16));
//         CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width, layout_o>>(o_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<64>::qo_height * 16));

//         unsigned long mem_size = 112000;
//         cudaFuncSetAttribute(fwd_attend_ker_dim<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

//         dim3 grid(N/(NUM_WORKERS*kittens::TILE_DIM), batch*heads, 1);

//         fwd_attend_ker_dim<64><<<grid, threads, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d);
//     }
//     else {
//         CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width, layout_q>>(q_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<128>::qo_height * 16));
//         CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width, layout_k>>(k_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<128>::kv_height * 16));
//         CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width, layout_v>>(v_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<128>::kv_height * 16));
//         CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width, layout_o>>(o_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<128>::qo_height * 16));

//         unsigned long mem_size = 112000;
//         cudaFuncSetAttribute(fwd_attend_ker_dim<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

//         dim3 grid(N/(NUM_WORKERS*kittens::TILE_DIM), batch*heads, 1);

//         fwd_attend_ker_dim<128><<<grid, threads, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d);
//     }
    
//     CHECK_CUDA_ERROR(cudaGetLastError());
// }

#include "harness.impl"