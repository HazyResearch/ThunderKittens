#include <iostream>
#include <math.h>
#include <assert.h>
#include <mma.h>
using namespace nvcuda;

#include "../../../src/kittens.cuh"
// #include "src/common/pyutils/torch_helpers.cuh"

#include <cuda/pipeline>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
// #include <ATen/cuda/CUDAContext.h>

#define ATTN_B 128
#define ATTN_H 16  
#define ATTN_D 64
#define ATTN_DV 320

using namespace kittens;

template<typename H> __device__ float4* _f4p(H *x)  { return (float4* ) x;}
template<typename H> __device__ const float4* _f4pc(H *x)  { return (const float4* ) x;}
template<typename H> __device__ int _bank(H*x) { return ((uint64_t)x)/4 % 32; }

template <typename H, typename T>
__global__
void based_simple_ker(const T* __q, const T* __k, const T* __v, T* __kv_state, T* __k_state, T* __out){ //, T* __denom) { 

    auto block_start = blockIdx.x;
    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();
    const int workers  = 8; 
    const int nThreads = workers*kittens::WARP_THREADS;

    // Data size information
    const int kv_state_size = ATTN_D * ATTN_DV;
    const int k_state_size  = ATTN_DV;

    const H *q_g = reinterpret_cast<const H*>(__q)+block_start*ATTN_DV;
    const H *k_g = reinterpret_cast<const H*>(__k)+block_start*ATTN_DV;
    const H *v_g = reinterpret_cast<const H*>(__v)+block_start*ATTN_D;
          H *kv_state_g = reinterpret_cast<H*>(__kv_state)+block_start*kv_state_size;
          H *k_state_g  = reinterpret_cast<H*>(__k_state)+block_start*k_state_size;
          H *num_g      = reinterpret_cast<H*>(__out)+block_start*ATTN_D;

    // Setup the extended shared memory. We want to be 128 byte aligned.
    const int row_bytes   = ATTN_D * sizeof(H);
    const int align_pad   = (row_bytes & 127) == 0 ? 0 : 128 - row_bytes & 127;
    const int buffer_rows = workers * 8; // 64
    const int buffer_size = (ATTN_D + align_pad)*buffer_rows;
    assert(align_pad == 0); 

    // Use the simple shared memory 
    __shared__ alignas(alignof(float4)) H kv_state[2][buffer_size];
    __shared__ alignas(alignof(float4)) H q[ATTN_DV];
    __shared__ alignas(alignof(float4)) H k[ATTN_DV];
    __shared__ alignas(alignof(float4)) H kq[ATTN_DV];
    __shared__ alignas(alignof(float4)) H k_state[ATTN_DV];
    __shared__ alignas(alignof(float4)) H v[ATTN_D];
    __shared__ alignas(alignof(float4)) H num[ATTN_D];
    __shared__ alignas(alignof(float4)) H num_help[workers][ATTN_D];
    __shared__ alignas(alignof(float4)) H den[1];

    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    if (threadIdx.x == 0) {init(&barrier, block.size());}
     __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier_cheat;
    if (threadIdx.x == 0) {init(&barrier_cheat, block.size());}
    block.sync(); // Need to make sure none calls before setup.
    
    // How many float4s -- these are 16 bytes.
    assert(ATTN_DV*sizeof(H) % sizeof(float4) == 0);
    assert(ATTN_D*sizeof(H) % sizeof(float4) == 0);

    const auto d_state_shape = cuda::aligned_size_t<alignof(float4)>(ATTN_DV*sizeof(H)); 
    const auto d_model_shape = cuda::aligned_size_t<alignof(float4)>(ATTN_D*sizeof(H)); 

    cuda::memcpy_async(block, q, q_g, d_state_shape, barrier_cheat);
    cuda::memcpy_async(block, k, k_g, d_state_shape, barrier_cheat);
    cuda::memcpy_async(block, k_state, k_state_g, d_state_shape, barrier_cheat);
    cuda::memcpy_async(block, v, v_g, d_model_shape, barrier_cheat);

    int tic = 0;
    int toc = 1;
    
    // Read the initial buffer slice of kv_state
    // const auto buffer_row_shape = cuda::aligned_size_t<alignof(float4)>(buffer_rows*ATTN_D*sizeof(H)); 
    // cuda::memcpy_async(block, kv_state[tic], kv_state_g, buffer_row_shape, barrier);

    // Sum k to kstate and do kv. k_state += k 
    barrier_cheat.arrive_and_wait(); // Make sure q,k,v have arrived.
    for(auto i = threadIdx.x; i < ATTN_DV; i+=nThreads) { 
        k_state[i] += k[i]; 
    }
    __syncwarp();
    
    // den = torch.einsum("f,f->1", q, k_state) + eps;
    for(auto i = threadIdx.x; i < ATTN_DV; i+=nThreads) { 
        // kq[i] = __typeconvert<float,H>(0.f);
        kq[i] = (q[i]*k_state[i] + __float2bfloat16(0.0000000001f)); 
    }
    if (warpid == 0 && lane == 0) { 
        den[0] = __float2bfloat16(0.f);
        for (auto i = 0; i < ATTN_DV; i++) { den[0] += kq[i]; }
    }
    __syncwarp();
    

    // Store v across threads for the next phase; 
    // Assumes ATTN_D fits in register and is a multiple of 32
    register H v_vals[ATTN_D / kittens::WARP_THREADS]; // 64 / 32 = 2
    register H num_vals[ATTN_D / kittens::WARP_THREADS];

    auto j0 = 0;
    for(auto j = lane; j < ATTN_D; j+=kittens::WARP_THREADS, ++j0) {
        v_vals[j0]   = v[j];
        num_vals[j0] = __float2bfloat16(0.f);
    }

    auto outer_batches = ATTN_DV / buffer_rows;
    auto extra_batch   = ATTN_DV % buffer_rows;
    
    const int bytes_per_batch = buffer_rows*ATTN_D*sizeof(H);  
    const int extra_bytes     = extra_batch*ATTN_D*sizeof(H);
    const auto batch_shape    = cuda::aligned_size_t<alignof(float4)>(bytes_per_batch); 
    
    // // Each row is of length ATTN_D
    // auto total_batches = (extra_batch > 0) ? outer_batches + 1 : outer_batches; 
    // auto _buffer_rows  = buffer_rows; // will change if extra batches!
    // // We iterate through each batch.
    // // * We load the next batch asynchronously (into toc)
    // // * We work on the current batch.
    // // There is some cleanup about the extra batch.
    // for(auto ob = 0; ob < total_batches; ob++, tic ^=1, toc ^=1) {
    //     auto cur_batch_idx  = ob       * buffer_rows * ATTN_D;
    //     auto next_batch_idx = (ob + 1) * buffer_rows * ATTN_D;

    //     barrier.arrive_and_wait(); // wait on the work buffer to be free
    //     if(ob + 1 < outer_batches) { // if there is more work fetch the next one.
    //         cuda::memcpy_async(
    //             block, kv_state[toc], 
    //             kv_state_g + next_batch_idx, 
    //             batch_shape , barrier
    //         );
    //     } else { // Last batch!
    //         if(extra_batch > 0) {
    //             // NOTE: ATTN_D must be aligned for this to be true 
    //             const auto extra_batch_shape = cuda::aligned_size_t<alignof(float4)>(extra_bytes); 
    //             cuda::memcpy_async(
    //                 block, kv_state[toc], 
    //                 kv_state_g + next_batch_idx, 
    //                 extra_batch_shape, barrier
    //             ); 
    //         }
    //     }
    //     _buffer_rows = (ob == outer_batches) ? extra_batch : buffer_rows;
        
    //     // Here we compute our chunk of: kv_state += torch.einsum("f,d->df", k, v)
    //     // We also compute: num = torch.einsum("f,df->d", q, kv_state)
    //     __syncwarp();
    //     for(auto i = warpid; i < _buffer_rows; i += workers) {
    //         H k_val = k[ob*buffer_rows + i]; // Broadcast this value to each thread in the warp.
    //         H q_val = q[ob*buffer_rows + i];
    //         auto p_kvs = kv_state[tic] + i*ATTN_D; // pointer to the row
    //         auto j0 = 0;
    //         for(auto j = lane; j < ATTN_D; j += kittens::WARP_THREADS, j0++) { 
    //             p_kvs[j]     += k_val*v_vals[j0];
    //             num_vals[j0] += q_val*p_kvs[j];    
    //         }
    //     }

    //     __syncthreads(); // Make sure the work is complete, then write back the buffered rows of kv_state
    //     auto _stores = (ob == outer_batches) ? (extra_bytes/sizeof(float4)) : (bytes_per_batch/sizeof(float4)); // 64 * 64 = 4096
    //     for(auto j = threadIdx.x; j < _stores; j+=nThreads) {
    //         _f4p(kv_state_g + cur_batch_idx)[j] = _f4p(kv_state[tic])[j];
    //     }
        
    //     __syncthreads(); // Write back k_state
    //     for(auto i = threadIdx.x; i < ATTN_DV*sizeof(H)/sizeof(float4); i+=nThreads) {
    //         _f4p(k_state_g)[i] = _f4p(k_state)[i];
    //     }

    // }
    
    // At the end of the loop, the threads hold fragments (shared on j and i) for q*KV i num_vals[j0]. 
    // We need to aggregate across the warps.
    __syncwarp();
    j0 = 0;
    for(auto j = lane; j < ATTN_D; j+= kittens::WARP_THREADS, ++j0) {
        num_help[warpid][j] = num_vals[j0];
    }

    __syncthreads(); // Num_help is done
    for(auto j = threadIdx.x; j < ATTN_D; j+=nThreads) {
        H nj = num_help[0][j];
        #pragma unroll
        for(auto w = 1; w < workers; w++) { nj += num_help[w][j]; }
        num[j] = nj / den[0];
    }

    __syncthreads();
    for(auto j = threadIdx.x; j < (ATTN_D * sizeof(H))/sizeof(float4); j+=nThreads) {
        _f4p(num_g)[j] = _f4p(num)[j];
    }
}

#include "harness.impl"

// void 
// based_step(torch::Tensor q, torch::Tensor k, torch::Tensor v, 
//             torch::Tensor kv_state, torch::Tensor k_state, torch::Tensor out, torch::Tensor denom) {
//     CHECK_INPUT(q);
//     CHECK_INPUT(k);
//     CHECK_INPUT(v);
//     CHECK_INPUT(kv_state);
//     CHECK_INPUT(k_state);
//     CHECK_INPUT(out);

//     int batch = q.size(0);
//     TORCH_CHECK(batch == k.size(0) && batch == v.size(0) && batch == kv_state.size(0) && k_state.size(0) == batch && out.size(0) == batch, "Differing batch sizes?");
//     TORCH_CHECK(q.size(1) == ATTN_DV, "Q is ATTN_DV?");
//     TORCH_CHECK(k.size(1) == ATTN_DV, "K is ATTN_DV?");
//     TORCH_CHECK(v.size(1) == ATTN_D, "V is ATTN_D?");
//     TORCH_CHECK(kv_state.size(1) == ATTN_DV && kv_state.size(2) == ATTN_D);
//     TORCH_CHECK(k_state.size(1) == ATTN_DV, "k_state is ATTN_DV");
//     TORCH_CHECK(out.size(1) == ATTN_D, "out is ATTN_D (the size of v)");

//     const int workers = 8;
//     using H = __nv_bfloat16;
//     using T = c10::BFloat16;
//     int threads = workers * kittens::WARP_THREADS;
//     printf("[based_inference] Requesting %d threads for %d batches\n", threads, batch); 
//     auto stream_wrapper = at::cuda::getCurrentCUDAStream(q.device().index());
//     cudaStream_t stream = stream_wrapper.stream();
//     based_simple_ker<H,T><<<batch,threads,0,stream>>>(
//                 q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(),
//                 kv_state.data_ptr<T>(), k_state.data_ptr<T>(),
//                 out.data_ptr<T>(), denom.data_ptr<T>());
// }