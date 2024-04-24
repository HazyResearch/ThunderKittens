#include <iostream>
#include <math.h>
#include <assert.h>
#include <mma_AB.h>
using namespace nvcuda;

# include "src/kittens.cuh"
# include "src/common/pyutils/torch_helpers.cuh"

// **** ASYNC INCLUDE *****
#include <cuda/pipeline>
#include <cooperative_groups.h>

using namespace kittens;

// Types
typedef rt_bf<1, 1> _rtd_qk;
typedef rt_bf<1, 4> _rtd_v;
typedef rt_fl<1, 1> _rtd_qk_accum;

template <typename H, typename T>
__global__
void micro_ker(
    int n, int d, int dv, 
    const T* __q, const T* __k, const T* __v, 
    T* __o_small
) { 
    auto warpid           = kittens::warp_id();
    auto lane             = kittens::laneid();
    const int NUM_WORKERS = kittens::N_WARPS;

    const H *_q       = reinterpret_cast<const H*>(__q)+blockIdx.x*(n*d);
    const H *_k       = reinterpret_cast<const H*>(__k)+blockIdx.x*(n*d);
          H *_o_small = reinterpret_cast<H*>(__o_small)+blockIdx.x*(n*d);
    const H *_v       = reinterpret_cast<const H*>(__v)+blockIdx.x*(n*dv);
    
    /*********
    SHARED MEMORY
    **********/
    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    st_bf_1x1<ducks::st_layout::xor_swizzle> (&q)[2][NUM_WORKERS] = al.allocate<st_bf_1x1<ducks::st_layout::xor_swizzle>, 2, NUM_WORKERS>();
    st_bf_1x1<ducks::st_layout::xor_swizzle> (&k)[2][NUM_WORKERS] = al.allocate<st_bf_1x1<ducks::st_layout::xor_swizzle>, 2, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&v)[2][NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, 2, NUM_WORKERS>();
    st_bf_1x1<ducks::st_layout::xor_swizzle> (&o_small)[NUM_WORKERS] = al.allocate<st_bf_1x1<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();

    /*********
    REGISTER
    **********/
    _rtd_qk qfrag, kfrag; 
    _rtd_v vfrag;
    _rtd_qk_accum o_accum_small;
    _rtd_qk o_frag_small;
     
    // Pipeline handlers and barriers
    int tic = 0, toc = 1;
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> qkv_barrier;
    if (threadIdx.x == 0) {init(&qkv_barrier, block.size());}
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> store_barrier;
    if (threadIdx.x == 0) {init(&store_barrier, block.size());}
    block.sync(); // Make sure no gets to the barrier before its initialized.

    // constants; num elements
    const int qk_tile_elements = st_bf_1x1<ducks::st_layout::xor_swizzle>::num_elements;
    const int  v_tile_elements = st_bf_1x4<ducks::st_layout::xor_swizzle>::num_elements;
    auto n_tiles  = n/kittens::TILE_DIM;
    auto n_blocks = n_tiles/NUM_WORKERS;
    assert(n_tiles % NUM_WORKERS == 0);

    // Load in a bunch of QKV as we go.
    kittens::load_async(q[tic][warpid], _q + warpid*qk_tile_elements, d,  qkv_barrier);
    kittens::load_async(k[tic][warpid], _k + warpid*qk_tile_elements, d,  qkv_barrier);
    kittens::load_async(v[tic][warpid], _v + warpid*v_tile_elements , dv, qkv_barrier);     
    __syncthreads();

    // iterate along the sequence dimension
    for(auto cur_block = 0; cur_block < n_blocks; cur_block++, tic ^= 1, toc ^= 1) {
        zero(o_accum_small);
        zero(o_frag_small);

        qkv_barrier.arrive_and_wait(); 
        // Kick off the next block load.
        if(cur_block < n_blocks - 1) {
            auto next_idx = (cur_block + 1)*NUM_WORKERS + warpid; 
            kittens::load_async(q[toc][warpid], _q + next_idx * qk_tile_elements, d, qkv_barrier);
            kittens::load_async(k[toc][warpid], _k + next_idx * qk_tile_elements, d, qkv_barrier);
            kittens::load_async(v[toc][warpid], _v + next_idx * v_tile_elements, dv, qkv_barrier);
        } 

        // Load data to register
        load(vfrag, v[tic][warpid]);
        load(qfrag, q[tic][warpid]);
        load(kfrag, k[tic][warpid]);

        // Dot product. C = torch.einsum("nd,nd->nd", A, B)
        copy(o_frag_small, qfrag);
        mul(o_frag_small, kfrag, o_frag_small);
        copy(o_accum_small, o_frag_small);
        store(o_small[warpid], o_accum_small);
        __syncthreads(); 
        store(_o_small + (cur_block * NUM_WORKERS + warpid)*qk_tile_elements, o_small[warpid], d);
    }
}

void
micro( torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o_small ) {

    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o_small);
    
    auto batch = q.size(0);
    auto head  = q.size(1);
    auto n     = q.size(2);
    auto d     = q.size(3);
    auto dv    = v.size(3);
    bool k_same = true;
    for(auto i = 0; i < 4; i++) { 
        k_same &= q.size(i) == k.size(i);
    }
    // This is just a restriction of what we're doing now...
    TORCH_CHECK(k_same, "Q and K should be same size");
    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "Q is a Bfloat");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "K is a Bfloat");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "V is a Bfloat");

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    const int workers = 8;

    // total shared memory allocation
    unsigned long mem_size  = 2*2*workers*sizeof(st_bf_1x1<ducks::st_layout::xor_swizzle>);      // q, k    (double buff)
                  mem_size += 2*1*workers*sizeof(st_bf_1x4<ducks::st_layout::xor_swizzle>);      // v       (double buff)
                  mem_size += 1*1*workers*sizeof(st_bf_1x1<ducks::st_layout::xor_swizzle>);      // o_small (single buff)

    TORCH_CHECK(n % (workers*kittens::TILE_DIM) == 0, "The number of elements should be divisible the number of workers times stored fragments");
    
    auto threads = workers * kittens::WARP_THREADS;
    printf("[simple_compute_ker] Requesting %lu bytes of memory for %d (%d) workers\n", mem_size, workers, threads);
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
             micro_ker<H, T>,
             cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size));
    
    micro_ker<H,T><<<batch*head,threads,mem_size>>>(
        n, d, dv, 
        q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), o_small.data_ptr<T>()
    );

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}