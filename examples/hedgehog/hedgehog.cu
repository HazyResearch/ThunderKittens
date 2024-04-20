# include "src/kittens.cuh"
#include <cuda/pipeline>

#define NUM_WORKERS (16) // hardcoded, don't change
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define D_QK (256) // hardcoded, don't change
#define D_VO (64) // hardcoded but can be changed with some effort

using namespace kittens;

using layout = kittens::ducks::st_layout::xor_swizzle;

// sum of an array of tiles -- in fp32 to preserve maximal accuracy
template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void tile_reduce(ST &dst, const ST (&src)[N_TILES]) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;
    constexpr int RESPONSIBLE_ELEMENTS = (ST::num_elements+STRIDE-1) / STRIDE; // we know in advance this divides evenly.
    float acc[RESPONSIBLE_ELEMENTS];
    #pragma unroll
    for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
        int idx = threadIdx.x + j*STRIDE;
        if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) acc[j] = __bfloat162float(dst.data[idx]); // start
    }
    // then propagate accumulation through
    for(int i = 0; i < N_TILES; i++) {
        #pragma unroll
        for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
            int idx = threadIdx.x + j*STRIDE;
            if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) acc[j] += __bfloat162float(src[i].data[idx]); // accumulate
        }
    }
    #pragma unroll
    for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
        int idx = threadIdx.x + j*STRIDE;
        if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) dst.data[idx] = acc[j]; // set
    }
}
// alternatively, sum onto the FIRST tile -- needed by attention.
template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void tile_reduce(ST (&dst)[N_TILES]) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;
    constexpr int RESPONSIBLE_ELEMENTS = (ST::num_elements+STRIDE-1) / STRIDE; // we know in advance this divides evenly.
    float acc[RESPONSIBLE_ELEMENTS];
    #pragma unroll
    for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
        int idx = threadIdx.x + j*STRIDE;
        if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) acc[j] = __bfloat162float(dst[0].data[idx]); // start
    }
    // then propagate accumulation through
    for(int i = 1; i < N_TILES; i++) {
        #pragma unroll
        for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
            int idx = threadIdx.x + j*STRIDE;
            if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) acc[j] += __bfloat162float(dst[i].data[idx]); // accumulate
        }
    }
    #pragma unroll
    for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
        int idx = threadIdx.x + j*STRIDE;
        if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) dst[0].data[idx] = acc[j]; // set
    }
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void hedgehog(int n, const bf16* __q, const bf16* __k, const bf16* __v, bf16* __o) {

    using G = kittens::group<NUM_WORKERS>;

    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    const bf16 *q_g   = reinterpret_cast<const bf16*>(__q)+blockIdx.x*(n*D_QK);
    const bf16 *k_g   = reinterpret_cast<const bf16*>(__k)+blockIdx.x*(n*D_QK);
    const bf16 *v_g   = reinterpret_cast<const bf16*>(__v)+blockIdx.x*(n*D_VO);
          bf16 *o_g   = reinterpret_cast<bf16*>      (__o)+blockIdx.x*(n*D_VO);

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    using QK_BLOCK = st_bf_1x1<layout>;
    using VO_BLOCK = st_bf_1x4<layout>;
    QK_BLOCK (&q_s)[2][NUM_WORKERS] = al.allocate<QK_BLOCK, 2, NUM_WORKERS>(); // 2 * 8192 bytes -- 16x256
    QK_BLOCK (&k_s)[2][NUM_WORKERS] = al.allocate<QK_BLOCK, 2, NUM_WORKERS>(); // 2 * 8192 bytes -- 16x256
    VO_BLOCK (&v_s)[2]              = al.allocate<VO_BLOCK, 2>(); // 2 * 2048 bytes
    VO_BLOCK (&o_s)[2]              = al.allocate<VO_BLOCK, 2>(); // 2 * 2048 bytes

    // att_accumulate is not actually a QK block, even if it happens to be the same type here.
    st_bf_1x1<layout> (&att_accumulate)[NUM_WORKERS] = al.allocate<st_bf_1x1<layout>, NUM_WORKERS>(); // 8192 bytes -- 16x256
    VO_BLOCK          (&kv_accumulate) [NUM_WORKERS] = al.allocate<VO_BLOCK,          NUM_WORKERS>(); // 32768 bytes -- 16x(16x64)

    rt_fl_1x4 kv_state; // kv state gets propagated through here, split among all 16 workers.

    zero(kv_state); // everyone zeroes their part of the kv state.

    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> qkv_barrier;
    if (threadIdx.x == 0) {init(&qkv_barrier, NUM_THREADS);}
    __syncthreads();
    load_async(q_s[0][warpid], q_g + warpid*QK_BLOCK::cols, D_QK, qkv_barrier);
    load_async(k_s[0][warpid], k_g + warpid*QK_BLOCK::cols, D_QK, qkv_barrier);
    G::load_async(v_s[0],      v_g, D_VO, qkv_barrier); // just collaboratively load v

    int n_blocks = n / kittens::TILE_DIM;

    int tic = 0, toc = 1;
    for(int block = 0; block < n_blocks; block++, tic^=1, toc^=1) {
        rt_bf_1x1<> q, k, local_attn_bf;
        rt_fl_1x1<> local_attn;
        rt_bf_1x4<> v;
        rt_fl_1x4<> o;

        // load new q, k, v into shared memory and zero o -- collaboratively, across the whole group
        // (the reason to do this is to fill larger transactions.)
        qkv_barrier.arrive_and_wait();
        if(block+1 < n_blocks) {
            load_async(q_s[toc][warpid], q_g + (block+1)*NUM_WORKERS*QK_BLOCK::num_elements + warpid*QK_BLOCK::cols, D_QK, qkv_barrier);
            load_async(k_s[toc][warpid], k_g + (block+1)*NUM_WORKERS*QK_BLOCK::num_elements + warpid*QK_BLOCK::cols, D_QK, qkv_barrier);
            G::load_async(v_s[toc],      v_g + (block+1)*VO_BLOCK::num_elements, D_VO, qkv_barrier); // just collaboratively load v
        }

        load(q, q_s[tic][warpid]);
        load(k, k_s[tic][warpid]);
        zero(local_attn);
        dot(local_attn, q, k, local_attn);
        store(att_accumulate[warpid], local_attn);
        // sum up local attention
        __syncthreads();
        tile_reduce<NUM_WORKERS>(att_accumulate); // now sum is in the first element.
        __syncthreads();
        load(v, v_s[tic]); // everyone needs v
        auto &v_col = swap_layout_inplace(v); // prepare for MMA
        if(warpid == 0) {
            load(local_attn_bf, att_accumulate[0]);
            make_causal(local_attn_bf, local_attn_bf);
            zero(o);
            mma(o, local_attn_bf, v_col, o); // causal bit.
            store(o_s[tic], o);
            // we have now taken care of the current attention block
        }

        // now we use the previous recurrent KV state to finish o_s[tic]
        rt_bf_1x4<> kv_bf;
        copy(kv_bf, kv_state);
        auto &kv_bf_col = swap_layout_inplace(kv_bf);
        zero(o);
        mma(o, q, kv_bf_col, o);
        store(kv_accumulate[warpid], o);
        __syncthreads();
        tile_reduce<NUM_WORKERS>(o_s[tic], kv_accumulate); // sum onto o_s.

        // we've now successfully compute o_s[tic] -- we can store it.
        __syncthreads();
        G::store(o_g + block*VO_BLOCK::num_elements, o_s[tic], D_VO);

        // finally we need to update the kv state for future iterations
        auto &kt = transpose_inplace(k); // k is now transposed! k has been invalidated; there is only kt.
        mma(kv_state, kt, v_col, kv_state);
    }
}

#include "harness.impl"