#include <iostream>
#include <math.h>
#include <assert.h>
#include <mma.h>
using namespace nvcuda; 

# include <cuda/pipeline>
# include <cooperative_groups.h>
# include "src/kittens.cuh"
# include "src/pyutils/torch_helpers.cuh"

#include <ATen/cuda/CUDAContext.h>  // Include necessary for getting the current stream

using namespace kittens;
typedef st_bf<4, 4> tile_4x4;
typedef st_bf<1, 4> tile_1x4;
// typedef sv_fl_row_4 vec_4;

// template<typename H>
__device__
void thread_block_load(tile_4x4 &_dst, const typename tile_4x4::dtype *_src, const int nThreads=256) {
    float4* dst = (float4*) _dst.data;
    float4* src = (float4*) _src; 
    using H = tile_4x4;
    using T = typename H::dtype;

    const int _row_stride = H::cols; // SA TODO: Confirm this is correct
    auto bytes_per_row    = H::cols * sizeof(T); // non-padded
    auto f4_stride        = (_row_stride*sizeof(T))/sizeof(float4);
    auto reads_per_row    = bytes_per_row / sizeof(float4);
    auto rows_per_block    = nThreads / reads_per_row; 
    auto row_skipping_only = (nThreads % reads_per_row) == 0; // if we read complete rows.
    auto f4_elements      = (H::num_elements * sizeof(T)) / sizeof(float4);
    
    if( row_skipping_only ) {
        auto col      = threadIdx.x % reads_per_row; // this will be fixed
        auto row_base = threadIdx.x / reads_per_row; 
        auto _stride  = f4_stride*rows_per_block; // we we will just skip!
         __syncthreads();
        auto idx = row_base*f4_stride + col;
        for(auto i = threadIdx.x; i < f4_elements; i+=nThreads, idx += _stride) {
            dst[idx] = src[i];
        }
    } else {
         __syncthreads();
        for(auto i = threadIdx.x; i < f4_elements; i+=nThreads) {
            auto col = i % reads_per_row;
            auto row = i / reads_per_row;
            dst[row*_row_stride + col] = src[i];
        }
    }
}

template<typename op>
__device__ 
void shm_broadcast(float &f, float *shm, const int workers = 4) {
    auto warpid = threadIdx.x / 32;
    auto lane   = threadIdx.x % 32;
    shm[warpid] = f;
    __syncthreads();
    if(warpid == 0) {
        if(lane == 0) {
            for(auto j = 1; j < workers; j++) {f = op::op(f,shm[j]);}
            for(auto j = 0; j < workers; j++) {shm[j] = f;}
        }
        __syncwarp();
    }
    __syncthreads();
    f = shm[warpid];
}


// GEMV
__device__
void gemv(rt_col_bf_4x1::row_vec  &o, rt_row_fl_4x1::col_vec &x, rt_row_fl_4x1::col_vec &a) { // SA: directions of these seem off
    rt_row_fl_4x1::col_vec t;
    copy(t,a);
    // The accumulator is row x column
    // a row multiply means that each row is multiplied by a column matrix.
    // So if the accumulator is 2 x 3 then 
    // mul_row_accum_rvec(t, x); // multiply vv in place with aa: a * v.unsqueeze(1)
    mul_row(t, x); // multiply vv in place with aa: a * v.unsqueeze(1)
    // row_sum(o, t); // aa.sum(0) sum across all the rows 
}

static bool __inline__ __device__ is_row_leader() { return kittens::laneid() % 4 == 0; } 
// template<typename T, typename U> __device__ inline U __typeconvert(T a);
// https://github.com/chrismre/cudatastic_trash/blob/a149e1b9d2ef591c60b5f7560155d5ef99be1a48/src/global_warp_tile/type_helper.cuh#L1C1-L11C18
// ptxas fatal   : Unresolved extern function '_Z13__typeconvertI6float2S0_ET0_T_'     

template<typename H>
static void __device__
rvec_to_vec(H *dst, const rt_row_fl_1x4::col_vec &v) {
    auto row = kittens::laneid() / 4;
    const int tile_height = 4;
    using T = rt_row_fl_1x4::dtype;
    __syncwarp();
    if(is_row_leader()) { // only the leaders write
        for(auto h = 0; h < tile_height; h++) {
            // dst[h*TILE_DIM + row]     = __typeconvert<T,H>(v[h][0]);
            // dst[h*TILE_DIM + row + 8] = __typeconvert<T,H>(v[h][1]);
        }
    }    
}

static
void __device__
vec_to_rvec(const rt_row_bf_4x1::col_vec &v, const float2 *src) {
    auto row = kittens::laneid() / 4;
    const int tile_height = 4;
    __syncwarp();    
    for(auto h = 0; h < tile_height; h++) {
        // v[h][0] = src[h*TILE_DIM + row];     
        // v[h][1] = src[h*TILE_DIM + row + 8];
    }
}


template<typename H, typename T>
__global__
void sliding_window_ker_hack(int n, int j, bool just_q, const T* __q, const T* __k, const T* __v, T* __o) {
    // ``just_q`` indicates whether q is for a single token vs. a chunk of toknens
    auto warpid = kittens::warp_id();
    const int d = 64;
    const int window_size = 64;
    const int workers = 4;
    const int threads = workers * kittens::WARP_SIZE;
    auto head_offset  = blockIdx.x * n * d;
    
    const H* _q = device_cast(__q) + (just_q ? blockIdx.x*d : head_offset);
    const H* _k = device_cast(__k) + head_offset;
    const H* _v = device_cast(__v) + head_offset;
          H* _o = device_cast(__o) + blockIdx.x*d; // just a single vector

    __shared__ tile_4x4 k,v;
    __shared__ tile_1x4::row_vec w;
    __shared__ float _max[workers], _sum[workers];

    const auto start_idx = just_q ? 0 : (j-window_size)*d;
    thread_block_load(k, _k + start_idx, threads);
    thread_block_load(v, _v + start_idx, threads);
    auto vec_idx = just_q ? 0 : j * d;

    rt_col_bf_1x4::row_vec qv; // full local copy 
    rt_row_fl_1x4::col_vec ws; 
    rt_col_bf_1x4::row_vec k_slice; // SA: Should this be float type?
    
    rt_row_bf_4x1::col_vec wv; // full local copy 
    rt_col_bf_4x1::row_vec os; // shards
    rt_col_bf_4x1::row_vec v_slice; // Each of the 4 workers stores a column

    // These are column slices of the matrix.: | K_1 | K_2 | K_3 |
    __syncthreads();
    TODO: load(qv, _q + vec_idx, d); // every warp gets a full copy of q 

    // We want a column-wise stripe of the vector. 
    // REPLACED rt1x4.tile_to_accum(k_slice, k.template subtile<1,4>(warpid, 0)); 
    // load(k_slice, k.data + warpid*kittens::TILE_DIM, d);
    
    // ********
    // The algorithm.
    // qs = [q for j in range(4)] # broadcast q to each warp
    // ks = [k[:,j*d//4:(j+1)*d//4] for j in range(4)] # shard k
    // ws = [torch.einsum("d, de->e", qs[j],ks[j]) for j in range(4)]
    zero(ws);
    // gemv(ws, qv, k_slice);

    // local_max = [ws[j].max() for j in range(4)] # compute local, then global max
    // the_max = torch.tensor(local_max).max()
    float local_max= -INFINITY;
    max(ws, local_max);
    shm_broadcast<ops::mul>(local_max, _max);
    
    // ews = [torch.exp(ws[j] - the_max) for j in range(4)]
    sub(ws, local_max);
    exp(ws);

    // es  = [ews[j].sum() for j in range(4)]
    float local_sum = 0.f;
    add(ws, local_sum);
    shm_broadcast<ops::sum>(local_sum, _sum);
    
    // w  /= the_sum
    div(ws, local_sum);

    // broadcast w back to shared memory
    // rvec_to_vec(&w.data[warpid*kittens::TILE_DIM], ws);
    __syncthreads(); // let the writes complete
    vec_to_rvec(wv, w.data); // read the *whole* v here.
    
    // we want a column stripe of V
    // TODO rt4x1.tile_to_accum(v_slice, v.template subtile<4,1>(0, warpid));
    zero(os);
    // gemv(os, wv, v_slice);
    
    // now we have a fragment of v and we write, this write is to *global* memory.
    // store(_o + warpid*kittens::TILE_DIM, os, d);
    //  argument types are: (__nv_bfloat16 *, kittens::rt<std::false_type, kittens::bf16_2, 4, 1>::row_vec, const int)      
}

void 
sliding_window(int j,   
    torch::Tensor q, torch::Tensor k, torch::Tensor v, 
    torch::Tensor o) {

    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    
    uint batch = q.size(0);
    uint head  = q.size(1);
    uint d     = q.size(3);
    TORCH_CHECK(d == 64, "Only dimension 64 implemented...");

    bool k_same = true, v_same = true;
    for(auto i = 0; i < 2; i++) { 
        k_same &= q.size(i) == k.size(i);
        v_same &= q.size(i) == v.size(i);
    }
    k_same &= d == k.size(3);
    v_same &= d == v.size(3);
    uint n     = k.size(2);
    v_same &= v.size(2) == n;

    // This is just a restriction of what we're doing now...
    TORCH_CHECK(k_same, "X and K_out should be same size");
    TORCH_CHECK(v_same, "X and V_out should be same size");
    
    const int workers = 4;
    using H = __nv_bfloat16;
    using T = c10::BFloat16;

    auto threads = workers * kittens::WARP_SIZE;

    auto stream_wrapper = at::cuda::getCurrentCUDAStream(q.device().index());
    cudaStream_t stream = stream_wrapper.stream();
    sliding_window_ker_hack<H,T><<<batch*head,threads,0,stream>>>(n, j, q.size(2) == 1,
                        q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), 
                        o.data_ptr<T>());
}

