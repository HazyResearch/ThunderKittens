#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <tuple>

#ifdef TORCH_COMPILE
#define TK_COMPILE_BASED
#endif

#define NUM_WORKERS  (16)
#define ACTIVE_TILES (8)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define D_QK (16)
#define D_VO (64)

using namespace kittens;

struct based_globals { 
    // shapes    
    static constexpr int dv = 64;
    static constexpr int fd = 16;

    using q_tile = st_bf<16, fd>;
    using k_tile = st_bf<16, fd>;
    using v_tile = st_bf<16, dv>;
    using o_tile = st_bf<16, dv>;

    // global layouts
    using q_gl     = gl<bf16,  -1, -1, -1, fd, q_tile>;
    using k_gl     = gl<bf16,  -1, -1, -1, fd, k_tile>;
    using v_gl     = gl<bf16,  -1, -1, -1, dv, v_tile>;
    using o_gl     = gl<bf16,  -1, -1, -1, dv, o_tile>;

    // pointers
    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;

    int n;
};

template<kittens::ducks::st::all ST, int N_TILES>
__device__ void accumulate_a0(ST (&o)[N_TILES], sv_fl<ST::cols> &running_sum, const ST (&v)[N_TILES]) {
    float acc;

    if(threadIdx.x < ST::cols) {
        int col = threadIdx.x;
        acc = running_sum[col]; 
        #pragma unroll
        for(int t = 0; t < N_TILES; t++) {
            #pragma unroll
            for(int i = 0; i < ST::rows; i++) {
                acc += __bfloat162float(v[t][int2{i, col}]);
                o[t][int2{i, col}] += __float2bfloat16(acc);
            }
        }
        running_sum[col] = acc;
    }
}

template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void cumsum_inplace(ST (&x)[N_TILES], int total_block_idx) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;

    for(int i = 1; i < N_TILES; i++) {
        #pragma unroll
        for(int j = threadIdx.x; j < ST::num_elements; j+=STRIDE) {
            x[(total_block_idx+i)%N_TILES].data[j] += x[(total_block_idx+i-1)%N_TILES].data[j];
        }
    }
}


template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void tile_reduce(ST &dst, const ST (&src)[N_TILES]) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;
    constexpr int RESPONSIBLE_ELEMENTS = ST::num_elements / STRIDE;
    float acc[RESPONSIBLE_ELEMENTS];
    #pragma unroll
    for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
        int idx = threadIdx.x + j*STRIDE;
        acc[j] = __bfloat162float(dst.data[idx]);
    }

    for(int i = 0; i < N_TILES; i++) {
        #pragma unroll
        for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
            int idx = threadIdx.x + j*STRIDE;
            acc[j] += __bfloat162float(src[i].data[idx]); 
        }
    }
    #pragma unroll
    for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
        int idx = threadIdx.x + j*STRIDE;
        dst.data[idx] = __float2bfloat16(acc[j]);
    }
}

__device__ static void mul_slice(rt_bf<16, 16> &reg) {

    const int target_col = kittens::warpid();
    const int lane       = kittens::laneid();
    
    #pragma unroll
    for(int row_offset = 0; row_offset < 2; row_offset++) {
        const int dst_row = row_offset*8 + lane / 4;
        const int src_thread = (lane / 4)*4 + (target_col%8)/2;
        const int col_offset = target_col >= 8;
        bf16_2 src_val = reg.tiles[0][0].data[2*col_offset + row_offset];
        bf16 val = __shfl_sync(kittens::MASK_ALL, (target_col%2 == 0) ? src_val.x : src_val.y, src_thread);

        val *= __float2bfloat16(0.70710678118);

        reg.tiles[0][0].data[row_offset] *= bf16_2{val, val};
        reg.tiles[0][0].data[row_offset+2] *= bf16_2{val, val};
    }
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void based_linear_attention(const __grid_constant__ based_globals g) {

    const int batch = blockIdx.y;
    const int head  = blockIdx.x;

    int laneid = kittens::laneid(); 
    int warpid = kittens::warpid(); 

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);

    st_bf<16,16> (&q_s)[ACTIVE_TILES]   = al.allocate<st_bf<16,16>, ACTIVE_TILES>();
    st_bf<16,16> (&k_s)[ACTIVE_TILES]   = al.allocate<st_bf<16,16>, ACTIVE_TILES>();
    st_bf<16,64> (&v_s)[ACTIVE_TILES]   = al.allocate<st_bf<16,64>, ACTIVE_TILES>();
    st_bf<16,64> (&o_s)[ACTIVE_TILES]   = al.allocate<st_bf<16,64>, ACTIVE_TILES>();

    st_bf<16, 64> (&a1_s)[ACTIVE_TILES + 1]  = al.allocate<st_bf<16, 64>, ACTIVE_TILES + 1>();
    st_bf<16, 64> (&a2_o_accum)[NUM_WORKERS] = al.allocate<st_bf<16, 64>, NUM_WORKERS>();

    int total_block_idx = 0; 

    rt_fl<16,64> a2; 

    sv_fl<64> &a0_total = al.allocate<sv_fl<64>>();

    if (warpid == 0) {
        kittens::warp::zero(a0_total);
    }
    if (warpid < ACTIVE_TILES + 1) {
        kittens::warp::zero(a1_s[warpid]);
    }
    kittens::warp::zero(a2); 

    int n_blocks = g.n / (ACTIVE_TILES * kittens::TILE_ROW_DIM<bf16>);

    for (int block = 0; block < n_blocks; block++) {
        rt_bf<16, 16> q, k, local_attn_bf; 
        rt_fl<16, 16> local_attn, temp_attn_accum;
        rt_bf<16, 64> v; 
        rt_fl<16, 64> o, accum; 

        int cur_idx;
        if(warpid < ACTIVE_TILES) {
            cur_idx = block*ACTIVE_TILES + warpid;
            kittens::warp::load(q_s[warpid], g.q, {batch, head, cur_idx, 0});
            kittens::warp::load(k_s[warpid], g.k, {batch, head, cur_idx, 0});
        }
        else {
            cur_idx = block*ACTIVE_TILES + warpid - ACTIVE_TILES;
            kittens::warp::load(v_s[warpid-8], g.v, {batch, head, cur_idx, 0});
        }
        __syncthreads();

        if(warpid < ACTIVE_TILES) {
            kittens::warp::load(q, q_s[warpid]);
            kittens::warp::load(k, k_s[warpid]);

            kittens::warp::zero(local_attn);
            kittens::warp::mma_ABt(local_attn, q, k, local_attn);

            kittens::warp::copy(temp_attn_accum, local_attn);

            kittens::warp::mul(temp_attn_accum, temp_attn_accum, temp_attn_accum);
            kittens::warp::mul(temp_attn_accum, temp_attn_accum, 0.5f); 
            kittens::warp::add(temp_attn_accum, temp_attn_accum, local_attn);

            kittens::warp::copy(local_attn_bf, temp_attn_accum);
            kittens::warp::apply(local_attn_bf, local_attn_bf, [](int row, int col, float val) { return row >= col ? val : 0.0f; });

            kittens::warp::load(v, v_s[warpid]);
            auto &v_col = kittens::warp::swap_layout_inplace(v);

            kittens::warp::zero(o);
            kittens::warp::mma_AB(o, local_attn_bf, v_col, o);

            kittens::warp::zero(accum);
            auto &kt = kittens::warp::transpose_inplace(k);
            kittens::warp::mma_AB(accum, kt, v_col, accum);
            kittens::warp::store(a1_s[(total_block_idx+warpid+1)%(ACTIVE_TILES+1)], accum);
        }

        __syncthreads();
        cumsum_inplace<NUM_WORKERS>(a1_s, total_block_idx);
        __syncthreads();

        if(warpid < ACTIVE_TILES) {
            rt_bf<16, 64> a1;
            kittens::warp::load(q, q_s[warpid]); 
            kittens::warp::load(a1, a1_s[(total_block_idx+warpid)%(ACTIVE_TILES+1)]);
            auto &a1_col = kittens::warp::swap_layout_inplace(a1);
            kittens::warp::mma_AB(o, q, a1_col, o); 
            kittens::warp::store(o_s[warpid], o);
        }
        total_block_idx = (total_block_idx+ACTIVE_TILES)%(ACTIVE_TILES+1); // count backwards on the ring
        __syncthreads();

        for(int t = 0; t < ACTIVE_TILES; t++) {
            kittens::warp::load(q, q_s[t]);
            mul_slice(q);

            rt_bf<16, 64> a2_bf;
            kittens::warp::copy(a2_bf, a2);
            auto &a2_bf_col = kittens::warp::swap_layout_inplace(a2_bf);
            kittens::warp::zero(o);
            kittens::warp::mma_AB(o, q, a2_bf_col, o);

            kittens::warp::load(k, k_s[t]);
            mul_slice(k);
            auto &kt = kittens::warp::transpose_inplace(k); 

            kittens::warp::load(v, v_s[t]);
            auto &v_col = kittens::warp::swap_layout_inplace(v);
            kittens::warp::mma_AB(a2, kt, v_col, a2);

            kittens::warp::store(a2_o_accum[warpid], o);

            __syncthreads();
            tile_reduce<NUM_WORKERS>(o_s[t], a2_o_accum);
            __syncthreads();
        }

        accumulate_a0(o_s, a0_total, v_s);
        __syncthreads();

        if(warpid < ACTIVE_TILES) {
            kittens::warp::store(g.o, o_s[warpid], {batch, head, cur_idx, 0});
        }
        __syncthreads();
    }

}

based_globals based_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    int ATTN_B, int ATTN_H, int ATTN_N
) {
    // global pointers
    int ATTN_D = 64; 
    int ATTN_D_SMALL = 16;

    using globals = based_globals;

    using q_tile     = globals::q_tile;
    using k_tile     = globals::k_tile;
    using v_tile     = globals::v_tile;
    using o_tile     = globals::o_tile;

    // global layouts
    using q_gl     = globals::q_gl;
    using k_gl     = globals::k_gl;
    using v_gl     = globals::v_gl;
    using o_gl     = globals::o_gl;

    q_gl     q_arg{d_q, ATTN_B, ATTN_H, ATTN_N, nullptr};
    k_gl     k_arg{d_k, ATTN_B, ATTN_H, ATTN_N, nullptr};
    v_gl     v_arg{d_v, ATTN_B, ATTN_H, ATTN_N, nullptr};
    o_gl     o_arg{d_o, ATTN_B, ATTN_H, ATTN_N, nullptr};

    globals g{
        q_arg, k_arg, v_arg, o_arg, ATTN_N
    };
    return g;
}

#ifdef TK_COMPILE_BASED
#include "pyutils/torch_helpers.cuh"
#include <iostream>
void dispatch_based( 
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    int ATTN_B, int ATTN_H, int ATTN_N
){
    based_globals g = based_init(
        d_q, d_k, d_v, d_o,
        ATTN_B, ATTN_H, ATTN_N
    );

    // launch
    unsigned long mem_size = 100000; // 4090
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(
        based_linear_attention,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(ATTN_H, ATTN_B);
    based_linear_attention<<<grid,NUM_THREADS,mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

std::tuple<torch::Tensor, torch::Tensor> based(
    const torch::Tensor q, 
    const torch::Tensor k,
    const torch::Tensor v
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    int B = q.size(0);
    int H = q.size(1);
    int DV = v.size(3);
    int N  = q.size(2);
    int FD = k.size(3);

    // checks
    TORCH_CHECK(k.size(0) == B, "k batch?");
    TORCH_CHECK(k.size(1) == H, "k heads?");
    TORCH_CHECK(k.size(2) == N, "k length?");

    TORCH_CHECK(v.size(0) == B, "v batch?");
    TORCH_CHECK(v.size(1) == H, "v heads?");
    TORCH_CHECK(v.size(2) == N, "v length?");

    // allocate output
    torch::Tensor out = torch::empty({B, H, N, DV}, v.options());
    torch::Tensor kv_a0 = torch::empty({B, H, 1,  DV}, v.options());
    torch::Tensor kv_a1 = torch::empty({B, H, DV, FD}, v.options());
    torch::Tensor kv_a2 = torch::empty({B, H, FD*FD, DV}, v.options());

    // convert to bf16
    c10::BFloat16 *q_bf16 = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_bf16 = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_bf16 = v.data_ptr<c10::BFloat16>();
    
    bf16 *d_q = reinterpret_cast<bf16*>(q_bf16);
    bf16 *d_k = reinterpret_cast<bf16*>(k_bf16);
    bf16 *d_v = reinterpret_cast<bf16*>(v_bf16);
    bf16 *d_o = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());
    bf16 *d_kv_a0 = reinterpret_cast<bf16*>(kv_a0.data_ptr<c10::BFloat16>());
    bf16 *d_kv_a1 = reinterpret_cast<bf16*>(kv_a1.data_ptr<c10::BFloat16>());
    bf16 *d_kv_a2 = reinterpret_cast<bf16*>(kv_a2.data_ptr<c10::BFloat16>());

    dispatch_based(
        d_q, d_k, d_v, d_o,
        B, H, N
    );

    kv_a1 = kv_a1.transpose(2, 3);
    torch::Tensor kv_concat = torch::cat({kv_a0, kv_a1, kv_a2}, /*dim=*/2);

    CHECK_CUDA_ERROR(cudaGetLastError());
    return std::make_tuple(out, kv_concat);
    cudaDeviceSynchronize();
}
#else
#include "harness_4090.impl"
#endif


