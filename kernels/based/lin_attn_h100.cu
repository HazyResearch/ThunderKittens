#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

#ifdef TORCH_COMPILE
#define TK_COMPILE_BASED
#endif

#define NUM_WORKERS (4) // hardcoded, don't change
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define D_QK (16) // hardcoded, don't change
#define D_VO (64) // hardcoded but can be changed with some effort

using namespace kittens;

struct based_globals { 
    // shapes    
    static constexpr int dv = 64;
    static constexpr int fd = 16;

    using q_tile = st_bf<4*16, fd>;
    using k_tile = st_bf<4*16, fd>;
    using v_tile = st_bf<4*16, dv>;
    using o_tile = st_bf<4*16, dv>;
    using kv_a0_tile = sv_bf<dv>; // kv state
    using kv_a1_tile = st_bf<dv, fd>; 
    using kv_a2_tile = st_bf<dv, 4*fd>;

    // global layouts
    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using kv_a0_gl = gl<bf16,  -1, -1, -1, -1, kv_a0_tile>;
    using kv_a1_gl = gl<bf16,  -1, -1, -1, -1, kv_a1_tile>;
    using kv_a2_gl = gl<bf16,  -1, -1, -1, -1, kv_a2_tile>;

    // pointers
    q_gl q;
    k_gl k;
    v_gl v, v3;
    o_gl o;
    kv_a0_gl kv_a0;
    kv_a1_gl kv_a1;
    kv_a2_gl kv_a2;
    int n;
};


// cumulative sum of v onto a0_total
template<kittens::ducks::st::all ST>
__device__ void accumulate_a0(sv_bf<64> &a0_total, const ST &v) {
    int col = threadIdx.x*2;
    constexpr int PARALLEL_LOADS = 16;
    float2 v_data[PARALLEL_LOADS];
    if(col < ST::cols) {
        float2 acc = __bfloat1622float2(*(bf16_2*)&a0_total[col]);
        #pragma unroll
        for(int k = 0; k < ST::rows/PARALLEL_LOADS; k++) {
            #pragma unroll
            for(int i = 0; i < PARALLEL_LOADS; i++) { // load it all in
                int row = k*PARALLEL_LOADS+i;
                v_data[i] = __bfloat1622float2(*(bf16_2*)&v[int2{row, col}]);
            }
            #pragma unroll
            for(int i = 0; i < PARALLEL_LOADS; i++) { // accumulate, through registers, and write
                acc.x += v_data[i].x;
                acc.y += v_data[i].y;
            }
        }
        *(bf16_2*)&a0_total[col] = __float22bfloat162_rn(acc);
    }
}

// in pytorch, this computes, for a 16x64 tensor dst and 16x16 tensor src:
// dst = torch.cat([src * src[:,starting_col+i].unsqueeze(0) for i in range(4)], dim=-1)
__device__ static void mul_slice_row(rt_bf<1*16,4*16> &dst, const rt_bf<1*16,1*16> &src, const int starting_col) {

    const int lane = kittens::laneid(); // 0...31    
    // each thread is responsible for two rows
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        copy(reinterpret_cast<rt_bf<1*16,1*16>&>(dst.tiles[0][i]), src);
        const int target_col = starting_col + i;
        #pragma unroll
        for(int row_offset = 0; row_offset < 2; row_offset++) {
            const int src_thread = (lane / 4)*4 + (target_col%8)/2;
            const int col_offset = target_col >= 8;
            bf16_2 src_val = dst.tiles[0][i].data[2*col_offset + row_offset];
            bf16 val = __shfl_sync(kittens::MASK_ALL, (target_col%2 == 0) ? src_val.x : src_val.y, src_thread); // correct value obtained and passed around

            dst.tiles[0][i].data[row_offset] *= bf16_2{val, val};
            dst.tiles[0][i].data[row_offset+2] *= bf16_2{val, val};
        }
    }
}

// in pytorch, this computes, for a 16x64 tensor dst and 16x16 tensor src:
// dst = torch.cat([src * src[:,starting_col].unsqueeze(-1) for _ in range(4)], dim=-1)
__device__ static void mul_slice_col(rt_bf<1*16,4*16> &dst, const rt_bf<1*16,4*16> &src, const int target_row) {

    const int lane = kittens::laneid(); // 0...31    
    // each thread is responsible for two cols
    copy(dst, src);
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        #pragma unroll
        for(int col_offset = 0; col_offset < 2; col_offset++) {
            const int src_thread = (target_row%8)*4 + (lane%4);
            const int row_offset = target_row >= 8;
            bf16_2 src_val = dst.tiles[0][i].data[2*col_offset + row_offset];
            bf16_2 val = __shfl_sync(kittens::MASK_ALL, src_val, src_thread); // correct value obtained and passed around

            dst.tiles[0][i].data[col_offset*2+0] *= val;
            dst.tiles[0][i].data[col_offset*2+1] *= val;
        }
    }
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void based_linear_attention(const __grid_constant__ based_globals g) {

    const int batch = blockIdx.y;
    const int head  = blockIdx.x;

    int laneid = kittens::laneid(); 
    int warpid = kittens::warpid(); 
    int tic = 0, toc = 1;

    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_bf<4*16,1*16> (&q_s)[2]   = al.allocate<st_bf<4*16,1*16>, 2>(); // 4096 bytes
    st_bf<4*16,1*16> (&k_s)[2]   = al.allocate<st_bf<4*16,1*16>, 2>(); // 4096 bytes
    st_bf<4*16,4*16> (&v_s)[2]   = al.allocate<st_bf<4*16,4*16>, 2>(); // 16384 bytes
    st_bf<4*16,4*16> (&v_s_2)[2] = al.allocate<st_bf<4*16,4*16>, 2>(); // 16384 bytes -- needed to prevent wgmma from breaking
    st_bf<4*16,4*16> (&v_s_3)[2] = al.allocate<st_bf<4*16,4*16>, 2>(); // 16384 bytes -- used to reduce bank conflicts for a0 sum
    st_bf<4*16,4*16> (&o_s)[2]   = al.allocate<st_bf<4*16,4*16>, 2>(); // 16384 bytes

    rt_fl<1*16,1*16> a1_trans; // transposed chunk of a1.
    rt_fl<1*16,4*16> a2[4]; // a2 gets propagated through here.
    st_bf<4*16,1*16> (&a1_trans_s) = al.allocate<st_bf<4*16,1*16>>(); // 2048 bytes
    st_bf<4*16,4*16> (&a2_s)  = al.allocate<st_bf<4*16,4*16>>(); // 8192 bytes

    sv_bf<4*16> &a0_total = al.allocate<sv_bf<4*16>>();

    if(warpid == 0) {
        zero(a0_total);
    }
    warpgroup::zero(a1_trans_s);
    zero(a1_trans); // everyone zeroes a2.
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        zero(a2[i]); // everyone zeroes a2.
    }

    int n_blocks = g.n / (q_s[0].rows);

    // initial load
    __shared__ barrier bar;
    if (warpid == 0) init_barrier(bar, 0, 1);
    __syncthreads();
    if (warpid == 0) {
        tma::expect_bytes(bar,
            size_bytes<typeof(q_s[0])> +
            size_bytes<typeof(k_s[0])> +
            size_bytes<typeof(v_s[0])>*3
        );
        int tile_idx = blockIdx.x * n_blocks;
        tma::load_async(q_s[tic], g.q, {batch, head, 0, 0}, bar);
        tma::load_async(k_s[tic], g.k, {batch, head, 0, 0}, bar);
        tma::load_async(v_s[tic], g.v, {batch, head, 0, 0}, bar); // it's actually faster to have TMA fill a few copies than for the warps to do it.
        tma::load_async(v_s_2[tic], g.v, {batch, head, 0, 0}, bar);
        tma::load_async(v_s_3[tic], g.v3, {batch, head, 0, 0}, bar);
    }

    for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) {
        rt_bf<1*16,4*16> local_attn_bf; // 4 registers each -- 16
        rt_fl<1*16,4*16> local_attn, temp_attn_accum; // 32 registers each -- 64
        rt_fl<1*16,4*16> o; // 32 registers each -- 64

        // arrive memory
        wait(bar, tic);
        __syncthreads(); // everybody on the same page?
        if (warpid == 0 && block+1<n_blocks) { // go get the next K from HBM
            tma::expect_bytes(bar,
                size_bytes<typeof(q_s[0])> +
                size_bytes<typeof(k_s[0])> +
                size_bytes<typeof(v_s[0])>*3
            );

            int next_tile_idx = (blockIdx.x * n_blocks) + block + 1;
            tma::load_async(q_s[toc], g.q, {batch, head, block+1, 0}, bar);
            tma::load_async(k_s[toc], g.k, {batch, head, block+1, 0}, bar);
            tma::load_async(v_s[toc], g.v, {batch, head, block+1, 0}, bar);
            tma::load_async(v_s_2[toc], g.v, {batch, head, block+1, 0}, bar);
            tma::load_async(v_s_3[toc], g.v3, {batch, head, block+1, 0}, bar);
        }

        // we start by doing the very local computations. Then, we'll follow up later with the rest.
        warpgroup::mma_fence(local_attn); // qk matmul fence
        warpgroup::mm_ABt(local_attn, q_s[tic], k_s[tic]); // clear registers -- note mm_ABt, not mma_ABt.
        warpgroup::mma_commit_group(); // dew it
        warpgroup::mma_async_wait(); // ding dong! matmuls arrived.

        // temperature scaling; divide a1 term by sqrt(d)
        mul(local_attn, local_attn, 0.25f);

        // our goal is to store local_attn + (local_attn^2 / 2) in local_attn_bf
        copy(temp_attn_accum, local_attn);
        mul(temp_attn_accum, temp_attn_accum, temp_attn_accum); // square it
        mul(temp_attn_accum, temp_attn_accum, 0.5f); // divide by 2
        add(temp_attn_accum, temp_attn_accum, local_attn); // add back in 1x for the linear term
        add(temp_attn_accum, temp_attn_accum, 1.f); // cumulative sum for a0
        copy(local_attn_bf, temp_attn_accum); // now stored.
        // now make causal
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            auto &attn_subtile = reinterpret_cast<rt_bf<1*16,1*16>&>(local_attn_bf.tiles[0][j]);
            if (j>warpid) zero(attn_subtile);
            else if (j==warpid) make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<bf16>::zero());
        }

        warpgroup::mma_fence(o); // av matmul fence
        warpgroup::mma_fence(a1_trans); // a1 accumulation matmul fence
        warpgroup::mm_AB(o, local_attn_bf, v_s[tic]); // reset o here, and do local chunk.
        warpgroup::mma_commit_group(); // dew it

        rt_bf<1*16,1*16> q_src; // the source 16x16 tiles -- we'll draw on these for future mul_slice's.
        warpgroup::load(q_src, q_s[tic]);
        // temperature scaling; divide by d
        mul(q_src, q_src, __float2bfloat16(0.25));
        
        warpgroup::mma_ABt(o, q_src, a1_trans_s); // incorporate a1 onto o (SA: FLAG WAS q_smem[tic] HERE)
        warpgroup::mma_commit_group(); // dew it

        // a1 kv state
        warpgroup::mma_AtB(a1_trans, v_s_2[tic], k_s[tic]); // we now have 4 1x4 registers that need to eventually be summed.
        warpgroup::mma_commit_group(); // dew it
        warpgroup::mma_async_wait(); // tmp
        warpgroup::store(a1_trans_s, a1_trans);

        mul(q_src, q_src, __float2bfloat16(0.70710678118)); // divide by 2 for A2 here; the mul_slices 
        rt_bf<4*16,1*16> k_src_tmp;
        rt_bf<1*16,4*16> k_src;
        load(k_src_tmp, k_s[tic]);
        transpose_sep(k_src, k_src_tmp); // transpose K into Kt
        
        // about 75% of execution time is in this loop
        #pragma unroll
        for(int t = 0; t < 4; t++) {
            rt_bf<1*16,4*16> q, k;
            mul_slice_row(q, q_src, t*4);
            mul_slice_col(k, k_src, t*4+warpid);
            warpgroup::store(a2_s, a2[t]); // take previous one and move up to smem for wgmma.
            __syncthreads();

            warpgroup::mma_fence(o); // av matmul fence
            warpgroup::mma_fence(a2[t]); // av matmul fence
            warpgroup::mma_AB(o, q, a2_s); // incorporate a1 onto o
            warpgroup::mma_commit_group(); // dew it
            warpgroup::mma_AB(a2[t], k, v_s[tic]); // incorporate KtV onto a2
            warpgroup::mma_commit_group(); // dew it
            warpgroup::mma_async_wait(); // ding dong! o matmuls have now arrived, too.
        }

        // now we do the sum of the previous a0 onto o
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            #pragma unroll
            for(int j = 0; j < 2; j++) {
                int col = i*16 + j*8 + (laneid%4)*2;
                float2 data = __bfloat1622float2(*(bf16_2*)&a0_total[col]);
                o.tiles[0][i].data[2*j].x += data.x;
                o.tiles[0][i].data[2*j].y += data.y;
                o.tiles[0][i].data[2*j+1].x += data.x;
                o.tiles[0][i].data[2*j+1].y += data.y;
            }
        }

        if (block>0) {
            tma::store_async_read_wait<1>();
            warpgroup::sync();
        }

        // do the cumulative sum last, after everything is stored
        warpgroup::store(o_s[tic], o);
        __syncthreads();
        accumulate_a0(a0_total, v_s_3[tic]); // cumulative sum of V onto O in shared memory
        __syncthreads();

        if (warpid == 0) { // go get the next K from HBM
            tma::store_async(g.o, o_s[tic], {batch, head, block, 0});
        }
    }
    tma::store_async_wait();

    // save the KV state (A2)
    for (int rt = 0; rt < 4; rt++) {
        // reinterpret_cast doesn’t change the bits or memory layout of the variable a2_s. 
        // it tells the compiler to treat a2_s as a reference to the type we've specified.
        auto &kv_smem_2 = reinterpret_cast<st_bf<4*16,4*16>&>(a2_s); // this layout is better for global HBM stores so we cast.
        mul(a2[rt], a2[rt], (0.70710678118*0.70710678118*0.25*1.41421)); // divides by math.sqrt(math.sqrt(D_QK))
        warpgroup::store(kv_smem_2, a2[rt]); 
        __syncthreads();
        if (warpid == 0) {
            // int tile_idx = (blockIdx.x * 4) + rt; 
            tma::store_async(g.kv_a2, kv_smem_2, {batch, head, rt, 0});  // tile_idx
            tma::store_commit_group(); 
        }
        tma::store_async_wait();
    }
    // save the KV state A1
    auto &kv_smem_1 = reinterpret_cast<st_bf<4*16,1*16>&>(a1_trans_s);
    mul(a1_trans, a1_trans, 0.5);  // divides by math.sqrt(math.sqrt(D_QK))
    warpgroup::store(kv_smem_1, a1_trans);   // from individual warps to shared address
    __syncthreads();
    if (warpid == 0) {    // one warp takes care of the write to HBM
        tma::store_async(g.kv_a1, kv_smem_1, {batch, head, 0, 0}); 
        tma::store_commit_group(); 
    }
    tma::store_async_wait();
    // save the KV state A0
    if (warpid == 0) {   
        tma::store_async(g.kv_a0, a0_total, {batch, head, 0, 0}); 
        tma::store_commit_group(); 
    }
    tma::store_async_wait();

}


based_globals based_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    bf16 *d_kv_a2, bf16 *d_kv_a1, bf16 *d_kv_a0,
    int ATTN_B, int ATTN_H, int ATTN_N
) {
    // global pointers
    int ATTN_D = 64; 
    int ATTN_D_SMALL = 16;

    static constexpr int dv = 64;
    static constexpr int fd = 16;

    using q_tile = st_bf<4*16, fd>;
    using k_tile = st_bf<4*16, fd>;
    using v_tile = st_bf<4*16, dv>;
    using o_tile = st_bf<4*16, dv>;
    using kv_a0_tile = sv_bf<dv>; // kv state
    using kv_a1_tile = st_bf<dv, fd>; 
    using kv_a2_tile = st_bf<dv, 4*fd>;

    // global layouts
    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using kv_a0_gl = gl<bf16,  -1, -1, -1, -1, kv_a0_tile>;
    using kv_a1_gl = gl<bf16,  -1, -1, -1, -1, kv_a1_tile>;
    using kv_a2_gl = gl<bf16,  -1, -1, -1, -1, kv_a2_tile>;

    using globals = based_globals;

    q_gl q_arg{d_q, ATTN_B, ATTN_H, ATTN_N, ATTN_D_SMALL};
    k_gl k_arg{d_k, ATTN_B, ATTN_H, ATTN_N, ATTN_D_SMALL};
    v_gl v_arg{d_v, ATTN_B, ATTN_H, ATTN_N, ATTN_D};
    v_gl v_3_arg{d_v, ATTN_B, ATTN_H, ATTN_N, ATTN_D};
    o_gl o_arg{d_o, ATTN_B, ATTN_H, ATTN_N, ATTN_D};
    kv_a0_gl kv_a0{d_kv_a0, ATTN_B, ATTN_H, 1, ATTN_D};
    kv_a1_gl kv_a1{d_kv_a1, ATTN_B, ATTN_H, ATTN_D, ATTN_D_SMALL};
    kv_a2_gl kv_a2{d_kv_a2, ATTN_B, ATTN_H, ATTN_D_SMALL*ATTN_D_SMALL, ATTN_D};

    globals g{
        q_arg, k_arg, v_arg, v_3_arg, o_arg, 
        kv_a0, kv_a1, kv_a2, ATTN_N
    };
    return g;
}


#ifdef TK_COMPILE_BASED
#include "common/pyutils/torch_helpers.cuh"
#include <iostream>
void dispatch_based( 
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    bf16 *d_kv_a2, bf16 *d_kv_a1, bf16 *d_kv_a0,
    int ATTN_B, int ATTN_H, int ATTN_N
){
    based_globals g = based_init(
        d_q, d_k, d_v, d_o, 
        d_kv_a2, d_kv_a1, d_kv_a0,
        ATTN_B, ATTN_H, ATTN_N
    );

    // launch
    unsigned long mem_size = 98000;
    cudaFuncSetAttribute(
        based_linear_attention,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(ATTN_H, ATTN_B);
    based_linear_attention<<<grid,NUM_THREADS,mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

torch::Tensor based(
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
    int N = q.size(2);
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
    torch::Tensor kv_a2 = torch::empty({B, H, DV, FD*FD}, v.options());

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
        d_kv_a2, d_kv_a1, d_kv_a0, 
        B, H, N
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    return out;
}
#else
#include "harness.impl"
#endif


