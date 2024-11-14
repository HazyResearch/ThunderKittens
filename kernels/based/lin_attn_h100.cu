#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <tuple>

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
    using q_gl     = gl<bf16,  -1, -1, -1, fd, q_tile>;
    using k_gl     = gl<bf16,  -1, -1, -1, fd, k_tile>;
    using v_gl     = gl<bf16,  -1, -1, -1, dv, v_tile>;
    using o_gl     = gl<bf16,  -1, -1, -1, dv, o_tile>;
    using kv_a0_gl = gl<bf16,  -1, -1,  1, dv, kv_a0_tile>;
    using kv_a1_gl = gl<bf16,  -1, -1, dv, fd, kv_a1_tile>;
    using kv_a2_gl = gl<bf16,  -1, -1, fd*fd, dv, kv_a2_tile>;

    // pointers
    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;
    kv_a0_gl kv_a0;
    kv_a1_gl kv_a1;
    kv_a2_gl kv_a2;
    int n;
};


// cumulative sum of v onto a0_total
template<kittens::ducks::st::all ST>
__device__ void accumulate_a0(sv_fl<64> &a0_float, const ST &v) {
    int col = laneid()*2;
    uint32_t handle = static_cast<uint32_t>(__cvta_generic_to_shared(&v.data[0]));
    float2 acc = float2{0.f, 0.f};
    // Unroll the entire loop to maximize ILP
    #pragma unroll
    for(int row = 0; row < ST::rows/4; row++) {
        bf16_2 tmp;
        move<bf16_2>::lds(tmp, v.idx(handle, {row+warpid()*(ST::rows/4), col}));
        float2 tmpf = __bfloat1622float2(tmp);
        acc.x += tmpf.x;
        acc.y += tmpf.y;
    }
    // __syncthreads();
    atomicAdd(&a0_float[col],   acc.x);
    atomicAdd(&a0_float[col+1], acc.y);
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
__device__ static void mul_slice_col(rt_bf<16,64> &dst, const rt_bf<64,16,col_l> &src, const int target_row) {

    const int lane = kittens::laneid(); // 0...31    
    // each thread is responsible for two cols
    copy(dst, reinterpret_cast<const rt_bf<16,64>&>(src));
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
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_bf<64,16> (&q_s)[2]   = al.allocate<st_bf<64,16>, 2>(); // 4096 bytes
    st_bf<64,16> (&k_s)[2]   = al.allocate<st_bf<64,16>, 2>(); // 4096 bytes
    st_bf<64,64> (&v_s)[2]   = al.allocate<st_bf<64,64>, 2>(); // 16384 bytes
    st_bf<64,64> (&v_s_2)[2] = al.allocate<st_bf<64,64>, 2>(); // 16384 bytes -- needed to prevent wgmma from breaking
    st_bf<64,64> (&o_s)[2]   = al.allocate<st_bf<64,64>, 2>(); // 16384 bytes

    rt_fl<16,16> a1_trans; // transposed chunk of a1.
    rt_fl<16,64> a2[4]; // a2 gets propagated through here.
    st_bf<64,16> (&a1_trans_s) = al.allocate<st_bf<64,16>>(); // 2048 bytes
    st_bf<64,64> (&a2_s)[4]  = al.allocate<st_bf<64,64>, 4>(); // 32768 bytes

    sv_fl<64> &a0_float = al.allocate<sv_fl<64>>();
    sv_bf<64> &a0_total = al.allocate<sv_bf<64>>();
    uint32_t a0_float_handle = static_cast<uint32_t>(__cvta_generic_to_shared(&a0_float[0]));

    warpgroup::zero(a1_trans_s);
    warpgroup::zero(a0_float);
    zero(a1_trans); // everyone zeroes a2.
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        zero(a2[i]); // everyone zeroes a2.
    }

    int n_blocks = g.n / (q_s[0].rows);

    // initial load
    __shared__ semaphore bar;
    if (warpid == 0) {
        init_semaphore(bar, 0, 1);
        tma::expect_bytes(bar,
            size_bytes<typeof(q_s[0])> +
            size_bytes<typeof(k_s[0])> +
            size_bytes<typeof(v_s[0])>*2
        );
        int tile_idx = blockIdx.x * n_blocks;
        tma::load_async(q_s[tic],   g.q, {batch, head, 0, 0}, bar);
        tma::load_async(k_s[tic],   g.k, {batch, head, 0, 0}, bar);
        tma::load_async(v_s[tic],   g.v, {batch, head, 0, 0}, bar); // it's actually faster to have TMA fill a few copies than for the warps to do it.
        tma::load_async(v_s_2[tic], g.v, {batch, head, 0, 0}, bar);
    }
    __syncthreads();

    for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) {
        rt_bf<16,64> local_attn_bf; // 4 registers each -- 16
        rt_fl<16,64> local_attn, temp_attn_accum; // 32 registers each -- 64
        rt_fl<16,64> o; // 32 registers each -- 64

        // arrive memory
        wait(bar, tic);

        // we start by doing the very local computations. Then, we'll follow up later with the rest.
        // note that local_attn rt shape is 1x4 since it's done by a warpgroup. 
        // even though you might think 4x4 since q_s x k_s is (4x1) x (1x4).
        warpgroup::mm_ABt(local_attn, q_s[tic], k_s[tic]); // clear registers -- note mm_ABt, not mma_ABt.
        if (warpid == 0 && block+1<n_blocks) { // go get the next QKV from HBM
            tma::expect_bytes(bar,
                size_bytes<typeof(q_s[0])> +
                size_bytes<typeof(k_s[0])> +
                size_bytes<typeof(v_s[0])>*2
            );
            tma::load_async(q_s[toc],   g.q, {batch, head, block+1, 0}, bar);
            tma::load_async(k_s[toc],   g.k, {batch, head, block+1, 0}, bar);
            tma::load_async(v_s[toc],   g.v, {batch, head, block+1, 0}, bar);
            tma::load_async(v_s_2[toc], g.v, {batch, head, block+1, 0}, bar);
        }
        // now we do the sum of the previous a0 onto o
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            #pragma unroll
            for(int j = 0; j < 2; j++) {
                int col = i*16 + j*8 + (laneid%4)*2;
                float2 data;
                move<float2>::lds(data, a0_float_handle + col*sizeof(float));
                o.tiles[0][i].data[2*j].x = data.x;
                o.tiles[0][i].data[2*j].y = data.y;
                o.tiles[0][i].data[2*j+1].x = data.x;
                o.tiles[0][i].data[2*j+1].y = data.y;
            }
        }
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

        warpgroup::mma_AB(o, local_attn_bf, v_s[tic]); // reset o here, and do local chunk.

        rt_bf<1*16,1*16> q_src; // the source 16x16 tiles -- we'll draw on these for future mul_slice's.
        warpgroup::load(q_src, q_s[tic]);
        // temperature scaling; divide by d
        mul(q_src, q_src, __float2bfloat16(0.25));
        warpgroup::mma_async_wait(); // tmp
        
        warpgroup::mma_ABt(o, q_src, a1_trans_s); // incorporate a1 onto o (SA: FLAG WAS q_smem[tic] HERE)
        // a1 kv state
        warpgroup::mma_AtB(a1_trans, v_s_2[tic], k_s[tic]); // we now have 4 1x4 registers that need to eventually be summed.
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            warpgroup::store(a2_s[i], a2[i]);
        }
        warpgroup::mma_async_wait(); // tmp
        warpgroup::store(a1_trans_s, a1_trans);
        mul(q_src, q_src, __float2bfloat16(0.70710678118)); // divide by 2 for A2 here; the mul_slices 
        rt_bf<64,16,col_l> k_src;
        load(k_src, k_s[tic]);

        // about 75% of execution time is in this loop
        __syncthreads();
        #pragma unroll
        for(int t = 0; t < 4; t++) {
            rt_bf<16,64> q, k;
            mul_slice_row(q, q_src, t*4);
            warpgroup::mma_AB(o, q, a2_s[t]); // incorporate a1 onto o
            warpgroup::mma_async_wait<1>(); // ding dong! o matmuls have now arrived, too.
            // Note: we originally have k_src_tmp and transpose it (line 283 above)
            // this is becuase AtB function is only usable if A is in SMEM. 
            // but we'd like to keep k in register, so we just transpose it upfront
            mul_slice_col(k, k_src, t*4+warpid);
            warpgroup::mma_AB(a2[t], k, v_s[tic]); // incorporate KtV onto a2
            warpgroup::mma_async_wait<1>(); // ding dong! o matmuls have now arrived, too.
        }
        tma::store_async_read_wait<1>();
        __syncthreads();
        warpgroup::mma_async_wait();

        // do the cumulative sum last, after everything is stored
        warpgroup::store(o_s[tic], o);
        accumulate_a0(a0_float, v_s[tic]); // cumulative sum of V onto O in shared memory
        __syncthreads();
        if (warpid == 0) { // go get the next K from HBM
            tma::store_async(g.o, o_s[tic], {batch, head, block, 0});
        }
    }
    warpgroup::copy(a0_total, a0_float);
    #pragma unroll
    for (int rt = 0; rt < 4; rt++) {
        mul(a2[rt], a2[rt], (0.70710678118f*0.25f)); // divides by math.sqrt(math.sqrt(D_QK))
        warpgroup::store(a2_s[rt], a2[rt]);
    }
    mul(a1_trans, a1_trans, 0.5);  // divides by math.sqrt(math.sqrt(D_QK))
    warpgroup::store(a1_trans_s, a1_trans);   // from individual warps to shared address
    tma::store_async_read_wait();
    __syncthreads();
    tma::store_async(g.kv_a2, a2_s[warpid], {batch, head, warpid, 0});  // tile_idx
    if (warpid == 0) {    // one warp takes care of the write to HBM
        tma::store_async(g.kv_a1, a1_trans_s, {batch, head, 0, 0});
        tma::store_async(g.kv_a0, a0_total, {batch, head, 0, 0});
    }
    tma::store_async_read_wait();
}


based_globals based_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    bf16 *d_kv_a2, bf16 *d_kv_a1, bf16 *d_kv_a0,
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
    using kv_a0_tile = globals::kv_a0_tile; // kv state
    using kv_a1_tile = globals::kv_a1_tile; 
    using kv_a2_tile = globals::kv_a2_tile;

    // global layouts
    using q_gl     = globals::q_gl;
    using k_gl     = globals::k_gl;
    using v_gl     = globals::v_gl;
    using o_gl     = globals::o_gl;
    using kv_a0_gl = globals::kv_a0_gl;
    using kv_a1_gl = globals::kv_a1_gl;
    using kv_a2_gl = globals::kv_a2_gl;

    q_gl     q_arg{d_q, ATTN_B, ATTN_H, ATTN_N, nullptr};
    k_gl     k_arg{d_k, ATTN_B, ATTN_H, ATTN_N, nullptr};
    v_gl     v_arg{d_v, ATTN_B, ATTN_H, ATTN_N, nullptr};
    o_gl     o_arg{d_o, ATTN_B, ATTN_H, ATTN_N, nullptr};
    kv_a0_gl kv_a0{d_kv_a0, ATTN_B, ATTN_H, nullptr, nullptr};
    kv_a1_gl kv_a1{d_kv_a1, ATTN_B, ATTN_H, nullptr, nullptr};
    kv_a2_gl kv_a2{d_kv_a2, ATTN_B, ATTN_H, nullptr, nullptr};

    globals g{
        q_arg, k_arg, v_arg, o_arg, 
        kv_a0, kv_a1, kv_a2, ATTN_N
    };
    return g;
}


#ifdef TK_COMPILE_BASED
#include "pyutils/torch_helpers.cuh"
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
        d_kv_a2, d_kv_a1, d_kv_a0, 
        B, H, N
    );

    kv_a1 = kv_a1.transpose(2, 3);
    torch::Tensor kv_concat = torch::cat({kv_a0, kv_a1, kv_a2}, /*dim=*/2);

    CHECK_CUDA_ERROR(cudaGetLastError());
    return std::make_tuple(out, kv_concat);
    cudaDeviceSynchronize();
}
#else
#include "harness.impl"
#endif


