#include "src/kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

#define NUM_WORKERS (4) // hardcoded, don't change
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define D_QK (16) // hardcoded, don't change
#define D_VO (64) // hardcoded but can be changed with some effort

using namespace kittens;


// cumulative sum of v onto a0_total
template<kittens::ducks::st::all ST>
__device__ void accumulate_a0(sv_bf<ST::width> &a0_total, const ST &v) {
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
__device__ static void mul_slice_row(rt_bf_1x4<> &dst, const rt_bf_1x1<> &src, const int starting_col) {

    const int lane = kittens::laneid(); // 0...31    
    // each thread is responsible for two rows
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        copy(reinterpret_cast<rt_bf_1x1<>&>(dst.tiles[0][i]), src);
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
__device__ static void mul_slice_col(rt_bf_1x4<> &dst, const rt_bf_1x4<> &src, const int target_row) {

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
void based_linear_attention (int n, int dec_enc_ratio, CUtensorMap* tma_q,     
                            CUtensorMap* tma_k,     CUtensorMap* tma_v,     CUtensorMap* tma_v_3, 
                            CUtensorMap* tma_k_enc, CUtensorMap* tma_v_enc, CUtensorMap* tma_v_3_enc, 
                            CUtensorMap* tma_o) 
{
    int laneid = kittens::laneid(); // who am i? when am i?
    int warpid = kittens::warpid(); // who am i? when am i?
    int tic = 0, toc = 1;

    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_bf_4x1<wgmma_swizzle_l>    (&q_s)  [2]   = al.allocate<st_bf_4x1<wgmma_swizzle_l>,    2>(); // 4096 bytes
    st_bf_4x1<wgmma_interleave_l> (&k_s)  [2]   = al.allocate<st_bf_4x1<wgmma_interleave_l>, 2>(); // 4096 bytes
    st_bf_4x4<wgmma_interleave_l> (&v_s)  [2]   = al.allocate<st_bf_4x4<wgmma_interleave_l>, 2>(); // 16384 bytes
    st_bf_4x4<wgmma_interleave_l> (&v_s_2)[2]   = al.allocate<st_bf_4x4<wgmma_interleave_l>, 2>(); // 16384 bytes -- needed to prevent wgmma from breaking
    st_bf_4x4<swizzle_l>          (&v_s_3)[2]   = al.allocate<st_bf_4x4<swizzle_l>,          2>(); // 16384 bytes -- used to reduce bank conflicts for a0 sum
    st_bf_4x4<swizzle_l>          (&o_s)  [2]   = al.allocate<st_bf_4x4<swizzle_l>,          2>(); // 16384 bytes

    rt_fl_1x1<> a1_trans; // transposed chunk of a1.
    rt_fl_1x4<> a2[4]; // a2 gets propagated through here.
    st_bf_4x1<wgmma_swizzle_l>    (&a1_trans_s) = al.allocate<st_bf_4x1<wgmma_swizzle_l>    >(); // 2048 bytes
    st_bf_4x4<wgmma_interleave_l> (&a2_s)       = al.allocate<st_bf_4x4<wgmma_interleave_l> >(); // 8192 bytes

    sv_bf_4 &a0_total = al.allocate<sv_bf_4>();

    if(warpid == 0) {
        zero(a0_total);
    }
    warpgroup::zero(a1_trans_s);
    zero(a1_trans); // everyone zeroes a2.
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        zero(a2[i]); // everyone zeroes a2.
    }

    int n_blocks_enc = n / (q_s[0].rows * dec_enc_ratio); 

    // kv_enc load
    __shared__ tma::barrier bar_enc;
    if (warpid == 0) tma::init_barrier(bar_enc);
    __syncthreads();
    if (warpid == 0) {
        tma::set_bytes(bar_enc,
            size_bytes<typeof(k_s[0])> +
            size_bytes<typeof(v_s[0])>*3
        );
        int tile_idx = blockIdx.x * n_blocks_enc;
        tma::load_async(k_s[tic],   tma_k_enc,   bar_enc, tile_idx);
        tma::load_async(v_s[tic],   tma_v_enc,   bar_enc, tile_idx);
        tma::load_async(v_s_2[tic], tma_v_enc,   bar_enc, tile_idx);
        tma::load_async(v_s_3[tic], tma_v_3_enc, bar_enc, tile_idx);
    }

    for (int block = 0; block < n_blocks_enc; block++, tic^=1, toc^=1) {

        // arrive memory
        tma::arrive_and_wait(bar_enc, tic);
        __syncthreads(); // everybody on the same page?
        if (warpid == 0 && block+1<n_blocks_enc) { // go get the next K from HBM
            tma::set_bytes(bar_enc,
                size_bytes<typeof(k_s[0])> +
                size_bytes<typeof(v_s[0])>*3
            );
            int next_tile_idx = (blockIdx.x * n_blocks_enc) + block + 1;
            tma::load_async(k_s[toc],   tma_k_enc,   bar_enc, next_tile_idx);
            tma::load_async(v_s[toc],   tma_v_enc,   bar_enc, next_tile_idx);
            tma::load_async(v_s_2[toc], tma_v_enc,   bar_enc, next_tile_idx);
            tma::load_async(v_s_3[toc], tma_v_3_enc, bar_enc, next_tile_idx);
        }

        warpgroup::mma_fence(a1_trans);                     // a1 accumulation matmul fence
        warpgroup::mma_AtB(a1_trans, v_s_2[tic], k_s[tic]); // we now have 4 1x4 registers that need to eventually be summed.
        warpgroup::mma_commit_group();                      // dew it
        
        warpgroup::mma_async_wait(); 
        
        warpgroup::store(a1_trans_s, a1_trans);

        rt_bf_4x1<> k_src_tmp;
        rt_bf_1x4<> k_src;
        load(k_src_tmp, k_s[tic]);
        transpose_sep(k_src, k_src_tmp); // transpose K into Kt
        
        // about 75% of execution time is in this loop
        #pragma unroll
        for(int t = 0; t < 4; t++) {
            rt_bf_1x4<> k;
            mul_slice_col(k, k_src, t*4+warpid);
            
            warpgroup::mma_fence(a2[t]); // av matmul fence
            warpgroup::mma_AB(a2[t], k, v_s[tic]); // incorporate KtV onto a2
            warpgroup::mma_commit_group(); // dew it
            
            warpgroup::mma_async_wait(); // ding dong! o matmuls have now arrived, too.
        }

        accumulate_a0(a0_total, v_s_3[tic]); // cumulative sum of V onto O in shared memory
        __syncthreads();
    }

    tic = 0;
    toc = 1;
    int n_blocks     = n / (q_s[0].rows);

    // initial load
    __shared__ tma::barrier bar;
    if (warpid == 0) tma::init_barrier(bar);
    __syncthreads();
    if (warpid == 0) {
        tma::set_bytes(bar,
            size_bytes<typeof(q_s[0])> +
            size_bytes<typeof(k_s[0])> +
            size_bytes<typeof(v_s[0])>*3
        );
        int tile_idx = blockIdx.x * n_blocks;
        tma::load_async(q_s[tic],   tma_q,   bar, tile_idx);
        tma::load_async(k_s[tic],   tma_k,   bar, tile_idx);
        tma::load_async(v_s[tic],   tma_v,   bar, tile_idx); // it's actually faster to have TMA fill a few copies than for the warps to do it.
        tma::load_async(v_s_2[tic], tma_v,   bar, tile_idx);
        tma::load_async(v_s_3[tic], tma_v_3, bar, tile_idx);
    }

    for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) {
        rt_bf_1x4<> local_attn_bf; // 4 registers each -- 16
        rt_fl_1x4<> local_attn, temp_attn_accum; // 32 registers each -- 64
        rt_fl_1x4<> o; // 32 registers each -- 64

        // arrive memory
        tma::arrive_and_wait(bar, tic);
        __syncthreads(); // everybody on the same page?
        if (warpid == 0 && block+1<n_blocks) { // go get the next K from HBM
            tma::set_bytes(bar,
                size_bytes<typeof(q_s[0])> +
                size_bytes<typeof(k_s[0])> +
                size_bytes<typeof(v_s[0])>*3
            );
            int next_tile_idx = (blockIdx.x * n_blocks) + block + 1;
            tma::load_async(q_s[toc],   tma_q,   bar, next_tile_idx);
            tma::load_async(k_s[toc],   tma_k,   bar, next_tile_idx);
            tma::load_async(v_s[toc],   tma_v,   bar, next_tile_idx);
            tma::load_async(v_s_2[toc], tma_v,   bar, next_tile_idx);
            tma::load_async(v_s_3[toc], tma_v_3, bar, next_tile_idx);
        }

        // we start by doing the very local computations. Then, we'll follow up later with the rest.
        warpgroup::mma_fence(local_attn); // qk matmul fence
        warpgroup::mm_ABt(local_attn, q_s[tic], k_s[tic]); // clear registers -- note mm_ABt, not mma_ABt.
        warpgroup::mma_commit_group(); // dew it
        warpgroup::mma_async_wait(); // ding dong! matmuls arrived.

        // our goal is to store local_attn + (local_attn^2 / 2) in local_attn_bf
        copy(temp_attn_accum, local_attn);
        // BEGIN comment-out for removing T2 (debug)
        mul(temp_attn_accum, temp_attn_accum, temp_attn_accum); // square it
        mul(temp_attn_accum, temp_attn_accum, 0.5f);            // divide by 2
        add(temp_attn_accum, temp_attn_accum, local_attn);      // add back in 1x for the linear term
        add(temp_attn_accum, temp_attn_accum, 1.f);             // cumulative sum for a0
        // END comment-out for removing T2 (debug)
        copy(local_attn_bf, temp_attn_accum); // now stored.
        // now make causal
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            auto &attn_subtile = reinterpret_cast<rt_bf_1x1<>&>(local_attn_bf.tiles[0][j]);
            if (j>warpid) zero(attn_subtile);
            else if (j==warpid) make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<bf16>::zero());
        }

        warpgroup::mma_fence(o);                            // av matmul fence
        warpgroup::mma_fence(a1_trans);                     // a1 accumulation matmul fence
        warpgroup::mm_AB(o, local_attn_bf, v_s[tic]);       // reset o here, and do local chunk.
        warpgroup::mma_commit_group();                      // dew it
        
        warpgroup::mma_ABt(o, q_s[tic], a1_trans_s);        // incorporate a1 onto o
        warpgroup::mma_commit_group();                      // dew it
        
        warpgroup::mma_AtB(a1_trans, v_s_2[tic], k_s[tic]); // we now have 4 1x4 registers that need to eventually be summed.
        warpgroup::mma_commit_group(); // dew it
        warpgroup::mma_async_wait();   // tmp
        
        warpgroup::store(a1_trans_s, a1_trans);

        rt_bf_1x1<> q_src; // the source 16x16 tiles -- we'll draw on these for future mul_slice's.
        warpgroup::load(q_src, q_s[tic]);
        mul(q_src, q_src, __float2bfloat16(0.70710678118)); // divide by 2 for A2 here.
        rt_bf_4x1<> k_src_tmp;
        rt_bf_1x4<> k_src;
        load(k_src_tmp, k_s[tic]);
        transpose_sep(k_src, k_src_tmp); // transpose K into Kt
        
        // about 75% of execution time is in this loop
        #pragma unroll
        for(int t = 0; t < 4; t++) {
            rt_bf_1x4<> q, k;
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
                o.tiles[0][i].data[2*j].x   += data.x;
                o.tiles[0][i].data[2*j].y   += data.y;
                o.tiles[0][i].data[2*j+1].x += data.x;
                o.tiles[0][i].data[2*j+1].y += data.y;
            }
        }

        // // do the cumulative sum last, after everything is stored
        warpgroup::store(o_s[tic], o);
        __syncthreads();
        accumulate_a0(a0_total, v_s_3[tic]); // cumulative sum of V onto O in shared memory
        __syncthreads();

        if (block>0) tma::store_async_wait();
        if (warpid == 0) { // go get the next K from HBM
            tma::store_async(tma_o, o_s[tic], (blockIdx.x * n_blocks) + block); 
            tma::store_commit_group(); // dew it
        }
    }
    tma::store_async_wait();
}

// #include "harness.impl"
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>
void jrt_prefill_tk(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor k_enc, torch::Tensor v_enc, torch::Tensor o) {
    CHECK_INPUT(q); 
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(k_enc);
    CHECK_INPUT(v_enc);
    CHECK_INPUT(o);

    auto batch = q.size(0);
    auto heads = q.size(1);
    auto threads = NUM_THREADS; 

    auto dec_seq_len = q.size(2);
    auto enc_seq_len = k_enc.size(2);
    auto dec_enc_ratio = dec_seq_len / enc_seq_len;

    bool k_same = true; 
    bool o_same = true; 

    for (auto i = 0; i < 4; i++) {
        k_same &= q.size(i) == k.size(i);
        o_same &= v.size(i) == o.size(i);
    }

    TORCH_CHECK(k_same, "q and k must have the same shape");
    TORCH_CHECK(o_same, "v and o must have the same shape");

    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "k must be bf16");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "v must be bf16");
    TORCH_CHECK(k_enc.scalar_type() == c10::ScalarType::BFloat16, "k_enc must be bf16");
    TORCH_CHECK(v_enc.scalar_type() == c10::ScalarType::BFloat16, "v_enc must be bf16");
    TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "o must be bf16");

    TORCH_CHECK(dec_seq_len % (NUM_WORKERS*kittens::TILE_DIM) == 0, "dec_seq_len must be a multiple of NUM_WORKERS*TILE_DIM");

    // convert to bfloat16
    c10::BFloat16* q_ptr     = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr     = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr     = v.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_enc_ptr = k_enc.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_enc_ptr = v_enc.data_ptr<c10::BFloat16>();
    c10::BFloat16* o_ptr     = o.data_ptr<c10::BFloat16>();

    const bf16* d_q     = reinterpret_cast<const bf16*>(q_ptr);
    const bf16* d_k     = reinterpret_cast<const bf16*>(k_ptr);
    const bf16* d_v     = reinterpret_cast<const bf16*>(v_ptr);
    const bf16* d_k_enc = reinterpret_cast<const bf16*>(k_enc_ptr);
    const bf16* d_v_enc = reinterpret_cast<const bf16*>(v_enc_ptr);
          bf16* d_o     = reinterpret_cast<bf16*>(o_ptr);

    CUtensorMap* tma_q_d       = tma::allocate_and_create_tensor_map<kittens::st_bf_4x1<wgmma_swizzle_l   >>(d_q,     (batch*heads*dec_seq_len)              /(4 * kittens::TILE_DIM));
    CUtensorMap* tma_k_d       = tma::allocate_and_create_tensor_map<kittens::st_bf_4x1<wgmma_interleave_l>>(d_k,     (batch*heads*dec_seq_len)              /(4 * kittens::TILE_DIM));
    CUtensorMap* tma_k_enc_d   = tma::allocate_and_create_tensor_map<kittens::st_bf_4x1<wgmma_interleave_l>>(d_k_enc, (batch*heads*dec_seq_len/dec_enc_ratio)/(4 * kittens::TILE_DIM));
    CUtensorMap* tma_v_d       = tma::allocate_and_create_tensor_map<kittens::st_bf_4x4<wgmma_interleave_l>>(d_v,     (batch*heads*dec_seq_len)              /(4 * kittens::TILE_DIM));
    CUtensorMap* tma_v_3_d     = tma::allocate_and_create_tensor_map<kittens::st_bf_4x4<swizzle_l         >>(d_v,     (batch*heads*dec_seq_len)              /(4 * kittens::TILE_DIM));
    CUtensorMap* tma_v_enc_d   = tma::allocate_and_create_tensor_map<kittens::st_bf_4x4<wgmma_interleave_l>>(d_v_enc, (batch*heads*dec_seq_len/dec_enc_ratio)/(4 * kittens::TILE_DIM));
    CUtensorMap* tma_v_enc_3_d = tma::allocate_and_create_tensor_map<kittens::st_bf_4x4<swizzle_l         >>(d_v_enc, (batch*heads*dec_seq_len/dec_enc_ratio)/(4 * kittens::TILE_DIM));
    CUtensorMap* tma_o_d       = tma::allocate_and_create_tensor_map<kittens::st_bf_4x4<swizzle_l         >>(d_o,     (batch*heads*dec_seq_len)              /(4 * kittens::TILE_DIM));

    unsigned long mem_size = 98000; 

    using T = kittens::bf16; 
    using H = kittens::bf16;
    cudaFuncSetAttribute(
        based_linear_attention,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    based_linear_attention<<<batch*heads, threads, mem_size>>>(dec_seq_len, dec_enc_ratio, tma_q_d, tma_k_d, tma_v_d, tma_v_3_d, tma_k_enc_d, tma_v_enc_d, tma_v_enc_3_d, tma_o_d);

    CHECK_CUDA_ERROR(cudaGetLastError());
}
