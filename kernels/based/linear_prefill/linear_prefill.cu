#define TORCH_COMPILE 

#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

#define NUM_WORKERS (4) // hardcoded, don't change
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define D_QK (16) // hardcoded, don't change
#define D_VO (64) // hardcoded but can be changed with some effort

#define tile_smem_bf_4x1 st_bf_4x1
#define tile_smem_fl_4x1 st<float, 4, 1>
#define tile_smem_bf_4x4 st_bf_4x4
#define tile_smem_fl_4x4 st<float, 4, 4>      
#define tile_smem_bf_4x4_wgmma_interleave st_bf_4x4
#define vec_smem_bf_4 sv_bf<4>
#define vec_smem_bf_1 sv_bf<1>
#define row_vec_smem_bf_1x4 row_vec<st_bf_1x4> 
#define row_vec_smem_fl_1x4 row_vec<st<float, 1, 4>> 

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


// in pytorch, this computes, for a 16x64 tensor dst and 16x64 tensor src:
// dst = src * src[:,starting_col].unsqueeze(-1)
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


// the overall algorithm partitions across 1 warpgroup of 4 workers.
__global__ __launch_bounds__(NUM_THREADS, 2)
void based_linear_attention(
    int n, int add_scale, int output_state, 
    CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_v_3, CUtensorMap* tma_o, 
    CUtensorMap* tma_kv_a2, CUtensorMap* tma_kv_a1, CUtensorMap* tma_kv_a0 
) {

    int laneid = kittens::laneid(); // who am i? when am i?
    int warpid = kittens::warpid(); // who am i? when am i?
    int tic = 0, toc = 1;

    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_bf_4x1 (&q_s)  [2]        = al.allocate<st_bf_4x1,2>(); // 4096 bytes
    tile_smem_bf_4x1 (&k_s)  [2] = al.allocate<tile_smem_bf_4x1, 2>();              // 4096 bytes
    tile_smem_bf_4x4_wgmma_interleave (&v_s)  [2]   = al.allocate<tile_smem_bf_4x4_wgmma_interleave, 2>(); // 16384 bytes
    tile_smem_bf_4x4_wgmma_interleave (&v_s_2)[2]   = al.allocate<tile_smem_bf_4x4_wgmma_interleave, 2>(); // 16384 bytes -- needed to prevent wgmma from breaking
    tile_smem_bf_4x4       (&v_s_3)[2]   = al.allocate<tile_smem_bf_4x4,          2>(); // 16384 bytes -- used to reduce bank conflicts for a0 sum
    tile_smem_bf_4x4       (&o_s)  [2]   = al.allocate<tile_smem_bf_4x4,          2>(); // 16384 bytes

    rt_fl_1x1<> a1_trans; // transposed chunk of a1.
    rt_fl_1x4<> a2[4];    // a2 gets propagated through here.
    st_bf_4x1 (&a1_trans_s) = al.allocate<st_bf_4x1>();   // 2048 bytes
    tile_smem_bf_4x4_wgmma_interleave (&a2_s)   = al.allocate<tile_smem_bf_4x4_wgmma_interleave >();   // 8192 bytes
    sv_bf_4 &a0_total = al.allocate<sv_bf_4>();

    if(warpid == 0) { zero(a0_total); }
    warpgroup::zero(a1_trans_s);
    zero(a1_trans); // everyone zeroes a2.
    #pragma unroll
    for(int i = 0; i < 4; i++) { zero(a2[i]); } // everyone zeroes a2.
    int n_blocks = n / (q_s[0].rows);

    // initial load
    __shared__ barrier bar;
    if (warpid == 0) init_barrier(bar, 0, 1); // don't wait on threads, do wait on one memory transaction
    __syncthreads();
    if (warpid == 0) {
        tma::expect_bytes(bar,
            size_bytes<typeof(q_s[0])> +
            size_bytes<typeof(k_s[0])> +
            size_bytes<typeof(v_s[0])>*3
        );
        int tile_idx = blockIdx.x * n_blocks;
        tma::load_async(q_s[tic],   tma_q,   bar, tile_idx);
        tma::load_async(k_s[tic],   tma_k,   bar, tile_idx);
        tma::load_async(v_s[tic],   tma_v,   bar, tile_idx); // faster to have TMA fill a copies than the warps do it.
        tma::load_async(v_s_2[tic], tma_v,   bar, tile_idx);
        tma::load_async(v_s_3[tic], tma_v_3, bar, tile_idx);
    }
    // if(threadIdx.x ==0 && blockIdx.x == 0) printf("%llu\n", (uint64_t)(&a0_cumsum) - (uint64_t)(&__shm[0]));

    for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) {
        // each iteration, we handle 64 tokens x 16 feature dim (k, q).
        rt_bf_1x4<> local_attn_bf;               // 4 registers each -- 16
        rt_fl_1x4<> local_attn, temp_attn_accum; // 32 registers each -- 64
        rt_fl_1x4<> o;                           // 32 registers each -- 64

        // arrive memory
        wait(bar, tic);
        __syncthreads(); // everybody on the same page?
        if (warpid == 0 && block+1<n_blocks) {   // go get the next K from HBM
            tma::expect_bytes(bar,
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

        // note that local_attn rt shape is 1x4 since it's done by a warpgroup. 
        // even though you might think 4x4 since q_s x k_s is (4x1) x (1x4). 
        warpgroup::mm_ABt(local_attn, q_s[tic], k_s[tic]); // clear registers -- note mm_ABt, not mma_ABt.
        warpgroup::mma_commit_group(); // dew it
        warpgroup::mma_async_wait(); // ding dong! matmuls arrived.

        // our goal is to store local_attn + (local_attn^2 / 2) in local_attn_bf
        if (add_scale > 0) { 
            mul(local_attn, local_attn, 0.25f);  // divide a1 term by sqrt(d)
        } 
        copy(temp_attn_accum, local_attn);

        // BEGIN comment-out for removing T2 (debug)
        mul(temp_attn_accum, temp_attn_accum, temp_attn_accum); // square it; this is divided by D now 
        if (add_scale > 0) { 
            mul(temp_attn_accum, temp_attn_accum, 0.5f);        // divide a2 term by 2
        } // divide by 2
        add(temp_attn_accum, temp_attn_accum, local_attn);      // add back in 1x for the linear term
        // END comment-out for removing T2 (debug)
        add(temp_attn_accum, temp_attn_accum, 1.f);             // cumulative sum for a0
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
        
        // loads
        rt_bf_1x1<> q_src; // the source 16x16 tiles -- we'll draw on these for future mul_slice's.
        warpgroup::load(q_src, q_s[tic]);
        if (add_scale > 0) { 
            mul(q_src, q_src, __float2bfloat16(0.25));          // divide by D for A2 here.
        } 
        rt_bf_4x1<> k_src_tmp;
        rt_fl_4x1<> k_src_tmp_fl;
        rt_bf_1x4<> k_src;
        load(k_src_tmp, k_s[tic]);
        copy(k_src_tmp_fl, k_src_tmp);
        transpose_sep(k_src, k_src_tmp); // transpose K into Kt
        
        // a1 mult
        warpgroup::mma_ABt(o, q_src, a1_trans_s);           // incorporate a1 onto o
        warpgroup::mma_commit_group();                      // dew it
        
        // a1 kv state
        warpgroup::mma_AtB(a1_trans, v_s_2[tic], k_s[tic]); // now 4 1x4 registers that need to be summed.
        warpgroup::mma_commit_group(); // dew it
        warpgroup::mma_async_wait();   // tmp
        warpgroup::store(a1_trans_s, a1_trans);

        if (add_scale > 0) { 
            mul(q_src, q_src, __float2bfloat16(0.70710678118)); // divide by 2 for A2 here; the mul_slices 
        }

        #pragma unroll
        for(int t = 0; t < 4; t++) {
            rt_bf_1x4<> q, k;
            mul_slice_row(q, q_src, t*4);        // Each warp handles 16 tokens of q, all features
            mul_slice_col(k, k_src, t*4+warpid); // Each warp handles all 64 tokens of k, but different features

            // take previous one and move up to smem for wgmma.
            warpgroup::store(a2_s, a2[t]); 
            __syncthreads();

            warpgroup::mma_fence(o);       // av matmul fence
            warpgroup::mma_fence(a2[t]);   // av matmul fence
            warpgroup::mma_AB(o, q, a2_s); // incorporate a1 onto o
            warpgroup::mma_commit_group(); // dew it
            
            // Note: we originally have k_src_tmp and transpose it (line 283 above)
            // this is becuase AtB function is only usable if A is in SMEM. 
            // but we'd like to keep k in register, so we just transpose it upfront
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

        warpgroup::store(o_s[tic], o);
        // do the cumulative sum last, after everything is stored
        accumulate_a0(a0_total, v_s_3[tic]); // cumulative sum of V onto O in shared memory

        // save the chunks of output
        if (block>0) tma::store_async_wait();
        if (warpid == 0) { // go get the next K from HBM
            tma::store_async(tma_o, o_s[tic], (blockIdx.x * n_blocks) + block); 
            tma::store_commit_group(); // dew it
        } tma::store_async_wait();
    }


    if (output_state > 0) {
        // save the KV state (A2)
        for (int rt = 0; rt < 4; rt++) {
            // reinterpret_cast doesn’t change the bits or memory layout of the variable a2_s. 
            // it tells the compiler to treat a2_s as a reference to the type we've specified.
            auto &kv_smem_2 = reinterpret_cast<tile_smem_bf_4x4&>(a2_s); // this layout is better for global HBM stores so we cast.
            if (add_scale > 0) { 
                mul(a2[rt], a2[rt], 0.70710678118); 
                mul(a2[rt], a2[rt], 0.70710678118); 
                mul(a2[rt], a2[rt], 0.25); 
            }
            warpgroup::store(kv_smem_2, a2[rt]); 
            __syncthreads();
            if (warpid == 0) {
                int tile_idx = (blockIdx.x * 4) + rt; 
                tma::store_async(tma_kv_a2, kv_smem_2, tile_idx); 
                tma::store_commit_group(); 
            }
            tma::store_async_wait();
        }

        // save the KV state A1
        auto &kv_smem_1 = reinterpret_cast<tile_smem_bf_4x1&>(a1_trans_s);
        if (add_scale > 0 ) { mul(a1_trans, a1_trans, 0.5); } // divides by math.sqrt(math.sqrt(D_QK))
        warpgroup::store(kv_smem_1, a1_trans);                // from individual warps to shared address
        __syncthreads();
        if (warpid == 0) {                                    // one warp takes care of the write to HBM
            tma::store_async(tma_kv_a1, kv_smem_1, blockIdx.x); 
            tma::store_commit_group(); 
        }
        tma::store_async_wait();

        // save the KV state A0
        if (warpid == 0) {   
            tma::store_async(tma_kv_a0, a0_total, blockIdx.x); 
            tma::store_commit_group(); 
        }
        tma::store_async_wait();
    }
}


#ifdef TORCH_COMPILE
#include "common/pyutils/torch_helpers.cuh"
#include <iostream>
void based_linear_prefill(
    int add_scale, int output_state,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, 
    torch::Tensor kv_a2, torch::Tensor kv_a1, torch::Tensor kv_a0
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    CHECK_INPUT(kv_a2);
    CHECK_INPUT(kv_a1);
    CHECK_INPUT(kv_a0);

    auto batch   = q.size(0);
    auto heads   = q.size(1);
    auto threads = NUM_THREADS; 
    auto n       = q.size(2); 

    bool k_same  = true; 
    bool o_same  = true; 
    for (auto i = 0; i < 4; i++) {
        k_same &= q.size(i) == k.size(i);
        o_same &= v.size(i) == o.size(i);
    }

    TORCH_CHECK(k_same, "q and k must have the same shape");
    TORCH_CHECK(o_same, "v and o must have the same shape");
    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "k must be bf16");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "v must be bf16");
    TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "o must be bf16");
    TORCH_CHECK(n % (NUM_WORKERS*kittens::TILE_DIM) == 0, "n must be divisible by 64");

    // convert to bf16
    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();
    c10::BFloat16* o_ptr = o.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kv_a2_ptr    = kv_a2.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kv_a1_ptr    = kv_a1.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kv_a0_ptr    = kv_a0.data_ptr<c10::BFloat16>();

    const bf16* d_q  = reinterpret_cast<const bf16*>(q_ptr);
    const bf16* d_k  = reinterpret_cast<const bf16*>(k_ptr);
    const bf16* d_v  = reinterpret_cast<const bf16*>(v_ptr);
          bf16* d_o  = reinterpret_cast<bf16*>(o_ptr);
          bf16* d_kv_a2 = reinterpret_cast<bf16*>(kv_a2_ptr);
          bf16* d_kv_a1 = reinterpret_cast<bf16*>(kv_a1_ptr); 
          bf16* d_kv_a0 = reinterpret_cast<bf16*>(kv_a0_ptr);

    CUtensorMap* tma_q_d   = tma::allocate_and_create_tensor_map<kittens::st_bf_4x1>   (d_q, (batch*heads*n)/(4 * kittens::TILE_DIM)); 
    CUtensorMap* tma_k_d   = tma::allocate_and_create_tensor_map<tile_smem_bf_4x1>(d_k, (batch*heads*n)/(4 * kittens::TILE_DIM)); 
    CUtensorMap* tma_v_d   = tma::allocate_and_create_tensor_map<tile_smem_bf_4x4_wgmma_interleave>(d_v, (batch*heads*n)/(tile_smem_bf_4x4_wgmma_interleave::rows)); 
    CUtensorMap* tma_v_3_d = tma::allocate_and_create_tensor_map<tile_smem_bf_4x4>(d_v, (batch*heads*n)/(4* kittens::TILE_DIM)); 
    CUtensorMap* tma_o_d   = tma::allocate_and_create_tensor_map<tile_smem_bf_4x4>(d_o, (batch*heads*n)/(4* kittens::TILE_DIM)); 
    
    // kv state maps
    CUtensorMap* tma_kv_a2_d = tma::allocate_and_create_tensor_map<tile_smem_bf_4x4> (d_kv_a2, batch*heads*(D_QK*D_QK));
    CUtensorMap* tma_kv_a1_d = tma::allocate_and_create_tensor_map<tile_smem_bf_4x1> (d_kv_a1, batch*heads*(D_QK));
    CUtensorMap* tma_kv_a0_d = tma::allocate_and_create_tensor_map<vec_smem_bf_4> (d_kv_a0, batch*heads);

    unsigned long mem_size = 98000; 
    using T = kittens::bf16;
    using H = kittens::bf16;

    cudaFuncSetAttribute(
        based_linear_attention,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    ); 

    based_linear_attention<<<batch*heads, threads, mem_size>>>(
        n, add_scale, output_state, 
        tma_q_d, tma_k_d, tma_v_d, tma_v_3_d, tma_o_d, 
        tma_kv_a2_d, tma_kv_a1_d, tma_kv_a0_d 
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
}
#else
#include "harness.impl"
#endif