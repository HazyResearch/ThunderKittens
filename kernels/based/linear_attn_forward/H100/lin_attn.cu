#include "kittens.cuh"
#include "based_utils.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

using namespace kittens;

template<bool add_scale>
__global__ __launch_bounds__(NUM_THREADS, 2)
void based_linear_attention(
    int n,
    CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o, 
    CUtensorMap* tma_kv_a2, CUtensorMap* tma_kv_a1, CUtensorMap* tma_kv_a0 //,
    // CUtensorMap* tma_k_a2, CUtensorMap* tma_k_a1
) {
    int laneid = kittens::laneid(); // who am i? when am i?
    int warpid = kittens::warpid(); // who am i? when am i?
    int tic = 0, toc = 1;
    int n_blocks = n / (q_tile::rows); // how many iters?

    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    q_tile (&q_smem)  [2]   = al.allocate<q_tile, 2>(); // 4096 bytes
    k_tile (&k_smem)  [2]   = al.allocate<k_tile, 2>(); // 4096 bytes
    v_tile (&v_smem)  [2]   = al.allocate<v_tile, 2>(); // 16384 bytes
    o_tile (&o_smem)  [2]   = al.allocate<o_tile, 2>(); // 16384 bytes

    rt_fl_1x1<> a1_trans_reg; // transposed chunk of a1. Consumes 8 registers.
    rt_fl_1x4<> a2_reg[4]; // a2 gets propagated through here. Consumes 128 registers.

    a0_vec        (&a0_smem)       = al.allocate<a0_vec>();
    a1_trans_tile (&a1_trans_smem) = al.allocate<a1_trans_tile>(); // 2048 bytes
    a2_tile       (&a2_smem)       = al.allocate<a2_tile>(); // 8192 bytes. I wish we could buffer the STS instructions but we don't seem to have the registers.

    // zero a0
    if(warpid == 0) {
        zero(a0_smem);
    }
    // zero a1
    zero(a1_trans_reg);
    // zero a2
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        zero(a2_reg[i]); // everyone zeroes a2.
    }

    // initial load
    __shared__ barrier bar;
    if (warpid == 0) {
        init_barrier(bar, 0, 1);
        tma::expect_bytes(bar,
            size_bytes<typeof(q_smem[0])> +
            size_bytes<typeof(k_smem[0])> +
            size_bytes<typeof(v_smem[0])>
        );
        int tile_idx = blockIdx.x * n_blocks; // submit initial loads
        tma::load_async(q_smem[tic], tma_q, bar, tile_idx);
        tma::load_async(k_smem[tic], tma_k, bar, tile_idx);
        tma::load_async(v_smem[tic], tma_v, bar, tile_idx);
    }
    __syncthreads();
    // gprintf(RED "FINISHED INIT\n" RESET);

    for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) {
        rt_bf_1x4<> local_attn_bf; // 4 registers each -- 16
        rt_fl_1x4<> local_attn, temp_attn_accum; // 32 registers each -- 64
        rt_fl_1x4<> o; // 32 registers

        // arrive memory
        wait(bar, tic);
        // the o_smem store synchronization means we don't need to do it again, here. Everyone *must* be done by then.
        if (warpid == 0 && block+1<n_blocks) { // go get the next K from HBM
            tma::expect_bytes(bar,
                size_bytes<typeof(q_smem[0])> +
                size_bytes<typeof(k_smem[0])> +
                size_bytes<typeof(v_smem[0])>
            );
            int next_tile_idx = (blockIdx.x * n_blocks) + block + 1;
            tma::load_async(q_smem[toc], tma_q, bar, next_tile_idx);
            tma::load_async(k_smem[toc], tma_k, bar, next_tile_idx);
            tma::load_async(v_smem[toc], tma_v, bar, next_tile_idx);
        }

        // we start by doing the very local computations. Then, we'll follow up later with the rest.
        __syncthreads();
        warpgroup::mma_fence(local_attn); // qk matmul fence
        warpgroup::mm_ABt(local_attn, q_smem[tic], k_smem[tic]); // clear registers -- note mm_ABt, not mma_ABt.
        warpgroup::mma_commit_group(); // dew it

        // while we matmul, let's initialize o with the previous a0
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            #pragma unroll
            for(int j = 0; j < 2; j++) {
                int col = i*16 + j*8 + (laneid%4)*2;
                float2 data = *(float2*)&a0_smem[col];
                o.tiles[0][i].data[2*j].x   += data.x;
                o.tiles[0][i].data[2*j].y   += data.y;
                o.tiles[0][i].data[2*j+1].x += data.x;
                o.tiles[0][i].data[2*j+1].y += data.y;
            }
        }

        // and then, let's store a1_trans_reg into smem
        warpgroup::store(a1_trans_smem, a1_trans_reg);

        warpgroup::mma_async_wait(); // ding dong! matmuls arrived.

        __syncthreads(); // need smem store to be visible
        warpgroup::mma_fence(o);                            // q@a1 matmul fence
        warpgroup::mma_ABt(o, q_smem[tic], a1_trans_smem);  // incorporate the last a1, in smem, onto o
        warpgroup::mma_commit_group();                      // dew it

        // we want to overlap the softmax with this matmul.

        // our goal is to store local_attn + (local_attn^2 / 2) in local_attn_bf
        if constexpr (add_scale > 0) { mul(local_attn, local_attn, 0.25f); } // divide by sqrt(d)
        copy(temp_attn_accum, local_attn);
        mul(temp_attn_accum, temp_attn_accum, temp_attn_accum); // square it; note this converts sqrt(d) to d
        mul(temp_attn_accum, temp_attn_accum, 0.5f);            // divide by 2
        add(temp_attn_accum, temp_attn_accum, local_attn);      // add back in 1x for the linear term
        add(temp_attn_accum, temp_attn_accum, 1.f);             // cumulative sum for a0
        copy(local_attn_bf, temp_attn_accum); // now stored.

        // now make causal
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            if (j>warpid) {
                #pragma unroll
                for(int i = 0; i < 4; i++) {
                    local_attn_bf.tiles[0][j].data[i] = kittens::base_types::constants<bf16_2>::zero();
                }
            }
            else if (j==warpid) {
                make_causal(local_attn_bf.tiles[0][j], local_attn_bf.tiles[0][j], kittens::base_types::constants<bf16>::zero());
            }
        }

        warpgroup::mma_async_wait(); // o

        warpgroup::mma_fence(o);                            // av matmul fence
        warpgroup::mm_AB(o, local_attn_bf, v_smem[tic]);    // reset o here, and do local chunk.
        warpgroup::mma_commit_group();                      // dew it

        accumulate_a0(a0_smem, v_smem[tic]); // while we wait for matmul, do the cumulative sum for the next iteration

        warpgroup::mma_async_wait(); // clear

        // a1 accumulation
        __syncthreads();
        warpgroup::mma_fence(a1_trans_reg);                   // a1 accumulation matmul fence
        warpgroup::mma_AtB(a1_trans_reg, v_smem[tic], k_smem[tic]); // we now have 4 1x4 registers that need to eventually be summed.
        warpgroup::mma_commit_group(); // dew it

        rt_bf_1x1<> q_src; // the source 16x16 tiles -- we'll draw on these for future mul_slice's.
        warpgroup::load(q_src, q_smem[tic]);
        mul(q_src, q_src, __float2bfloat16(0.70710678118)); // divide by 2 for A2 here. But do it as sqrt(2)
        if constexpr (add_scale > 0) { mul(q_src, q_src, __float2bfloat16(0.25)); } // divide by sqrt(sqrt(d=16)) for A2 here, but on both Q and K, but just do it on Q
        rt_bf_4x1<> k_src_tmp;
        rt_bf_1x4<> k_src;
        load(k_src_tmp, k_smem[tic]);
        transpose_sep(k_src, k_src_tmp); // transpose K into Kt

        warpgroup::mma_async_wait();

        // 2nd order taylor
        #pragma unroll 4
        for(int t = 0; t < 4; t++) {
            rt_bf_1x4<> q, k;
            mul_slice_row(q, q_src, t*4);
            mul_slice_col(k, k_src, t*4+warpid);

            warpgroup::store(a2_smem, a2_reg[t]); // take previous one and move up to smem for wgmma.
            __syncthreads();
            warpgroup::mma_fence(o); // av matmul fence
            warpgroup::mma_fence(a2_reg[t]); // av matmul fence
            warpgroup::mma_AB(o, q, a2_smem); // incorporate a1 onto o
            warpgroup::mma_commit_group(); // dew it
            warpgroup::mma_async_wait(); // ding dong! o matmuls have now arrived, too.
            
            warpgroup::mma_AB(a2_reg[t], k, v_smem[tic]); // incorporate KtV onto a2
            warpgroup::mma_commit_group(); // dew it
            
            warpgroup::mma_async_wait(); // ding dong! o matmuls have now arrived, too.
        }

        // do the cumulative sum last, after everything is stored
        warpgroup::store(o_smem[tic], o);
        __syncthreads();

        // save the chunks of output
        if (block>0) tma::store_async_wait();
        if (warpid == 0) { // go get the next K from HBM
            tma::store_async(tma_o, o_smem[tic], (blockIdx.x * n_blocks) + block); 
            tma::store_commit_group(); // dew it
        }
        // gprintf(GREEN "OUTPUT STORED\n" RESET);
    }
    // gprintf(BLUE "LOOP FINISHED\n" RESET);

    constexpr int output_state = 0;
    if (output_state > 0) {
        // if (warpid == 0 && threadIdx.x == 0) {printf("output state");}

        // save the KV state (A2)
        for (int rt = 0; rt < 4; rt++) {
            // The reinterpret_cast doesn’t change the bits or memory layout of the variable a2_s. 
            // Instead, it tells the compiler to treat the memory location of a2_s as if it were a reference to st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>.
            auto &kv_smem_2 = *reinterpret_cast<kv_a2_tile*>(&a2_smem[0]); // this layout is better for global HBM stores so we cast.
            mul(a2_reg[rt], a2_reg[rt], 0.70710678118); // Taylor normalization
            if constexpr (add_scale) {
                mul(a2_reg[rt], a2_reg[rt], 0.25);
            }
            warpgroup::store(kv_smem_2, a2_reg[rt]); 
            __syncthreads();
            if (warpid == 0) {
                int tile_idx = (blockIdx.x * 4) + rt; 
                tma::store_async(tma_kv_a2, kv_smem_2, tile_idx); 
                tma::store_commit_group(); 
            }
            tma::store_async_wait();
        }

        // save the KV state A1
        auto &kv_smem_1 = *reinterpret_cast<kv_a1_tile*>(&a1_trans_smem);
        if constexpr (add_scale) {
            // if (warpid==0 and threadIdx.x==0) {printf("scale 4.");} 
            mul(a1_trans_reg, a1_trans_reg, 0.5);       // divides by math.sqrt(math.sqrt(D_QK))
        }
        warpgroup::store(kv_smem_1, a1_trans_reg);   // from individual warps to shared address
        __syncthreads();
        if (warpid == 0) {      // one warp takes care of the write to HBM
            int tile_idx = (blockIdx.x); 
            tma::store_async(tma_kv_a1, kv_smem_1, tile_idx); 
            tma::store_commit_group(); 
        }
        tma::store_async_wait();

        // save the KV state A0
        if (warpid == 0) {      // one warp takes care of the write to HBM
            int tile_idx = (blockIdx.x); 
            tma::store_async(tma_kv_a0, a0_smem, tile_idx); 
            tma::store_commit_group(); 
        }
        tma::store_async_wait();

    }
}

#include "harness.impl"

// #include "src/common/pyutils/torch_helpers.cuh"
// #include <iostream>
// void based_fwd_tk(
//     int add_scale, int output_state,
//     torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, 
//     torch::Tensor kv_a2, torch::Tensor kv_a1, torch::Tensor kv_a0
// ) {
//     CHECK_INPUT(q);
//     CHECK_INPUT(k);
//     CHECK_INPUT(v);
//     CHECK_INPUT(o);
//     CHECK_INPUT(kv_a2);
//     CHECK_INPUT(kv_a1);
//     CHECK_INPUT(kv_a0);

//     auto batch   = q.size(0);
//     auto heads   = q.size(1);
//     auto threads = NUM_THREADS; 
//     auto n       = q.size(2); 

//     bool k_same  = true; 
//     bool o_same  = true; 
//     for (auto i = 0; i < 4; i++) {
//         k_same &= q.size(i) == k.size(i);
//         o_same &= v.size(i) == o.size(i);
//     }

//     TORCH_CHECK(k_same, "q and k must have the same shape");
//     TORCH_CHECK(o_same, "v and o must have the same shape");
//     TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "q must be bf16");
//     TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "k must be bf16");
//     TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "v must be bf16");
//     TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "o must be bf16");
//     TORCH_CHECK(n % (NUM_WORKERS*kittens::TILE_DIM) == 0, "n must be divisible by 64");

//     // convert to bf16
//     c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
//     c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
//     c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();
//     c10::BFloat16* o_ptr = o.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *kv_a2_ptr    = kv_a2.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *kv_a1_ptr    = kv_a1.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *kv_a0_ptr    = kv_a0.data_ptr<c10::BFloat16>();

//     const bf16* d_q  = reinterpret_cast<const bf16*>(q_ptr);
//     const bf16* d_k  = reinterpret_cast<const bf16*>(k_ptr);
//     const bf16* d_v  = reinterpret_cast<const bf16*>(v_ptr);
//           bf16* d_o  = reinterpret_cast<bf16*>(o_ptr);
//           bf16* d_kv_a2 = reinterpret_cast<bf16*>(kv_a2_ptr);
//           bf16* d_kv_a1 = reinterpret_cast<bf16*>(kv_a1_ptr); 
//           bf16* d_kv_a0 = reinterpret_cast<bf16*>(kv_a0_ptr);

//     CUtensorMap* tma_q_d   = tma::allocate_and_create_tensor_map<kittens::st_bf_4x1<wgmma_swizzle_l>>   (d_q, (batch*heads*n)/(4 * kittens::TILE_DIM)); 
//     CUtensorMap* tma_k_d   = tma::allocate_and_create_tensor_map<kittens::st_bf_4x1<wgmma_interleave_l>>(d_k, (batch*heads*n)/(4 * kittens::TILE_DIM)); 
//     CUtensorMap* tma_v_d   = tma::allocate_and_create_tensor_map<kittens::st_bf_4x4<wgmma_interleave_l>>(d_v, (batch*heads*n)/(kittens::st_bf_4x4<wgmma_interleave_l>::rows)); 
//     CUtensorMap* tma_v_3_d = tma::allocate_and_create_tensor_map<kittens::st_bf_4x4<swizzle_l>>         (d_v, (batch*heads*n)/(4 * kittens::TILE_DIM)); 
//     CUtensorMap* tma_o_d   = tma::allocate_and_create_tensor_map<kittens::st_bf_4x4<swizzle_l>>         (d_o, (batch*heads*n)/(4 * kittens::TILE_DIM)); 
//     CUtensorMap* tma_kv_a2_d = tma::allocate_and_create_tensor_map<tile_kv_a2_2_smem> (d_kv_a2, batch*heads*(D_QK*D_QK));
//     CUtensorMap* tma_kv_a1_d = tma::allocate_and_create_tensor_map<tile_kv_a1_2_smem> (d_kv_a1, batch*heads*(D_QK));
//     CUtensorMap* tma_kv_a0_d = tma::allocate_and_create_tensor_map<tile_kv_a0_2_smem> (d_kv_a0, batch*heads);

//     unsigned long mem_size = 98000; 
    
//     using T = kittens::bf16;
//     using H = kittens::bf16;

//     cudaFuncSetAttribute(
//         based_linear_attention,
//         cudaFuncAttributeMaxDynamicSharedMemorySize,
//         mem_size
//     ); 

//     based_linear_attention<<<batch*heads, threads, mem_size>>>(
//         n, add_scale, output_state, 
//         tma_q_d, tma_k_d, tma_v_d, tma_v_3_d, tma_o_d, 
//         tma_kv_a2_d, tma_kv_a1_d, tma_kv_a0_d
//     );

//     CHECK_CUDA_ERROR(cudaGetLastError());
// }

