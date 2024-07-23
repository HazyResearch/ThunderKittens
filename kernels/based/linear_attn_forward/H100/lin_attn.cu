#include "kittens.cuh"
#include "based_utils.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

using namespace kittens;
namespace cg = cooperative_groups;

template<bool add_scale>
__global__ __launch_bounds__(NUM_THREADS, 1)
void based_linear_attention(
    int n,
    CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o, 
    CUtensorMap* tma_kv_a2, CUtensorMap* tma_kv_a1, CUtensorMap* tma_kv_a0 //,
    // CUtensorMap* tma_k_a2, CUtensorMap* tma_k_a1
) {
    int laneid = kittens::laneid(); // who am i? when am i?
    int warpid = kittens::warpid(); // who am i? when am i?
    int warpgroupid = warpid / 4;
    // cg::thread_group cg_wg = cg::tiled_partition(cg::this_thread_block(), kittens::WARPGROUP_THREADS);
    int tic = 0, toc = 1;
    int n_blocks = n / (q_tile::rows); // how many iters?

    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    q_tile (&q_smem)[2]      = al.allocate<q_tile, 2>(); //  4096 bytes
    k_tile (&k_smem)[2]      = al.allocate<k_tile, 2>(); //  4096 bytes
    v_tile (&v_smem)[2]      = al.allocate<v_tile, 2>(); // 16384 bytes
    o_tile (&o_smem)[2]      = al.allocate<o_tile, 2>(); // 16384 bytes
    // o_tile (&zero_o_smem)    = al.allocate<o_tile>(); // for clearing the next o_smem

    a0_vec        (&a0_smem)               = al.allocate<a0_vec>       ();
    a1_trans_tile (&a1_trans_smem)         = al.allocate<a1_trans_tile>(); // 2048 bytes
    a2_tile       (&a2_smem_working)[2][2] = al.allocate<a2_tile, 2, 2>   (); // 8192*4 bytes in use, for matmul

    // zero a2
    if(warpgroupid < 2) {
        for(int j = 0; j < 2; j++) {
            warpgroup::zero(a2_smem_working[warpgroupid][j]);
        }
    }    
    if(warpgroupid == 2) {
        // zero a0
        warpgroup::zero(a0_smem);
        // zero a1
        warpgroup::zero(a1_trans_smem);
        // zero the clearing smem for a2
        // warpgroup::zero(zero_o_smem);
    }

    // initial load
    __shared__ barrier qk_bar[2], v_bar[2], compute_done_bar[2];
    if (warpid == 0) {
        init_barrier(qk_bar[tic], 0, 1);
        init_barrier(qk_bar[toc], 0, 1);
        tma::expect_bytes(qk_bar[tic],
            size_bytes<typeof(q_smem[0])> +
            size_bytes<typeof(k_smem[0])>
        );
        int tile_idx = blockIdx.x * n_blocks; // submit initial loads
        tma::load_async(q_smem[tic], tma_q, qk_bar[tic], tile_idx);
        tma::load_async(k_smem[tic], tma_k, qk_bar[tic], tile_idx);
    }
    else if(warpid == 1) {
        init_barrier(v_bar[tic], 0, 1);
        init_barrier(v_bar[toc], 0, 1);
        tma::expect_bytes(v_bar[tic],
            size_bytes<typeof(v_smem[0])>
        );
        int tile_idx = blockIdx.x * n_blocks; // submit initial loads
        tma::load_async(v_smem[tic], tma_v, v_bar[tic], tile_idx);
    }
    else if(warpid == 2) {
        init_barrier(compute_done_bar[tic], 8, 0); // 8 consumer warps need to sync
        init_barrier(compute_done_bar[toc], 8, 0); // 8 consumer warps need to sync
    }
    __syncthreads();

    if(warpgroupid == 2) { // producer
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(24));
        if(warpid == 8) {
            for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) {
                int next_tile_idx = (blockIdx.x * n_blocks) + block + 1;
                tma::expect_bytes(qk_bar[toc],
                    size_bytes<typeof(q_smem[0])> +
                    size_bytes<typeof(k_smem[0])>
                );
                tma::load_async(q_smem[toc], tma_q, qk_bar[toc], next_tile_idx);
                tma::load_async(k_smem[toc], tma_k, qk_bar[toc], next_tile_idx);
                tma::expect_bytes(v_bar[toc],
                    size_bytes<typeof(v_smem[0])>
                );
                tma::load_async(v_smem[toc], tma_v, v_bar[toc], next_tile_idx);
                wait(compute_done_bar[tic], (block/2)%2); // wait for consumers to all be done
            }
        }
    }
    else if(warpgroupid == 0) { // consumer 1
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(224));

        // this is going to handle local attention and its part of a2
        rt_fl_1x4<> a2_reg[2]; // 64 registers
        zero(a2_reg[0]);
        zero(a2_reg[1]);

        for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) { // FIRST CONSUMER -- LOCAL FOCUSED
            rt_fl_1x4<> local_attn, temp_attn_accum; // 32 registers each -- 64. (But should be optimized down to ~40)
            rt_bf_1x4<> local_attn_bf; // 4 registers each -- 16
            rt_fl_1x4<> o_reg; // 32 registers

            wait(qk_bar[tic], (block/2)%2);

            warpgroup::mma_fence(local_attn); // qk matmul fence
            warpgroup::mm_ABt(local_attn, q_smem[tic], k_smem[tic]); // clear registers -- note mm_ABt, not mma_ABt.
            warpgroup::mma_commit_group(); // dew it

            wait(v_bar[tic], (block/2)%2);

            warpgroup::mma_async_wait();

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
                if (j>warpid) zero(reinterpret_cast<rt_bf_1x1<>&>(local_attn_bf.tiles[0][j]));
                else if (j==warpid) make_causal(local_attn_bf.tiles[0][j], local_attn_bf.tiles[0][j], kittens::base_types::constants<bf16>::zero());
            }

            warpgroup::mma_fence(o_reg);                            // av matmul fence
            warpgroup::mm_AB(o_reg, local_attn_bf, v_smem[tic]);   // reset o_reg here, and do local chunk.
            warpgroup::mma_commit_group();                          // dew it
            warpgroup::mma_async_wait();

            // ----- now common stuff around a2 and store out -----

            rt_bf_1x1<> q_src; // the source 16x16 tiles -- we'll draw on these for future mul_slice's.
            warpgroup::load(q_src, q_smem[tic]);
            mul(q_src, q_src, __float2bfloat16(0.70710678118)); // divide by 2 for A2 here. But do it as sqrt(2)
            if constexpr (add_scale > 0) { mul(q_src, q_src, __float2bfloat16(0.25)); } // divide by sqrt(sqrt(d=16)) for A2 here, but on both Q and K, but just do it on Q
            
            rt_bf_4x1<> k_src_tmp;
            load(k_src_tmp, k_smem[tic]);
            rt_bf_1x4<> k_src_t;
            transpose_sep(k_src_t, k_src_tmp); // transpose K into Kt

            asm volatile("bar.sync 0, 128;\n");
            __threadfence_block(); // need memory to hit before we can go into this loop

            #pragma unroll
            for(int t = 0; t < 2; t++) {
                rt_bf_1x4<> q, k_t;
                mul_slice_row(q,   q_src,   8*t); // 0...3 and 8...11
                
                warpgroup::mma_fence(o_reg); // av matmul fence
                warpgroup::mma_AB(o_reg, q, a2_smem_working[warpgroupid][t]); // incorporate a1 onto o
                warpgroup::mma_commit_group(); // dew it

                mul_slice_col(k_t, k_src_t, 8*t + warpid); // 0...3 and 8...11
                
                warpgroup::mma_fence(a2_reg[t]); // av matmul fence
                warpgroup::mma_AB(a2_reg[t], k_t, v_smem[tic]); // incorporate KtV onto a2
                warpgroup::mma_commit_group(); // dew it
                warpgroup::mma_async_wait(); // ding dong! o matmuls have now arrived, too.
            }

            if(laneid == 0) arrive(compute_done_bar[tic]); // we have finished what we need with q, k, v

            asm volatile("bar.sync 0, 128;\n");
            warpgroup::store(o_smem[warpgroupid], o_reg);
            __threadfence_block(); // need memory to hit before we can launch tma store
            if (warpid == 0) { // launch o as a store_add_async, since we don't seem to be memory bound anyways
                tma::store_add_async(tma_o, o_smem[warpgroupid], (blockIdx.x * n_blocks) + block); 
                tma::store_commit_group(); // dew it
            }
            // store smem for next time around
            #pragma unroll
            for(int t = 0; t < 2; t++) {
                warpgroup::store(a2_smem_working[warpgroupid][t], a2_reg[t]); // take previous one and move up to smem for wgmma.
            }
            tma::store_async_wait();
        }
    }
    else if(warpgroupid == 1) { // consumer 2
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(256));

        // this is going to handle a0, a1 and its part of a2
        rt_fl_1x1<> a1_trans_reg; //  8 registers -- transposed chunk of a1. This one needs to be persistent
        rt_fl_1x4<> a2_reg[2];    // 64 registers
        zero(a1_trans_reg);
        zero(a2_reg[0]);
        zero(a2_reg[1]);

        // gprintf("(warp %d) outside loop\n", warpid);
        for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) { // SECOND CONSUMER -- A1 FOCUSED
            // gprintf("(warp %d) inside loop %d\n", warpid, block);
            rt_fl_1x4<> o_reg; // 32 registers

            wait(qk_bar[tic], (block/2)%2);

            __threadfence_block(); // we need to make sure the a1_trans_smem is loaded before we launch the matmul
            warpgroup::mma_fence(o_reg);                            // q@a1 matmul fence
            warpgroup::mm_ABt(o_reg, q_smem[tic], a1_trans_smem);  // incorporate the last a1, in smem, onto o_reg
            warpgroup::mma_commit_group();
            wait(v_bar[tic], (block/2)%2); // wait on V
            warpgroup::mma_fence(a1_trans_reg);                   // a1 accumulation matmul fence
            warpgroup::mma_AtB(a1_trans_reg, v_smem[tic], k_smem[tic]); // we now have 4 1x4 registers that need to eventually be summed.
            warpgroup::mma_commit_group(); // dew it
            #pragma unroll
            for(int t = 0; t < 2; t++) {
                warpgroup::store(a2_smem_working[warpgroupid][t], a2_reg[t]); // take previous one and move up to smem for wgmma.
            }
            warpgroup::mma_async_wait();

            // ----- now common stuff around a2 and store out -----

            rt_bf_1x1<> q_src; // the source 16x16 tiles -- we'll draw on these for future mul_slice's.
            warpgroup::load(q_src, q_smem[tic]);
            mul(q_src, q_src, __float2bfloat16(0.70710678118)); // divide by 2 for A2 here. But do it as sqrt(2)
            if constexpr (add_scale > 0) { mul(q_src, q_src, __float2bfloat16(0.25)); } // divide by sqrt(sqrt(d=16)) for A2 here, but on both Q and K, but just do it on Q
            
            rt_bf_4x1<> k_src_tmp;
            load(k_src_tmp, k_smem[tic]);
            rt_bf_1x4<> k_src_t;
            transpose_sep(k_src_t, k_src_tmp); // transpose K into Kt

            asm volatile("bar.sync 1, 128;\n");
            __threadfence_block(); // need memory to hit before we can go into this loop

            #pragma unroll
            for(int t = 0; t < 2; t++) {
                rt_bf_1x4<> q, k_t;
                mul_slice_row(q,   q_src,   8*t + 4); // 4...7 and 12...15
                
                warpgroup::mma_fence(o_reg); // av matmul fence
                warpgroup::mma_AB(o_reg, q, a2_smem_working[warpgroupid][t]); // incorporate a2 onto o
                warpgroup::mma_commit_group(); // dew it

                mul_slice_col(k_t, k_src_t, 8*t + warpid); // 4...7 and 12...15
                
                warpgroup::mma_fence(a2_reg[t]); // av matmul fence
                warpgroup::mma_AB(a2_reg[t], k_t, v_smem[tic]); // incorporate KtV onto a2
                warpgroup::mma_commit_group(); // dew it
                warpgroup::mma_async_wait(); // ding dong! o matmuls have now arrived, too.
            }

            // while we matmul, let's initialize o_reg with the previous a0
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                #pragma unroll
                for(int j = 0; j < 2; j++) {
                    int col = i*16 + j*8 + (laneid%4)*2;
                    float2 data;
                    kittens::move<float2>::lds(data, &a0_smem[col]);
                    o_reg.tiles[0][i].data[2*j].x   += data.x;
                    o_reg.tiles[0][i].data[2*j].y   += data.y;
                    o_reg.tiles[0][i].data[2*j+1].x += data.x;
                    o_reg.tiles[0][i].data[2*j+1].y += data.y;
                }
            }

            warpgroup::store(o_smem[warpgroupid], o_reg);
            asm volatile("bar.sync 1, 128;\n");
            __threadfence_block(); // need memory to hit before we can launch tma store
            // save the chunks of output
            if (warpid == 4) { // launch o as a store_add_async, since we don't seem to be memory bound anyways
                tma::store_add_async(tma_o, o_smem[warpgroupid], (blockIdx.x * n_blocks) + block); 
                tma::store_commit_group(); // dew it
            }
            // store smem for next time around
            warpgroup::store(a1_trans_smem, a1_trans_reg);
            accumulate_a0(a0_smem, v_smem[tic]); // while we wait for matmul, do the cumulative sum for the next iteration
            if(laneid == 0) arrive(compute_done_bar[tic]); // we have finished what we need with q, k, v
            tma::store_async_wait();
        }
    }

    // constexpr int output_state = 0;
    // if constexpr (output_state > 0) {
    //     // if (warpid == 0 && threadIdx.x == 0) {printf("output state");}

    //     // save the KV state (A2)
    //     for (int rt = 0; rt < 4; rt++) {
    //         // The reinterpret_cast doesn’t change the bits or memory layout of the variable a2_s. 
    //         // Instead, it tells the compiler to treat the memory location of a2_s as if it were a reference to st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>.
    //         auto &kv_smem_2 = *reinterpret_cast<kv_a2_tile*>(&a2_smem[0]); // this layout is better for global HBM stores so we cast.
    //         rt_fl_1x4<> a2_reg_fl;
    //         warpgroup::load(a2_reg_fl, a2_smem[rt]);
    //         mul(a2_reg_fl, a2_reg_fl, 0.70710678118); // Taylor normalization
    //         if constexpr (add_scale) {
    //             mul(a2_reg_fl, a2_reg_fl, 0.25);
    //         }
    //         warpgroup::store(kv_smem_2, a2_reg_fl); 
    //         __syncthreads();
    //         if (warpid == 0) {
    //             int tile_idx = (blockIdx.x * 4) + rt; 
    //             tma::store_async(tma_kv_a2, kv_smem_2, tile_idx); 
    //             tma::store_commit_group(); 
    //         }
    //         tma::store_async_wait();
    //     }

    //     // save the KV state A1
    //     auto &kv_smem_1 = *reinterpret_cast<kv_a1_tile*>(&a1_trans_smem);
    //     if constexpr (add_scale) {
    //         // if (warpid==0 and threadIdx.x==0) {printf("scale 4.");} 
    //         mul(a1_trans_reg, a1_trans_reg, 0.5);       // divides by math.sqrt(math.sqrt(D_QK))
    //     }
    //     warpgroup::store(kv_smem_1, a1_trans_reg);   // from individual warps to shared address
    //     __syncthreads();
    //     if (warpid == 0) {      // one warp takes care of the write to HBM
    //         int tile_idx = (blockIdx.x); 
    //         tma::store_async(tma_kv_a1, kv_smem_1, tile_idx); 
    //         tma::store_commit_group(); 
    //     }
    //     tma::store_async_wait();

    //     // save the KV state A0
    //     if (warpid == 0) {      // one warp takes care of the write to HBM
    //         int tile_idx = (blockIdx.x); 
    //         tma::store_async(tma_kv_a0, a0_smem, tile_idx); 
    //         tma::store_commit_group(); 
    //     }
    //     tma::store_async_wait();

    // }
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

