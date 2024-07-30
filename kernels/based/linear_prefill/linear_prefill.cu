#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include "based_utils.cuh"

#define NUM_WORKERS (12) // hardcoded, don't change
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define D_QK (16) // hardcoded, don't change
#define D_VO (64) // hardcoded but can be changed with some effort

// the overall algorithm partitions across 1 warpgroup of 4 workers.
__global__ __launch_bounds__(NUM_THREADS, 1)
void based_linear_attention(
    int include_a0, int include_a1, int include_a2,
    int n, int add_scale, int add_norm, int output_state, 
    CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o, 
    CUtensorMap* tma_kv_a2, CUtensorMap* tma_kv_a1, CUtensorMap* tma_kv_a0, 
    CUtensorMap* tma_k_a2, CUtensorMap* tma_k_a1
) {

    int laneid      = kittens::laneid(); // who am i? when am i?
    int warpid      = kittens::warpid(); // who am i? when am i?
    int warpgroupid = warpid / 4;
    int tic = 0, toc = 1;
    int n_blocks = n / (st_bf_4x1::rows);

    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_bf_4x1 (&q_s)  [2]   = al.allocate<st_bf_4x1, 2>(); // 4096 bytes
    st_bf_4x1 (&k_s)  [2]   = al.allocate<st_bf_4x1, 2>();              // 4096 bytes
    st_bf_4x4 (&v_s)  [2]   = al.allocate<st_bf_4x4, 2>(); // 16384 bytes
    st_bf_4x4 (&v_s_2)[2]   = al.allocate<st_bf_4x4, 2>(); // 16384 bytes -- needed to prevent wgmma from breaking
    st_bf_4x4 (&v_s_3)[2]   = al.allocate<st_bf_4x4, 2>(); // 16384 bytes -- used to reduce bank conflicts for a0 sum
    st_bf_4x4 (&o_s)  [2]   = al.allocate<st_bf_4x4, 2>(); // 16384 bytes

    st_bf_4x1 (&a1_trans_s) = al.allocate<st_bf_4x1>();   // 2048 bytes
    st_bf_4x4 (&a2_s)       = al.allocate<st_bf_4x4 >();   // 8192 bytes
    sv_bf_4   (&a0_total)   = al.allocate<sv_bf_4>();
    sv_bf_1   (&ks_smem_a1) = al.allocate<sv_bf_1>();  // 1024 bytes

    // initial load
    __shared__ kittens::barrier qkv_bar[2], compute_done[2];
    if (warpid == 0) {
        init_barrier(qkv_bar[0], 0, 1);
        init_barrier(qkv_bar[1], 0, 1);
        init_barrier(compute_done[0], 0, (NUM_WORKERS-4));
        init_barrier(compute_done[1], 0, (NUM_WORKERS-4));
        tma::expect_bytes(qkv_bar[0],
            size_bytes<typeof(q_s[0])> +
            size_bytes<typeof(k_s[0])> +
            size_bytes<typeof(v_s[0])>*3
        );
        int tile_idx = blockIdx.x * n_blocks;
        tma::load_async(q_s[tic],   tma_q, qkv_bar[0], tile_idx);
        tma::load_async(k_s[tic],   tma_k, qkv_bar[0], tile_idx);
        tma::load_async(v_s[tic],   tma_v, qkv_bar[0], tile_idx); // faster to have TMA fill a copies than the warps do it.
        tma::load_async(v_s_2[tic], tma_v, qkv_bar[0], tile_idx);
        tma::load_async(v_s_3[tic], tma_v, qkv_bar[0], tile_idx);
    }
    __syncthreads();

    if(warpgroupid == (NUM_WORKERS/4)-1) { // producer
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(24));

        __syncthreads();

        if(warpid == NUM_WORKERS-4) {
            for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) {
                int next_tile_idx = (blockIdx.x * n_blocks) + block + 1;
                tma::expect_bytes(qkv_bar[toc],
                    size_bytes<typeof(q_s[0])> +
                    size_bytes<typeof(k_s[0])> +
                    size_bytes<typeof(v_s[0])>*3
                );
                tma::load_async(q_s[toc],   tma_q, qkv_bar[toc], next_tile_idx);
                tma::load_async(k_s[toc],   tma_k, qkv_bar[toc], next_tile_idx);
                tma::load_async(v_s[toc],   tma_v, qkv_bar[toc], next_tile_idx);
                tma::load_async(v_s_2[toc], tma_v, qkv_bar[toc], next_tile_idx);
                tma::load_async(v_s_3[toc], tma_v, qkv_bar[toc], next_tile_idx);
                wait(compute_done[tic], (block/2)%2); // wait for consumers to all be done
            }
        }
    }
    else if(warpgroupid == 0) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(256));

        __syncthreads();
            
        rt_fl_1x1<> a1_trans; // transposed chunk of a1.
        rt_fl_1x4<> a2[4];    // a2 gets propagated through here.

        // k_state (a1 terms)
        warpgroup::zero(ks_smem_a1);
        tile_k_reg_fl k_cumsum_a1_reg;

        // k_state (a2 terms)
        tile_k_reg_fl k_cumsum_a2_reg [4];
        col_vec<rt_bf<1,1>>  ks_a2[4];
        for(int i = 0; i < 4; i++) { zero(ks_a2[i]); } // everyone zeroes ks_a2.
        col_vec<st_bf<4,1>>  (&ks_a2_s) = al.allocate<col_vec<st_bf<4,1>>>();
        warpgroup::zero(ks_a2_s);
        st_bf_4x4 (&k_smem_a2) = al.allocate<st_bf_4x4>();
        warpgroup::zero(k_smem_a2);

        // k_state (a0 terms)
        sv_fl_4 (&a0_cumsum) = al.allocate<sv_fl_4>(); 
        sv_fl_4 &a0_fixed_vec  = al.allocate<sv_fl_4>(); 
        warpgroup::zero(a0_cumsum);
        warpgroup::zero(a0_fixed_vec);
        if ( warpid == 0 && laneid < 32) { 
            a0_cumsum[laneid] = (laneid+1.0f);
            a0_cumsum[32+laneid] = 32.0+laneid+1.0; 
            a0_fixed_vec[laneid] =  64.0;
            a0_fixed_vec[laneid+32] = 64.0;
        }
        asm volatile("bar.sync 1, 128;\n");

        if(warpid == 0) { zero(a0_total); }
        warpgroup::zero(a1_trans_s);
        zero(a1_trans); // everyone zeroes a2.
        #pragma unroll
        for(int i = 0; i < 4; i++) { zero(a2[i]); } // everyone zeroes a2.

        for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) {
            // each iteration, we handle 64 tokens x 16 feature dim (k, q).
            rt_bf_1x4<> local_attn_bf;               // 4 registers each -- 16
            rt_fl_1x4<> local_attn, temp_attn_accum; // 32 registers each -- 64
            rt_fl_1x4<> o;                           // 32 registers each -- 64

            // arrive memory
            wait(qkv_bar[tic], (block/2)%2);
            asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?

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
            if ( include_a2 > 0 ) {
                mul(temp_attn_accum, temp_attn_accum, temp_attn_accum); // square it; this is divided by D now 
                if (add_scale > 0) { 
                    mul(temp_attn_accum, temp_attn_accum, 0.5f);        // divide a2 term by 2
                } // divide by 2
                add(temp_attn_accum, temp_attn_accum, local_attn);      // add back in 1x for the linear term
            }
            // END comment-out for removing T2 (debug)
            if ( include_a0 > 0 ) {
                add(temp_attn_accum, temp_attn_accum, 1.f);             // cumulative sum for a0
            }
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
            
            if (include_a1 > 0 ) {
                if (add_scale > 0 ) { mul(a1_trans_s, a1_trans_s, __float2bfloat16(0.5)); } // div by sqrt(sqrt(D_QK))
                warpgroup::mma_ABt(o, q_s[tic], a1_trans_s);        // incorporate a1 onto o
            }
            warpgroup::mma_commit_group();                      // dew it
            
            warpgroup::mma_AtB(a1_trans, v_s_2[tic], k_s[tic]); // now 4 1x4 registers that need to be summed.
            warpgroup::mma_commit_group(); // dew it
            warpgroup::mma_async_wait();   // tmp
            warpgroup::store(a1_trans_s, a1_trans);

            // loads
            rt_bf_1x1<> q_src; // the source 16x16 tiles -- we'll draw on these for future mul_slice's.
            warpgroup::load(q_src, q_s[tic]);
            if (add_scale > 0) { 
                mul(q_src, q_src, __float2bfloat16(0.70710678118)); // divide by 2 for A2 here; the mul_slices 
                mul(q_src, q_src, __float2bfloat16(0.25));          // divide by D for A2 here.
            } 
            rt_bf_4x1<> k_src_tmp;
            rt_fl_4x1<> k_src_tmp_fl;
            rt_bf_1x4<> k_src;
            load(k_src_tmp, k_s[tic]);
            copy(k_src_tmp_fl, k_src_tmp);
            transpose_sep(k_src, k_src_tmp); // transpose K into Kt

            // denominator for a1
            rt_fl_1x1<> q_reg_a1_tile;
            rt_fl_1x1<> cumsum_k_a1_reg_tile;
            rt_fl_1x1<>::col_vec linear_norm_vec; 
            zero(q_reg_a1_tile);
            zero(linear_norm_vec); 
            if (output_state > 0) {
                cumulative_add(ks_smem_a1, k_s[tic]); // TODO: remove
            }
            if ( add_norm > 0) { 
                cumulative_add(k_cumsum_a1_reg, k_src_tmp_fl, block);
                asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?
                cumsum_k_a1_reg_tile.tiles[0][0] = k_cumsum_a1_reg.tiles[warpid][0];
                warpgroup::load(q_reg_a1_tile, q_s[tic]);  
                if (add_scale > 0) { 
                    mul(q_reg_a1_tile, q_reg_a1_tile, 0.25f); 
                } // for q and k scale sqrt(sqrt(D))**2
                mul(q_reg_a1_tile, q_reg_a1_tile, cumsum_k_a1_reg_tile);
                if (add_scale > 0) { 
                    mul(cumsum_k_a1_reg_tile, cumsum_k_a1_reg_tile,0.5f); 
                } 
                asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?
                row_sum(linear_norm_vec, q_reg_a1_tile, linear_norm_vec);
            }
            
            // about 75% of execution time is in this loop
            rt_fl_1x1<>::col_vec linear_norm_vec_a2; // N/4 elements over 4 warps
            rt_bf_4x1<> k_t_den;
            rt_fl_4x1<> k_t_den_fl;
            rt_fl_1x4<> q_fl;
            rt_fl_1x4<> cumsum_k_a2_reg_tile;
            zero(linear_norm_vec_a2);
            #pragma unroll
            for(int t = 0; t < 4; t++) {
                rt_bf_1x4<> q, k;
                mul_slice_row(q, q_src, t*4);        // Each warp handles 16 tokens of q, all features
                mul_slice_col(k, k_src, t*4+warpid); // Each warp handles all 64 tokens of k, but different features

                // take previous one and move up to smem for wgmma.
                warpgroup::store(a2_s, a2[t]); 
                asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?

                warpgroup::mma_fence(o);       // av matmul fence
                warpgroup::mma_fence(a2[t]);   // av matmul fence

                if ( include_a2 > 0 ) { 
                    warpgroup::mma_AB(o, q, a2_s); // incorporate a1 onto o
                }
                warpgroup::mma_commit_group(); // dew it
                
                // Note: we originally have k_src_tmp and transpose it (line 283 above)
                // this is becuase AtB function is only usable if A is in SMEM. 
                // but we'd like to keep k in register, so we just transpose it upfront
                warpgroup::mma_AB(a2[t], k, v_s[tic]); // incorporate KtV onto a2
                warpgroup::mma_commit_group(); // dew it

                // write k_state a2 to smem  
                if (output_state > 0) {
                    row_sum(ks_a2[t], k, ks_a2[t]); // cumulative add
                    asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?
                }

                // *************** norm ***************
                zero(k_t_den);
                zero(k_t_den_fl);
                zero(q_fl);
                zero(cumsum_k_a2_reg_tile);
                if ( add_norm > 0 ) {
                    // along the local 64 tokens, sum one feature dim.
                    transpose_sep(k_t_den, k);
                    copy(k_t_den_fl, k_t_den);
                    copy(q_fl, q);
                    cumulative_add(k_cumsum_a2_reg[t], k_t_den_fl, block);
                    asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?

                    // store the 4 cumsum strips for 4 warps in 4x4 smem
                    auto k_smem_a2_subtile = subtile_inplace<4,1>(k_smem_a2, 0, warpid);   
                    store(k_smem_a2_subtile, k_cumsum_a2_reg[t]);
                    asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?

                    // compute norm in registers
                    warpgroup::load(cumsum_k_a2_reg_tile, k_smem_a2);      // load 
                    mul(q_fl, q_fl, cumsum_k_a2_reg_tile);                       // elementwise mult
                    asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?
                    row_sum(linear_norm_vec_a2, q_fl, linear_norm_vec_a2);    // reduction along dimension
                }
                warpgroup::mma_async_wait(); // ding dong! o matmuls have now arrived, too.
            }

            // now we do the sum of the previous a0 onto o
            #pragma unroll
            if (include_a0 > 0) { 
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
            }

            rt_fl_1x1<>::col_vec linear_norm_vec_a0; 
            rt_fl_1x1<>::col_vec linear_norm_vec_total;
            zero(linear_norm_vec_total);
            if ( add_norm > 0 ) { 
                // norm d1
                if ( include_a1 > 0 ) { 
                    add(linear_norm_vec_total, linear_norm_vec_total, linear_norm_vec);
                }

                if ( include_a0 > 0 ) {
                    // norm d0
                    warpgroup::load(linear_norm_vec_a0, a0_cumsum); 
                    add(linear_norm_vec_total, linear_norm_vec_total, linear_norm_vec_a0);
                }

                // norm d2
                if ( include_a2 > 0 ) {
                    add(linear_norm_vec_total, linear_norm_vec_total, linear_norm_vec_a2);
                }

                // eps
                add(linear_norm_vec_total, linear_norm_vec_total, 1e-6);
                asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?

                // divide
                if ( include_a0 > 0 | include_a1 > 0 | include_a2 > 0 ) { 
                    div_row(o, o, linear_norm_vec_total); 
                }
                asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?
            }

            warpgroup::store(o_s[tic], o);
            // do the cumulative sum last, after everything is stored
            accumulate_a0(a0_total, v_s_3[tic]); // cumulative sum of V onto O in shared memory
            if (block > 0 && warpid == 0) {  add(a0_cumsum, a0_cumsum, a0_fixed_vec);  } // denominator

            // save the chunks of output
            if (warpid == 0) { // go get the next K from HBM
                tma::store_async(tma_o, o_s[tic], (blockIdx.x * n_blocks) + block); 
                tma::store_commit_group(); // dew it
            }
            tma::store_async_wait();

            if(laneid == 0) arrive(compute_done[tic]);
        }


        if (output_state > 0) {
            // save the KV state (A2)
            for (int rt = 0; rt < 4; rt++) {
                // reinterpret_cast doesn’t change the bits or memory layout of the variable a2_s. 
                // it tells the compiler to treat a2_s as a reference to the type we've specified.
                auto &kv_smem_2 = reinterpret_cast<st_bf_4x4&>(a2_s); // this layout is better for global HBM stores so we cast.
                if (add_scale > 0) { 
                    mul(a2[rt], a2[rt], 0.70710678118); 
                    mul(a2[rt], a2[rt], 0.25); 
                }
                warpgroup::store(kv_smem_2, a2[rt]); 
                asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?
                if (warpid == 0) {
                    int tile_idx = (blockIdx.x * 4) + rt; 
                    tma::store_async(tma_kv_a2, kv_smem_2, tile_idx); 
                    tma::store_commit_group(); 
                }
                tma::store_async_wait();
            }

            // save the KV state A1
            auto &kv_smem_1 = reinterpret_cast<st_bf_4x1&>(a1_trans_s);
            if (add_scale > 0 ) { mul(a1_trans, a1_trans, 0.5); } // divides by math.sqrt(math.sqrt(D_QK))
            warpgroup::store(kv_smem_1, a1_trans);                // from individual warps to shared address
            asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?
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

            // save the K state A1
            asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?
            if ( add_scale > 0 && warpid == 0 ) { 
                mul(ks_smem_a1, ks_smem_a1, __float2bfloat16(0.5f)); 
            }
            asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?
            if (warpid == 0) { 
                tma::store_async(tma_k_a1, ks_smem_a1, blockIdx.x); 
                tma::store_commit_group(); 
            }
            tma::store_async_wait();

            // save the K state A2
            for (int rt = 0; rt < 4; rt++) {
                auto (&k_smem_2) = reinterpret_cast<col_vec<st_bf_4x1>&>(ks_a2_s);
                if (add_scale > 0) { 
                    mul(ks_a2[rt], ks_a2[rt], __float2bfloat16(0.70710678118)); 
                    mul(ks_a2[rt], ks_a2[rt], __float2bfloat16(0.25)); 
                }
                asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?
                warpgroup::store(k_smem_2, ks_a2[rt]);
                asm volatile("bar.sync 1, 128;\n"); // everybody on the same page?
                if (warpid == 0) {
                    int tile_idx = (blockIdx.x * 4) + rt; 
                    tma::store_async(tma_k_a2, k_smem_2, tile_idx); 
                    tma::store_commit_group(); 
                }
                tma::store_async_wait();
            }
        }
    }
    else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(216));

        __syncthreads();

        for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) {
            // empty loop for the time being
            wait(qkv_bar[tic], (block/2)%2);
            if(laneid == 0) arrive(compute_done[tic]);
        }
    }
    __syncthreads();
}


#ifdef TORCH_COMPILE
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>
void based_linear_prefill(
    int include_0, int include_1, int include_2,
    int add_scale, int add_norm, int output_state,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, 
    torch::Tensor kv_a2, torch::Tensor kv_a1, torch::Tensor kv_a0, 
    torch::Tensor k_a2, torch::Tensor k_a1
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    CHECK_INPUT(kv_a2);
    CHECK_INPUT(kv_a1);
    CHECK_INPUT(kv_a0);
    CHECK_INPUT(k_a2);
    CHECK_INPUT(k_a1);

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
    c10::BFloat16 *k_a2_ptr     = k_a2.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_a1_ptr     = k_a1.data_ptr<c10::BFloat16>();

    const bf16* d_q  = reinterpret_cast<const bf16*>(q_ptr);
    const bf16* d_k  = reinterpret_cast<const bf16*>(k_ptr);
    const bf16* d_v  = reinterpret_cast<const bf16*>(v_ptr);
          bf16* d_o  = reinterpret_cast<bf16*>(o_ptr);
          bf16* d_kv_a2 = reinterpret_cast<bf16*>(kv_a2_ptr);
          bf16* d_kv_a1 = reinterpret_cast<bf16*>(kv_a1_ptr); 
          bf16* d_kv_a0 = reinterpret_cast<bf16*>(kv_a0_ptr);
          bf16* d_k_a2 = reinterpret_cast<bf16*>(k_a2_ptr); 
          bf16* d_k_a1 = reinterpret_cast<bf16*>(k_a1_ptr);

    CUtensorMap* tma_q_d   = tma::allocate_and_create_tensor_map<st_bf_4x1>(d_q, (batch*heads*n)/(4 * kittens::TILE_DIM)); 
    CUtensorMap* tma_k_d   = tma::allocate_and_create_tensor_map<st_bf_4x1>(d_k, (batch*heads*n)/(4 * kittens::TILE_DIM)); 
    CUtensorMap* tma_v_d   = tma::allocate_and_create_tensor_map<st_bf_4x4>(d_v, (batch*heads*n)/(4 * kittens::TILE_DIM));
    CUtensorMap* tma_o_d   = tma::allocate_and_create_tensor_map<st_bf_4x4>(d_o, (batch*heads*n)/(4 * kittens::TILE_DIM)); 

    // kv state maps
    CUtensorMap* tma_kv_a2_d = tma::allocate_and_create_tensor_map<st_bf_4x4> (d_kv_a2, batch*heads*(D_QK*D_QK));
    CUtensorMap* tma_kv_a1_d = tma::allocate_and_create_tensor_map<st_bf_4x1> (d_kv_a1, batch*heads*(D_QK));
    CUtensorMap* tma_kv_a0_d = tma::allocate_and_create_tensor_map<sv_bf_4> (d_kv_a0, batch*heads);

    // k state maps  
    CUtensorMap* tma_k_a2_d   = tma::allocate_and_create_tensor_map<col_vec<st_bf_4x1>>(d_k_a2, 4*batch*heads); 
    CUtensorMap* tma_k_a1_d   = tma::allocate_and_create_tensor_map<sv_bf_1>(d_k_a1, batch*heads); 

    unsigned long mem_size = 108000; 
    
    using T = kittens::bf16;
    using H = kittens::bf16;

    cudaFuncSetAttribute(
        based_linear_attention,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    ); 

    based_linear_attention<<<batch*heads, threads, mem_size>>>(
        include_0, include_1, include_2,
        n, add_scale, add_norm, output_state, 
        tma_q_d, tma_k_d, tma_v_d, tma_o_d, 
        tma_kv_a2_d, tma_kv_a1_d, tma_kv_a0_d, 
        tma_k_a2_d, tma_k_a1_d
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
}
#else
#include "harness.impl"
#endif