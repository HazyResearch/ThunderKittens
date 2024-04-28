# include "src/kittens.cuh"

#define NUM_WORKERS (4) // hardcoded, don't change
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define D_QK (16) // hardcoded, don't change
#define D_VO (64) // hardcoded but can be changed with some effort

using namespace kittens;
// cumulative sum of v onto o
template<kittens::ducks::st::all ST1, kittens::ducks::st::all ST2>
__device__ void accumulate_v0(ST1 &o, sv_bf<ST1::width> &running_sum, const ST2 &v) {
    static_assert(ST1::rows == ST2::rows);
    static_assert(ST1::cols == ST2::cols);
    float acc;
    // simple version first
    if(threadIdx.x < ST1::cols) {
        int col = threadIdx.x;
        acc = __bfloat162float(running_sum[col]);
        #pragma unroll
        for(int i = 0; i < ST1::rows; i++) {
            acc += __bfloat162float(v[int2{i, col}]); // get row, col
            float f = acc + __bfloat162float(o[int2{i, col}]);
            o[int2{i, col}] = __float2bfloat16(f);
        }
        running_sum[col] = __float2bfloat16(acc);
    }
}
// in pytorch, this computes, for a 16x64 tensor dst and 16x16 tensor src:
// dst = torch.cat([src * src[:,starting_col+i:].unsqueeze(0) for i in range(4)], dim=-1)
__device__ static void mul_slice(rt_bf_1x4<> &dst, const rt_bf_1x1<> &src, const int starting_col) {

    const int lane = kittens::laneid(); // 0...31    
    // each thread is responsible for two rows
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        copy(reinterpret_cast<rt_bf_1x1<>&>(dst.tiles[0][i]), src);
        const int target_col = starting_col + i;
        #pragma unroll
        for(int row_offset = 0; row_offset < 2; row_offset++) {
            const int dst_row = row_offset*8 + lane / 4;
            const int src_thread = (lane / 4)*4 + (target_col%8)/2;
            const int col_offset = target_col >= 8;
            bf16_2 src_val = dst.tiles[0][i].data[2*col_offset + row_offset];
            bf16 val = __shfl_sync(kittens::MASK_ALL, (target_col%2 == 0) ? src_val.x : src_val.y, src_thread); // correct value obtained and passed around

            dst.tiles[0][i].data[row_offset] *= bf16_2{val, val};
            dst.tiles[0][i].data[row_offset+2] *= bf16_2{val, val};
        }
    }
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void based_linear_attention(int n, CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o) {//, CUtensorMap* __kv_state) {

    int warpid = kittens::warpid(); // who am i? when am i?
    int tic = 0, toc = 1;

    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_bf_4x1<wgmma_swizzle_l>    (&q_s)[2] = al.allocate<st_bf_4x1<wgmma_swizzle_l>,    2>(); // 4096 bytes
    st_bf_4x1<wgmma_interleave_l> (&k_s)[2] = al.allocate<st_bf_4x1<wgmma_interleave_l>, 2>(); // 4096 bytes
    st_bf_4x4<wgmma_interleave_l> (&v_s)[2] = al.allocate<st_bf_4x4<wgmma_interleave_l>, 2>(); // 16384 bytes
    st_bf_4x4<swizzle_l>          (&o_s)[2] = al.allocate<st_bf_4x4<swizzle_l>,          2>(); // 16384 bytes

    rt_fl_1x1<> a1_trans; // transposed chunk of a1.
    rt_fl_1x4<> a2[4]; // a2 gets propagated through here.
    st_bf_4x1<wgmma_swizzle_l>    (&a1_trans_s) = al.allocate<st_bf_4x1<wgmma_swizzle_l>   >(); // 2048 bytes
    st_bf_4x4<wgmma_interleave_l> (&a2_s)       = al.allocate<st_bf_4x4<wgmma_interleave_l>>(); // 8192 bytes
    st_bf_4x4<wgmma_interleave_l> (&k_mul_s)    = al.allocate<st_bf_4x4<wgmma_interleave_l>>(); // 8192 bytes

    st_bf_4x4<wgmma_interleave_l> (&test_chunk_0) = al.allocate<st_bf_4x4<wgmma_interleave_l>>(); // 8192 bytes

    sv_bf_4 &a0_total = al.allocate<sv_bf_4>();
    int a0_offset;
    __shared__ float a0_ring[320];
    a0_ring[threadIdx.x] = 0;

    if(warpid == 0) {
        zero(a0_total);
        one(test_chunk_0);
    }
    warpgroup::zero(a1_trans_s);
    zero(a1_trans); // everyone zeroes a2.
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        zero(a2[i]); // everyone zeroes a2.
    }

    int n_blocks = n / (q_s[0].rows);

    // initial load
    __shared__ tma::barrier bar;
    if (warpid == 0) tma::init_barrier(bar);
    __syncthreads();
    if (warpid == 0) {
        tma::set_bytes(bar,
            size_bytes<typeof(q_s[0])> +
            size_bytes<typeof(k_s[0])> +
            size_bytes<typeof(v_s[0])>
        );
        int tile_idx = blockIdx.x * n_blocks;
        tma::load_async(q_s[tic], tma_q, bar, tile_idx);
        tma::load_async(k_s[tic], tma_k, bar, tile_idx);
        tma::load_async(v_s[tic], tma_v, bar, tile_idx);
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
                size_bytes<typeof(v_s[0])>
            );
            int next_tile_idx = (blockIdx.x * n_blocks) + block + 1;
            tma::load_async(q_s[toc], tma_q, bar, next_tile_idx);
            tma::load_async(k_s[toc], tma_k, bar, next_tile_idx);
            tma::load_async(v_s[toc], tma_v, bar, next_tile_idx);
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
        mul(temp_attn_accum, temp_attn_accum, 0.5f); // divide by 2
        add(temp_attn_accum, temp_attn_accum, local_attn); // add back in 1x for the linear term
        // END comment-out for removing T2 (debug)
        copy(local_attn_bf, temp_attn_accum); // now stored.
        // now make causal
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            auto &attn_subtile = reinterpret_cast<rt_bf_1x1<>&>(local_attn_bf.tiles[0][j]);
            if (j>warpid) zero(attn_subtile);
            else if (j==warpid) make_causal(attn_subtile, attn_subtile);
        }

        warpgroup::mma_fence(o); // av matmul fence
        warpgroup::mm_AB(o, local_attn_bf, v_s[tic]); // reset o here, and do local chunk.
        warpgroup::mma_commit_group(); // dew it
        warpgroup::mma_async_wait(); // ding dong! o matmuls have arrived
        warpgroup::mma_fence(o); // av matmul fence
        warpgroup::mma_ABt(o, q_s[tic], a1_trans_s); // incorporate a1 onto o
        warpgroup::mma_commit_group(); // dew it
        warpgroup::mma_async_wait(); // ding dong! o matmuls have arrived

        warpgroup::copy(test_chunk_0, v_s[tic]); // this extra copy should not be necessary, but stuff breaks without it for unclear reasons.
        warpgroup::mma_fence(a1_trans); // a1 accumulation matmul fence
        warpgroup::mma_AtB(a1_trans, test_chunk_0, k_s[tic]); // we now have 4 1x4 registers that need to eventually be summed.
        warpgroup::mma_commit_group(); // dew it
        warpgroup::mma_async_wait(); // tmp
        warpgroup::store(a1_trans_s, a1_trans);

        rt_bf_1x1<> q_src, k_src; // the source 16x16 tiles -- we'll draw on these for future mul_slice's.
        warpgroup::load(q_src, q_s[tic]);
        mul(q_src, q_src, __float2bfloat16(0.70710678118)); // divide by 2 for A2 here.
        warpgroup::load(k_src, k_s[tic]);
        
        // about 75% of execution time is in this loop
        #pragma unroll
        for(int t = 0; t < 4; t++) {
            rt_bf_1x4<> q, k;
            mul_slice(q, q_src, t*4);
            mul_slice(k, k_src, t*4);
            warpgroup::store(a2_s, a2[t]); // take previous one and move up to smem for wgmma.
            warpgroup::store(k_mul_s, k);
            __syncthreads();

            warpgroup::mma_fence(o); // av matmul fence
            warpgroup::mma_AB(o, q, a2_s); // incorporate a1 onto o
            warpgroup::mma_commit_group(); // dew it
            warpgroup::mma_fence(a2[t]); // av matmul fence
            warpgroup::mma_AtB(a2[t], k_mul_s, v_s[tic]); // incorporate KtV onto a2
            warpgroup::mma_commit_group(); // dew it
            warpgroup::mma_async_wait(); // ding dong! o matmuls have now arrived, too.
        }

        // // do the cumulative sum last, after everything is stored
        warpgroup::store(o_s[tic], o);
        __syncthreads();
        accumulate_v0(o_s[tic], a0_total, v_s[tic]); // cumulative sum of V onto O in shared memory
        __syncthreads();

        if (block>0) tma::store_async_wait();
        if (warpid == 0) { // go get the next K from HBM
            tma::store_async(tma_o, o_s[tic], (blockIdx.x * n_blocks) + block); 
            tma::store_commit_group(); // dew it
        }
    }
    tma::store_async_wait();
}

#include "harness.impl"

