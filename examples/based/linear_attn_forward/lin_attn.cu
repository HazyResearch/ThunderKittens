# include "src/kittens.cuh"

#define NUM_WORKERS (16) // hardcoded, don't change
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define ACTIVE_TILES (8) // hardcoded, don't change
#define D_QK (16) // hardcoded, don't change
#define D_VO (64) // hardcoded but can be changed with some effort

using namespace kittens;

using layout = kittens::ducks::st_layout::xor_swizzle;

// cumulative sum of v onto o
template<kittens::ducks::st::all ST, int N_TILES>
__device__ void accumulate_v0(ST (&o)[N_TILES], sv_bf<ST::width> &running_sum, const ST (&v)[N_TILES]) {
    float acc;
    // simple version first
    if(threadIdx.x < ST::cols) {
        int col = threadIdx.x;
        acc = __bfloat162float(running_sum[col]);
        #pragma unroll
        for(int t = 0; t < N_TILES; t++) {
            #pragma unroll
            for(int i = 0; i < ST::rows; i++) {
                acc += __bfloat162float(v[t][int2{i, col}]); // get row, col
                o[t][int2{i, col}] += __float2bfloat16(acc);
            }
        }
        running_sum[col] = __float2bfloat16(acc);
    }
}

// cumulative sum of an array of tiles
template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void cumsum_inplace(ST (&x)[N_TILES], int total_block_idx) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;
    // then propagate accumulation through
    for(int i = 1; i < N_TILES; i++) {
        #pragma unroll
        for(int j = threadIdx.x; j < ST::num_elements; j+=STRIDE) {
            x[(total_block_idx+i)%N_TILES].data[j] += x[(total_block_idx+i-1)%N_TILES].data[j]; // accumulate with previous one
        }
    }
}

// cumulative sum of an array of tiles -- in fp32 to preserve maximal accuracy
template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void tile_reduce(ST &dst, const ST (&src)[N_TILES]) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;
    constexpr int RESPONSIBLE_ELEMENTS = ST::num_elements / STRIDE; // we know in advance this divides evenly.
    float acc[RESPONSIBLE_ELEMENTS];
    #pragma unroll
    for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
        int idx = threadIdx.x + j*STRIDE;
        acc[j] = __bfloat162float(dst.data[idx]); // start
    }
    // then propagate accumulation through
    for(int i = 0; i < N_TILES; i++) {
        #pragma unroll
        for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
            int idx = threadIdx.x + j*STRIDE;
            acc[j] += __bfloat162float(src[i].data[idx]); // accumulate
        }
    }
    #pragma unroll
    for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
        int idx = threadIdx.x + j*STRIDE;
        dst.data[idx] = acc[j]; // set
    }
}

// in pytorch, this computes, for a 16x16 tensor reg:
// reg *= reg[:,warpid].unsqueeze(0)
__device__ static void mul_slice(rt_bf_1x1<> &reg) {

    const int target_col = kittens::warpid(); // 0...15
    const int lane       = kittens::laneid(); // 0...31
    
    // each thread is responsible for two rows
    #pragma unroll
    for(int row_offset = 0; row_offset < 2; row_offset++) {
        const int dst_row = row_offset*8 + lane / 4;
        const int src_thread = (lane / 4)*4 + (target_col%8)/2;
        const int col_offset = target_col >= 8;
        bf16_2 src_val = reg.tiles[0][0].data[2*col_offset + row_offset];
        bf16 val = __shfl_sync(kittens::MASK_ALL, (target_col%2 == 0) ? src_val.x : src_val.y, src_thread); // correct value obtained and passed around

        // It turns out this is the most convenient time to incorporate the fact we want T2/2 in the end.
        // So, we multiply by 1/sqrt(2) -- once on Q, and once on K. Together, this gives us what we want.
        val *= __float2bfloat16(0.70710678118);

        reg.tiles[0][0].data[row_offset] *= bf16_2{val, val};
        reg.tiles[0][0].data[row_offset+2] *= bf16_2{val, val};
    }
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void based_linear_attention(int n, const bf16* __q, const bf16* __k, const bf16* __v, bf16* __o) {

    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    const bf16 *q_g   = reinterpret_cast<const bf16*>(__q)+blockIdx.x*(n*D_QK);
    const bf16 *k_g   = reinterpret_cast<const bf16*>(__k)+blockIdx.x*(n*D_QK);
    const bf16 *v_g   = reinterpret_cast<const bf16*>(__v)+blockIdx.x*(n*D_VO);
          bf16 *o_g   = reinterpret_cast<bf16*>      (__o)+blockIdx.x*(n*D_VO);

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf_1x1<layout> (&q_s)[ACTIVE_TILES]  = al.allocate<st_bf_1x1<layout>, ACTIVE_TILES>(); // 4096 bytes
    st_bf_1x1<layout> (&k_s)[ACTIVE_TILES]  = al.allocate<st_bf_1x1<layout>, ACTIVE_TILES>(); // 4096 bytes
    st_bf_1x4<layout> (&v_s)[ACTIVE_TILES]  = al.allocate<st_bf_1x4<layout>, ACTIVE_TILES>(); // 16384 bytes
    st_bf_1x4<layout> (&o_s)[ACTIVE_TILES]  = al.allocate<st_bf_1x4<layout>, ACTIVE_TILES>(); // 16384 bytes

    // This one is a bit weird. Essentially, we need to have ACTIVE_TILES, but also a separate total.
    // HOWEVER, we also need to have each tile use the previous one. The solution is to turn this into a ring buffer.
    // Every time we advance the block index, we count in the opposite direction for the index of the total block.
    st_bf_1x4<layout> (&a1_s)[ACTIVE_TILES+1] = al.allocate<st_bf_1x4<layout>, ACTIVE_TILES+1>(); // 18432 bytes
    // a2 is just used for the block sum reduction on the tiles
    st_bf_1x4<layout> (&a2_o_accumulate)[NUM_WORKERS]    = al.allocate<st_bf_1x4<layout>, NUM_WORKERS>(); // 32768 bytes
    int total_block_idx = 0;

    rt_fl_1x4 a2; // a2 gets propagated through here.

    sv_bf_4 a0_total = al.allocate<sv_bf_4>();

    if(warpid == 0) {
        zero(a0_total);
    }
    if(warpid < ACTIVE_TILES+1) {
        zero(a1_s[warpid]);
    }
    zero(a2); // everyone zeroes a2.

    int n_blocks = n / (ACTIVE_TILES * kittens::TILE_DIM);

    for(int block = 0; block < n_blocks; block++) {
        rt_bf_1x1<> q, k, local_attn_bf; // 4 registers each -- 12
        rt_fl_1x1<> local_attn, temp_attn_accum; // 8 registers each -- 8
        rt_bf_1x4<> v; // 16 registers each -- 16
        rt_fl_1x4<> o, accum; // 32 registers each -- 64
        // total: 100 registers in use here.

        // load new q, k, v into shared memory and zero o
        int cur_idx;
        if(warpid < ACTIVE_TILES) {
            cur_idx = block*ACTIVE_TILES + warpid;
            load(q_s[warpid], q_g + cur_idx * q_s[warpid].num_elements, D_QK);
            load(k_s[warpid], k_g + cur_idx * k_s[warpid].num_elements, D_QK);
        }
        else {
            cur_idx = block*ACTIVE_TILES + warpid - 8;
            load(v_s[warpid-8], v_g + cur_idx * v_s[warpid-8].num_elements, D_VO);
        }
        __syncthreads();

        // we start by doing the very local computations. Then, we'll follow up later with the rest.
        if(warpid < ACTIVE_TILES) {
            load(q, q_s[warpid]);
            load(k, k_s[warpid]);

            zero(local_attn);
            dot(local_attn, q, k, local_attn);

            // our goal is to store local_attn + (local_attn^2 / 2) in local_attn_bf
            copy(temp_attn_accum, local_attn);
            // BEGIN comment-out for removing T2 (debug)
            mul(temp_attn_accum, temp_attn_accum, temp_attn_accum); // square it
            mul(temp_attn_accum, temp_attn_accum, 0.5f); // divide by 2
            add(temp_attn_accum, temp_attn_accum, local_attn); // add back in 1x for the linear term
            // END comment-out for removing T2 (debug)
            copy(local_attn_bf, temp_attn_accum); // now stored.
            make_causal(local_attn_bf, local_attn_bf);

            load(v, v_s[warpid]);
            auto &v_col = swap_layout_inplace(v); // prepare for MMA

            zero(o);
            mma(o, local_attn_bf, v_col, o);

            zero(accum); // plan to save 24 registers: break this up and store between.
            auto &kt = transpose_inplace(k); // k is now transposed! k has been invalidated; there is only kt.
            mma(accum, kt, v_col, accum);
            store(a1_s[(total_block_idx+warpid+1)%(ACTIVE_TILES+1)], accum);
        }
        __syncthreads();
        cumsum_inplace<NUM_WORKERS>(a1_s, total_block_idx);
        __syncthreads();
        if(warpid < ACTIVE_TILES) {
            rt_bf_1x4 a1;
            load(q, q_s[warpid]); // load q again
            load(a1, a1_s[(total_block_idx+warpid)%(ACTIVE_TILES+1)]);
            auto &a1_col = swap_layout_inplace(a1); // prepare for MMA
            mma(o, q, a1_col, o); // mma onto O in registers
            store(o_s[warpid], o); // store current o to shared memory
        }
        total_block_idx = (total_block_idx+ACTIVE_TILES)%(ACTIVE_TILES+1); // count backwards on the ring
        __syncthreads();

        // about 75% of execution time is in this loop
        for(int t = 0; t < ACTIVE_TILES; t++) {
            load(q, q_s[t]); // everyone loads the same t
            mul_slice(q); // but they multiply by a different column

            // now we use a2 to accumulate a local o update
            rt_bf_1x4<> a2_bf;
            copy(a2_bf, a2);
            auto &a2_bf_col = swap_layout_inplace(a2_bf);
            zero(o);
            mma(o, q, a2_bf_col, o);

            // next we have everyone load k and do the same
            load(k, k_s[t]);
            mul_slice(k);
            auto &kt = transpose_inplace(k); // k is now transposed! k has been invalidated; there is only kt.

            load(v, v_s[t]);
            auto &v_col = swap_layout_inplace(v); // prepare for MMA
            mma(a2, kt, v_col, a2); // accumulate onto a2

            // good chance I need to make this faster
            store(a2_o_accumulate[warpid], o);
            __syncthreads();
            tile_reduce<NUM_WORKERS>(o_s[t], a2_o_accumulate);
            // the below __syncthreads() is needed for formal correcntess, as everyone must be done before we can proceed
            // however, it will basically always be correct without it, and 3% faster.
            __syncthreads();
        }

        // do the cumulative sum last, after everything is stored
        accumulate_v0(o_s, a0_total, v_s); // cumulative sum on V
        __syncthreads();

        if(warpid < ACTIVE_TILES) {
            store(o_g + cur_idx * o_s[warpid].num_elements, o_s[warpid], D_VO);
        }
    }
}

#include "harness.impl"