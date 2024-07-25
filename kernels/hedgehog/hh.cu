// #define TORCH_COMPILE // defined by default for PyTorch bindings - to use cpp harness, comment this out

#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>


#define NUM_WORKERS (12)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define NUM_WARPGROUPS (NUM_WORKERS/kittens::WARPGROUP_WARPS)

using namespace kittens;

// ----- DEBUG -----
#define RED  "\033[91m" 
#define GREEN  "\033[92m" 
#define YELLOW  "\033[93m" 
#define BLUE  "\033[94m" 
#define MAGENTA  "\033[95m" 
#define CYAN  "\033[96m" 
#define WHITE  "\033[97m" 
#define RESET  "\033[0m" 
template<typename... Args> __device__ void gprintf(Args... args) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x % 32 == 0) {
        printf(args...);
    }
}
// ----- DEBUG -----

template<ducks::sv::all SV>
__device__ inline void cumulative_add(SV &dst, const SV (&src)[4]) {
    // this is called along a warpgroup
    static_assert(SV::length <= 64);
    int lane = threadIdx.x % 64;
    if(lane < SV::length) {
        float f = dst[lane];
        // acc equal to the last row of dst
        for (auto i = 0; i < 4; i++) {
            f += src[i][lane];
        }
        dst[lane] = f;
    }
}

template<ducks::rt::all RT>
__device__ inline void softmax_featuremap_inplace(RT &tile) {
    col_vec<RT> max_vec, sum_vec;
    row_max(max_vec, tile);
    sub_row(tile, tile, max_vec); // now in range (-infty, 0) for numerical stability
    exp(tile, tile);
    row_sum(sum_vec, tile);
    div_row(tile, tile, sum_vec);
}

#define ATTN_D 128
#define ATTN_F 128

using q_tile = st_bf<4, 8>;
using k_tile = st_bf<4, 8>;
using v_tile = st_bf<4, 8>;
using o_tile = st_bf<4, 8>;

using kv_state_tile = st_fl<8, 8>;
using k_state_vec = sv_fl_8;

using qk_map_tile = st_bf<8, 4>;

// should be launched with a grid of size (HEADS, BATCH) and blocks of 256 threads.
__global__ __launch_bounds__(NUM_THREADS, 1)
void hedgehog_linear_attention_smd (int n, 
                                    const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, // inputs
                                    CUtensorMap* tma_o, // outputs of O state for each query
                                    CUtensorMap* tma_k_state, CUtensorMap* tma_kv_state, // global outputs of K state and KV state
                                    const CUtensorMap* tma_qmap, const CUtensorMap* tma_kmap,
                                    const float *alphas, const float *betas)  { // alpha is for linear component, beta is for sliding window component. Array, per head.
    const int batch_id = blockIdx.y;
    const int head_id  = blockIdx.x;
    const int batch_head_id = batch_id*gridDim.x + head_id;
    float alpha = alphas[head_id]; // the weighting of linear vs sliding window attention for this head. alpha controls linear.
    float beta  = betas [head_id]; // the weighting of the sliding window attention for this head. beta controls the sliding window.
    int warpid = kittens::warpid(), warpgroupid = warpid/WARPGROUP_WARPS;
    int tic = 0, toc = 1; // for the double buffer of Q, and the replicated barriers in a producer consumer model
    int ring_id = 0; // for the circular buffer of K and V
    int blocks = n / (q_tile::rows);
    
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    q_tile (&q_smem)[2] = al.allocate<q_tile, 2>(); // 32k, (tic/toc)*16k
    k_tile (&k_smem)[3] = al.allocate<k_tile, 3>(); // 48k, (3-ring)*(64x128)
    v_tile (&v_smem)[3] = al.allocate<v_tile, 3>(); // 48k, (3-ring)*(64x128)
    o_tile (&o_smem)    = al.allocate<o_tile>   (); // 16k

    qk_map_tile (&qf_map) = al.allocate<qk_map_tile>(); // 16k, for fusing featuremap computation
    qk_map_tile (&kf_map) = al.allocate<qk_map_tile>(); // 16k, for fusing featuremap computation

    st_bf<4, 8> (&kv_smem)[2] = al.allocate<st_bf<4, 8>, 2>(); // 32k, 64x128 featurized 
    
    row_vec<st_fl<4,4>> (&cumsum_k_smem)[2]            = al.allocate<row_vec<st_fl<4,4>>, 2   >(); // smol (<1024)
    row_vec<st_fl<4,4>> (&cumsum_k_smem_scratch)[2][4] = al.allocate<row_vec<st_fl<4,4>>, 2, 4>(); // fairly smol (2048)
    col_vec<st_fl<4,4>> (&norm_exchange)               = al.allocate<col_vec<st_fl<4,4>>      >(); // smol (<1024)

    st_bf<4, 4> (*k_scratch_smem) = reinterpret_cast<st_bf<4, 4>*>(&kv_smem[0].data[0]);

    __shared__ kittens::barrier qk_smem_arrived[2], v_smem_arrived[2], compute_done[2], inter_stored, inter_retrieved;
    if (warpid == 0) {
        init_barrier(qk_smem_arrived[0], 0, 1);
        init_barrier(qk_smem_arrived[1], 0, 1);
        tma::expect_bytes(qk_smem_arrived[0], 
            size_bytes<q_tile> + 
            size_bytes<k_tile> + 
            // we need qk maps to be loaded on this first iter, too.
            size_bytes<qk_map_tile> +
            size_bytes<qk_map_tile>
        );
        int tile_idx = (batch_head_id * blocks) + 0;
        // first thing we need to do is load the QK map
        tma::load_async(qf_map, tma_qmap, qk_smem_arrived[0], head_id); // need to load the right head
        tma::load_async(kf_map, tma_kmap, qk_smem_arrived[0], head_id);
        // now we also load the first data we need
        tma::load_async(q_smem[tic],       tma_q, qk_smem_arrived[0], tile_idx);
        tma::load_async(k_smem[ring_id+1], tma_k, qk_smem_arrived[0], tile_idx);
    }
    else if(warpid == 1) {
        init_barrier(v_smem_arrived[0], 0, 1);
        init_barrier(v_smem_arrived[1], 0, 1);
        tma::expect_bytes(v_smem_arrived[0], size_bytes<v_tile>);
        int tile_idx = (batch_head_id * blocks) + 0;
        tma::load_async(v_smem[ring_id+1], tma_v, v_smem_arrived[0],  tile_idx); // load v to its own barrier
    }
    else if(warpid < 4) {
        init_barrier(compute_done[warpid-2], 0, 8); // both consumer warpgroups have to hit these
        if(warpid == 2) {
            init_barrier(inter_stored, 0, 4); // just one consumer warpgroup hits each of these
            init_barrier(inter_retrieved, 0, 4); // just one consumer warpgroup hits each of these
        }
    }
    else if(warpgroupid == 1) {
        warpgroup::zero(v_smem[ring_id]);
        warpgroup::zero(cumsum_k_smem[0]);
    }
    else if(warpgroupid == 2) {
        warpgroup::zero(cumsum_k_smem[1]);
    }

    __syncthreads();

    if(warpgroupid == 2) { // producer
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));
   
        if(warpid == 8) {
            for (auto block = 0; block < blocks-1; block++, tic=tic^1, toc=toc^1, ring_id=(ring_id+1)%3) {
                int next_tile_idx = (batch_head_id * blocks) + block + 1;
                tma::expect_bytes(qk_smem_arrived[toc], size_bytes<q_tile> + size_bytes<k_tile>);
                tma::load_async(q_smem[toc],           tma_q, qk_smem_arrived[toc], next_tile_idx);
                tma::load_async(k_smem[(ring_id+2)%3], tma_k, qk_smem_arrived[toc], next_tile_idx);
                wait(compute_done[tic], (block/2)%2);
            }
        }
        else if(warpid == 9) {
            for (auto block = 0; block < blocks-1; block++, tic=tic^1, toc=toc^1, ring_id=(ring_id+1)%3) {
                int next_tile_idx = (batch_head_id * blocks) + block + 1;
                tma::expect_bytes(v_smem_arrived[toc], size_bytes<v_tile>);
                tma::load_async(v_smem[(ring_id+2)%3], tma_v, v_smem_arrived[toc], next_tile_idx);
                wait(compute_done[tic], (block/2)%2);
            }
        }
    }
    else if(warpgroupid == 0) { // sliding window attn consumer
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(216));

        // tell the other warpgroup it's allowed to store into the exchange buffer
        if(laneid() == 0) arrive(inter_retrieved);

        for (int block = 0; block < blocks; block++, tic^=1, toc^=1, ring_id=(ring_id+1)%3) {

            // ******* sliding window attn ******* // 

            rt_fl<1, 8> sliding_o, linear_o;
            rt_fl<1, 4>::col_vec sliding_norm_vec, linear_norm_vec;
            zero(sliding_o);
            zero(sliding_norm_vec);

            rt_fl<1, 4> att_block[2];
            rt_bf<1, 4> att_block_bf[2];
            rt_fl<1, 4>::col_vec max_vec;

            neg_infty(max_vec); // zero registers for the Q chunk
            
            wait(qk_smem_arrived[tic], (block/2)%2);  // ding! memory arrived

            #pragma unroll
            for(int subtile = 0; subtile < 2; subtile++) {
                warpgroup::mma_fence(att_block[subtile]);
                warpgroup::mm_ABt(att_block[subtile], q_smem[tic], k_smem[(ring_id+subtile)%3]);
                warpgroup::mma_commit_group();
            }
            wait(v_smem_arrived[tic], (block/2)%2);  // ding! memory arrived
            warpgroup::mma_async_wait();
            if(block == 0) { // initial block must be zeroed in the first 64x64 chunk
                neg_infty(att_block[0]);
            }
            // make last block causal
            #pragma unroll
            for(int j = 0; j < 4; j++) {
                auto &attn_subtile = reinterpret_cast<rt_fl_1x1<>&>(att_block[1].tiles[0][j]);
                if (j>warpid) neg_infty(attn_subtile);
                else if (j==warpid) make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty());
            }
            // now do the softmax. first we subtract max for numerical stability. then exp.
            #pragma unroll
            for(int subtile = 0; subtile < 2; subtile++) {
                mul(att_block[subtile], att_block[subtile], 0.08838834764 * 1.44269504089); // temperature adjustment, with lg(e) so we can use exp2
                row_max(max_vec, att_block[subtile], max_vec); // accumulate onto the max_vec
            }
            #pragma unroll
            for(int subtile = 0; subtile < 2; subtile++) {
                sub_row(att_block[subtile], att_block[subtile], max_vec);
                exp2(att_block[subtile], att_block[subtile]);
                mul(att_block[subtile], att_block[subtile], beta);
            }
            // now we sum so that we can divide (normalize) later
            #pragma unroll
            for(int subtile = 0; subtile < 2; subtile++) {
                row_sum(sliding_norm_vec, att_block[subtile], sliding_norm_vec); // incorporates beta
                copy(att_block_bf[subtile], att_block[subtile]); // cast to bf16 for next matmul
            }
            warpgroup::mma_fence(sliding_o);
            #pragma unroll
            for(int subtile = 0; subtile < 2; subtile++) {
                warpgroup::mma_AB(sliding_o, att_block_bf[subtile], v_smem[(ring_id+subtile)%3]);
                warpgroup::mma_commit_group();
            }
            warpgroup::mma_async_wait();

            if(laneid() == 0) arrive(compute_done[tic]); // we are done with Q, K, V now

            // exchange
            wait(inter_stored, tic); // this is on tic, so it does wait
            warpgroup::load(linear_norm_vec, norm_exchange);
            warpgroup::load(linear_o, o_smem);
            asm volatile("bar.sync 10, 128;\n");

            add(sliding_norm_vec, sliding_norm_vec, linear_norm_vec);
            add(sliding_o, sliding_o, linear_o);
            div_row(sliding_o, sliding_o, sliding_norm_vec); // this half is now normalized

            // tma::store_async_wait();
            warpgroup::store(o_smem, sliding_o);
            asm volatile("bar.sync 10, 128;\n"); // just this warpgroup needs to hit this

            if(warpid == 0) {
                tma::store_async(tma_o, o_smem, batch_head_id*blocks + block);
                tma::store_commit_group();
            }
            asm volatile("bar.sync 10, 128;\n"); // just this warpgroup needs to hit this
            if(laneid() == 0) arrive(inter_retrieved); // we have successfully retrieved the norm vector, so the next one can be stored
        }
    }
    else { // lin attn consumer
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(256));

        rt_fl<1, 8> local_kv[2]; // this is going to be split across the two warpgroups involved.
        for(int i = 0; i < 2; i++) zero(local_kv[i]);

        // we don't do anything here on the first block
        wait(qk_smem_arrived[tic], 0);  // ding! memory arrived
        wait(v_smem_arrived[tic], 0);  // ding! memory arrived

        // For the first iteration, we just need to send zeros.
        warpgroup::zero(o_smem);
        warpgroup::zero(norm_exchange);
        if(laneid() == 0) arrive(inter_stored);

        // mark all compute as being done
        if(laneid() == 0) arrive(compute_done[tic]);

        // we now need to get to the next block
        tic ^= 1;
        toc ^= 1;
        ring_id = (ring_id+1)%3;

        for (int block = 1; block < blocks; block++, tic^=1, toc^=1, ring_id=(ring_id+1)%3) {
            // gprintf(RED "(warp %d) linear made it to block %d\n" RESET, warpid, block);
            
            wait(qk_smem_arrived[tic], (block/2)%2);  // ding! memory arrived
            wait(v_smem_arrived[tic], (block/2)%2);  // ding! memory arrived

            // ******* linear attn ******** // 

            rt_fl<1, 8> linear_o; // this is partitioned across the two warpgroups.
            rt_fl<1, 4>::col_vec linear_norm_vec; // sliding filled in through norm exchange
            rt_fl_1x1<> norm_tile;
            zero(norm_tile);
            zero(linear_o);

            // matmul to generate linear_q and linear_k before softmax

            rt_fl<1, 4> linear_q[2];

            warpgroup::mma_fence(linear_q[1]);
            warpgroup::mm_AB(linear_q[1], q_smem[tic], qf_map); // reset
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait(); // q is now projected

            copy(linear_q[0], linear_q[1]);
            mul(linear_q[1], linear_q[1], -1.f); // 1 is the negative version

            #pragma unroll
            for(int i = 0; i < 2; i++) {
                softmax_featuremap_inplace(linear_q[i]);
                mul(linear_q[i], linear_q[i], alpha);
                rt_bf<1, 4> linear_q_bf;
                copy(linear_q_bf, linear_q[i]); // now to bf16

                // copy the local KV cache into shared memory to shared memory and do matmul
                warpgroup::store(kv_smem[i], local_kv[i]);
                asm volatile("bar.sync 11, 128;\n"); // just this warpgroup needs to hit this
                warpgroup::mma_fence(linear_o);
                warpgroup::mma_AB(linear_o, linear_q_bf, kv_smem[i]);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                // next we need to go figure out the norm.
                // first we load sum(k) from smem to registers.
                row_vec<rt_bf<1,4>> cumsum_k_reg;
                load(cumsum_k_reg, cumsum_k_smem[i]);
                // now we can project this up into a register tile
                // we're broadcasting along the column axis (filling all rows with the same value)
                rt_bf<1,4> cumsum_k_reg_tile;
                broadcast_col(cumsum_k_reg_tile, cumsum_k_reg);
                // next we matmul! this gives us a tile.
                mma_ABt(norm_tile, linear_q_bf, cumsum_k_reg_tile, norm_tile);
                // ^ note this incorporates alpha since it was premultiplied onto linear_q!
            }
            row_max(linear_norm_vec, norm_tile); // technically any column slice would work but this is EZ PZ

            // we can now do the first chunk of our norm exchange
            tma::store_async_wait(); // make sure o_smem is not in flight
            wait(inter_retrieved, tic); // although this is on tic, it's okay because the other warpgroup arrives at the start.
            warpgroup::store(norm_exchange, linear_norm_vec);
            warpgroup::store(o_smem, linear_o);
            asm volatile("bar.sync 11, 128;\n");
            if(laneid() == 0) arrive(inter_stored); // we have successfully stored the norm vector and our o, and will leave the rest to the other warpgroup
            
            // now accumulate KV onto the matmul for the future.
            rt_fl<1, 4> linear_k[2];
            row_vec<rt_fl<1,4>> k_sum;

            // matmul to generate linear_k before softmax
            warpgroup::mma_fence(linear_k[1]);
            warpgroup::mm_AB(linear_k[1], k_smem[ring_id], kf_map); // reset
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait(); // k is now projected

            copy(linear_k[0], linear_k[1]);
            mul(linear_k[1], linear_k[1], -1.f); // again, 1 means the negative version
            #pragma unroll
            for(int i = 0; i < 2; i++) {
                softmax_featuremap_inplace(linear_k[i]);
                col_sum(k_sum, linear_k[i]);
                store(cumsum_k_smem_scratch[i][warpid-4], k_sum);
            }

            warpgroup::store(k_scratch_smem[0], linear_k[0]); // screw it, this is now just a scratchpad.
            warpgroup::store(k_scratch_smem[1], linear_k[1]); // screw it, this is now just a scratchpad.
            asm volatile("bar.sync 11, 128;\n");

            warpgroup::mma_fence(local_kv[0]);
            warpgroup::mma_fence(local_kv[1]);
            warpgroup::mma_AtB(local_kv[0], k_scratch_smem[0], v_smem[ring_id]);
            warpgroup::mma_AtB(local_kv[1], k_scratch_smem[1], v_smem[ring_id]);
            warpgroup::mma_commit_group();
            if(warpid == 4 || warpid == 5) cumulative_add(cumsum_k_smem[0], cumsum_k_smem_scratch[0]);
            else                           cumulative_add(cumsum_k_smem[1], cumsum_k_smem_scratch[1]);
            warpgroup::mma_async_wait();

            if(laneid() == 0) arrive(compute_done[tic]);
        }
        wait(compute_done[toc], ((blocks-1)/2)%2); // ensure other warpgroup has finished, too, before we overwrite q_smem and k_smem
        kv_state_tile (&kv_state_smem_tmp) = reinterpret_cast<kv_state_tile&>(q_smem[0].data[0]); // could also do this using a subtile but i like this
        auto st1 = subtile_inplace<4,8>(kv_state_smem_tmp, 0, 0);
        warpgroup::store(st1, local_kv[0]);
        auto st2 = subtile_inplace<4,8>(kv_state_smem_tmp, 1, 0);
        warpgroup::store(st2, local_kv[1]);
    }

    // Finally we want to write out the kv state and the k state
    __syncthreads();
    if(warpid == 0) {
        // write out kv state
        kv_state_tile (&kv_state_smem_st)  = reinterpret_cast<kv_state_tile&>(q_smem[0].data[0]); // we can overwrite early stuff, it's fine
        tma::store_async(tma_kv_state, kv_state_smem_st, batch_head_id);
        tma::store_commit_group();
        // reinterpret k state as a vector of length 128, to save a tma call
        k_state_vec (&k_state_smem) = *reinterpret_cast<k_state_vec*>(&cumsum_k_smem[0].data[0]);
        tma::store_async(tma_k_state, k_state_smem, batch_head_id);
        tma::store_commit_group();
    }
    __syncthreads();
    tma::store_async_wait();
}

#ifdef TORCH_COMPILE
#include "common/pyutils/torch_helpers.cuh"
#include <iostream>

void hedgehog_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
    torch::Tensor k_state, torch::Tensor kv_state,
    torch::Tensor q_map, torch::Tensor k_map,
    torch::Tensor alphas, torch::Tensor betas
) {
    // get general parameters to check
    TORCH_CHECK(q.dim() == 4, "q must have 4 dimensions (B,H,N,D)");
    auto batch = q.size(0);
    auto heads = q.size(1);
    auto N = q.size(2);
    TORCH_CHECK(N>0 && N%64 == 0, "N must be a multiple of 64");
    auto D = q.size(3);
    TORCH_CHECK(D == 128, "D must be 128");

    std::cout << "batch: " << batch << " heads: " << heads << " N: " << N << " D: " << D << std::endl;

    // check K, V, O dimensions, too.
    TORCH_CHECK(k.dim() == 4 && k.size(0) == batch && k.size(1) == heads && v.size(2) == N && k.size(3) == D, "k must be (B,H,N,128)");
    TORCH_CHECK(v.dim() == 4 && v.size(0) == batch && v.size(1) == heads && v.size(2) == N && v.size(3) == D, "v must be (B,H,N,128)");
    TORCH_CHECK(o.dim() == 4 && o.size(0) == batch && o.size(1) == heads && o.size(2) == N && o.size(3) == D, "o must be (B,H,N,128)");

    // Check the rest of Q,K,V,O attributes
    CHECK_INPUT(q); 
    CHECK_INPUT(k); 
    CHECK_INPUT(v); 
    CHECK_INPUT(o);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bf16");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bf16");
    TORCH_CHECK(o.scalar_type() == torch::kBFloat16, "o must be bf16");

    // check k_state, kv_state inputs
    CHECK_INPUT(k_state);
    CHECK_INPUT(kv_state);
    TORCH_CHECK(k_state.dim() == 3 && k_state.size(0) == batch && k_state.size(1) == heads && k_state.size(2) == 128, "k_state must be (B,H,128)");
    TORCH_CHECK(kv_state.dim() == 4 && kv_state.size(0) == batch && kv_state.size(1) == heads && kv_state.size(2) == 128 && kv_state.size(3) == 128, "kv_state must be (B,H,128,128)");
    TORCH_CHECK(k_state.scalar_type() == torch::kFloat32, "k_state must be fp32");
    TORCH_CHECK(kv_state.scalar_type() == torch::kFloat32, "kv_state must be fp32");

    // check q_map, k_map inputs
    CHECK_INPUT(q_map);
    CHECK_INPUT(k_map);
    TORCH_CHECK(q_map.dim() == 3 && q_map.size(0) == heads && q_map.size(1) == 128 && q_map.size(2) == 64, "q_map must have Hx128x64 shape");
    TORCH_CHECK(k_map.dim() == 3 && k_map.size(0) == heads && k_map.size(1) == 128 && k_map.size(2) == 64, "k_map must have Hx128x64 shape");
    TORCH_CHECK(q_map.scalar_type() == torch::kBFloat16, "q_map must be bf16");
    TORCH_CHECK(k_map.scalar_type() == torch::kBFloat16, "k_map must be bf16");

    CHECK_INPUT(alphas);
    CHECK_INPUT(betas);
    TORCH_CHECK(alphas.dim() == 1 && alphas.size(0) == heads, "alphas must be of shape (H,)");
    TORCH_CHECK(betas.dim() == 1 && betas.size(0) == heads, "betas must be of shape (H,)");
    TORCH_CHECK(alphas.scalar_type() == torch::kFloat32, "alphas must be fp32");
    TORCH_CHECK(betas.scalar_type() == torch::kFloat32, "betas must be fp32");

    c10::BFloat16 *q_ptr        = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_ptr        = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_ptr        = v.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr        = o.data_ptr<c10::BFloat16>();
    float         *k_state_ptr  = k_state.data_ptr<float>();
    float         *kv_state_ptr = kv_state.data_ptr<float>();
    c10::BFloat16 *q_map_ptr    = q_map.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_map_ptr    = k_map.data_ptr<c10::BFloat16>();
    float         *alphas_ptr   = alphas.data_ptr<float>();
    float         *betas_ptr    = betas.data_ptr<float>();

    const bf16* d_q = reinterpret_cast<const bf16*>(q_ptr); 
    const bf16* d_k = reinterpret_cast<const bf16*>(k_ptr);  
    const bf16* d_v = reinterpret_cast<const bf16*>(v_ptr);  
    bf16* d_o = reinterpret_cast<bf16*>(o_ptr);
    float* d_kv_state = reinterpret_cast<float*>(kv_state_ptr);  
    float* d_k_state  = reinterpret_cast<float*>(k_state_ptr);
    const bf16* d_q_map = reinterpret_cast<const bf16*>(q_map_ptr);
    const bf16* d_k_map = reinterpret_cast<const bf16*>(k_map_ptr);
    const float* d_alphas = reinterpret_cast<const float*>(alphas_ptr);
    const float* d_betas = reinterpret_cast<const float*>(betas_ptr);

    CUtensorMap* tma_q_map_d     = tma::allocate_and_create_tensor_map<qk_map_tile>(d_q_map, heads); 
    CUtensorMap* tma_k_map_d     = tma::allocate_and_create_tensor_map<qk_map_tile>(d_k_map, heads);
    CUtensorMap* tma_q_d         = tma::allocate_and_create_tensor_map<q_tile>(d_q, batch*heads*N/q_tile::rows); 
    CUtensorMap* tma_k_d         = tma::allocate_and_create_tensor_map<k_tile>(d_k, batch*heads*N/k_tile::rows);
    CUtensorMap* tma_v_d         = tma::allocate_and_create_tensor_map<v_tile>(d_v, batch*heads*N/v_tile::rows);
    CUtensorMap* tma_o_d         = tma::allocate_and_create_tensor_map<o_tile>(d_o, batch*heads*N/o_tile::rows);
    CUtensorMap* tma_k_state_d   = tma::allocate_and_create_tensor_map<k_state_vec>(d_k_state, batch*heads); 
    CUtensorMap* tma_kv_state_d  = tma::allocate_and_create_tensor_map<kv_state_tile>(d_kv_state, batch*heads);

    constexpr unsigned long mem_size = kittens::MAX_SHARED_MEMORY;
    cudaFuncSetAttribute(
        hedgehog_linear_attention_smd,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    hedgehog_linear_attention_smd<<<dim3(heads,batch), NUM_THREADS, mem_size>>>(
        N,
        tma_q_d, tma_k_d, tma_v_d,
        tma_o_d,
        tma_k_state_d, tma_kv_state_d,
        tma_q_map_d, tma_k_map_d,
        d_alphas, d_betas
    ); 

    CHECK_CUDA_ERROR(cudaGetLastError());
}

#else
#include "harness.impl"
#endif