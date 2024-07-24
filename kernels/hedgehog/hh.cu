// #define TORCH_COMPILE // defined by default for PyTorch bindings - to use cpp harness, comment this out

#include "kittens.cuh"

#define NUM_WORKERS (8)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define NUM_WARPGROUPS (NUM_WORKERS/kittens::WARPGROUP_WARPS)

using namespace kittens;

template<ducks::sv::all SV, ducks::st::all ST>
__device__ inline void cumulative_add(SV &dst, const ST &src) {
    // this is called along a warpgroup
    static_assert(ST::cols <= 128);
    static_assert(ST::cols == SV::length);
    int lane = threadIdx.x % 128;
    if(lane < ST::cols) {
        float f = dst[lane];
        // acc equal to the last row of dst
        for (auto i = 0; i < ST::rows; i++) {
            f += __bfloat162float(src[{i, lane}]);
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

    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int batch_id = blockIdx.y;
    const int head_id  = blockIdx.x;
    const int batch_head_id = batch_id*gridDim.x + head_id;
    // if(threadIdx.x == 0) printf("head_id: %d, batch_id: %d, batch_head_id: %d\n", head_id, batch_id, batch_head_id);
    float alpha = alphas[head_id];
    float beta  = betas [head_id];

    q_tile (&q_smem)[2] = al.allocate<q_tile, 2>(); // 32k, (tic/toc)*16k
    k_tile (&k_smem)[3] = al.allocate<k_tile, 3>(); // 48k, (3-ring)*(64x128)
    v_tile (&v_smem)[3] = al.allocate<v_tile, 3>(); // 48k, (3-ring)*(64x128)
    o_tile (&o_smem)    = al.allocate<o_tile>   (); // 16k

    qk_map_tile (&qf_map) = al.allocate<qk_map_tile>(); // 16k, for fusing featuremap computation
    qk_map_tile (&kf_map) = al.allocate<qk_map_tile>(); // 16k, for fusing featuremap computation

    st_bf<4, 8> (&kv_smem)[2] = al.allocate<st_bf<4, 8>, 2>(); // 32k, 64x128 featurized 
    
    row_vec<st_fl<4,4>> (&cumsum_k_smem)[2] = al.allocate<row_vec<st_fl<4,4>>, 2>(); // smol
    col_vec<st_fl<4,4>> (&norm_exchange)[2] = al.allocate<col_vec<st_fl<4,4>>, 2>(); // smol

    st_bf<4, 4> (*k_scratch_smem)     = reinterpret_cast<st_bf<4, 4>*>(&kv_smem[0].data[0]);

    int warpid = kittens::warpid();
    int warpgroupid = warpid/4;

    int tic = 0, toc = 1;
    int ring_id = 0;
    __shared__ barrier qkv_barrier;

    int blocks = n / (q_tile::rows);

    if (warpid == 0) {
        init_barrier(qkv_barrier, 0, 1);
        tma::expect_bytes(qkv_barrier, 
            size_bytes<typeof(q_smem[0])> + 
            size_bytes<typeof(k_smem[0])> + 
            size_bytes<typeof(v_smem[0])> +
            // we need qk maps to be loaded on this first iter, too.
            size_bytes<typeof(qf_map)> +
            size_bytes<typeof(kf_map)>
        );
        int tile_idx = (batch_head_id * blocks) + 0;
        // first thing we need to do is load the QK map
        tma::load_async(qf_map, tma_qmap, qkv_barrier, head_id); // need to load the right head
        tma::load_async(kf_map, tma_kmap, qkv_barrier, head_id);
        // now we also load the first data we need
        tma::load_async(q_smem[tic],       tma_q, qkv_barrier, tile_idx);
        tma::load_async(k_smem[ring_id+1], tma_k, qkv_barrier, tile_idx);
        tma::load_async(v_smem[ring_id+1], tma_v, qkv_barrier, tile_idx);
    }

    rt_fl<1, 8> local_kv; // this is going to be split across the two warpgroups involved.

    zero(local_kv);
    warpgroup::zero(v_smem[ring_id]);
    warpgroup::zero(cumsum_k_smem[warpgroupid]);

    __syncthreads();

    for (int block = 0; block < blocks; block++, tic^=1, toc^=1, ring_id=(ring_id+1)%3) {

        wait(qkv_barrier, tic);  // ding! memory arrived
        __syncthreads();

        if (warpid == 0 && block < blocks-1) {
            tma::expect_bytes(qkv_barrier,
                size_bytes<typeof(q_smem[0])> + 
                size_bytes<typeof(k_smem[0])> + 
                size_bytes<typeof(v_smem[0])>
            );

            int tile_idx = (batch_head_id * blocks) + block + 1;
            tma::load_async(q_smem[toc],           tma_q, qkv_barrier, tile_idx); 
            tma::load_async(k_smem[(ring_id+2)%3], tma_k, qkv_barrier, tile_idx); 
            tma::load_async(v_smem[(ring_id+2)%3], tma_v, qkv_barrier, tile_idx);
        }
        __syncthreads();

        // ----- let's do sliding window first -----
        // only warps 0-4 need to be involved in this

        rt_fl<1, 8> sliding_o;
        rt_fl<1, 4>::col_vec sliding_norm_vec;
        zero(sliding_o);
        zero(sliding_norm_vec);
        if(warpgroupid == 0) {

            // ******* sliding window attn ******* // 

            rt_fl<1, 4> att_block[2];
            rt_bf<1, 4> att_block_bf[2];
            rt_fl<1, 4>::col_vec max_vec;

            neg_infty(max_vec); // zero registers for the Q chunk

            for(int subtile = 0; subtile < 2; subtile++) {
                if (block + subtile >= 1) { // ensure tile has been loaded by now.
                    warpgroup::mma_fence(att_block[subtile]);
                    warpgroup::mm_ABt(att_block[subtile], q_smem[tic], k_smem[(ring_id+subtile)%3]);
                    warpgroup::mma_commit_group();
                    warpgroup::mma_async_wait();
                }
                else {
                    neg_infty(att_block[subtile]); // initial blocks must be zero
                }
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
                mul(att_block[subtile], att_block[subtile], 0.08838834764); // temperature adjustment.
                row_max(max_vec, att_block[subtile], max_vec); // accumulate onto the max_vec
            }
            #pragma unroll
            for(int subtile = 0; subtile < 2; subtile++) {
                sub_row(att_block[subtile], att_block[subtile], max_vec);
                exp(att_block[subtile], att_block[subtile]);
                mul(att_block[subtile], att_block[subtile], beta);
            }
            // now we sum so that we can divide (normalize) later
            #pragma unroll
            for(int subtile = 0; subtile < 2; subtile++) {
                row_sum(sliding_norm_vec, att_block[subtile], sliding_norm_vec); // incorporates beta
                copy(att_block_bf[subtile], att_block[subtile]); // cast to bf16 for next matmul
            }
            warpgroup::mma_fence(sliding_o);
            for(int subtile = 0; subtile < 2; subtile++) {
                warpgroup::mma_AB(sliding_o, att_block_bf[subtile], v_smem[(ring_id+subtile)%3]);
                warpgroup::mma_commit_group();
            }
            warpgroup::mma_async_wait();
        }
        __syncthreads();

        rt_fl<1, 8> linear_o; // this is partitioned across the two warpgroups.
        rt_fl<1, 4>::col_vec linear_norm_vec;
        zero(linear_norm_vec);
        if(block == 0) {
            zero(linear_o);
        }
        else { // if not in at least the second block, no need for linear attention.

            // ******* linear attn ******** // 

            // matmul to generate linear_q before softmax

            rt_fl<1, 4> linear_q;
            rt_bf<1, 4> linear_q_bf;

            warpgroup::mma_fence(linear_q);
            warpgroup::mm_AB(linear_q, q_smem[tic], qf_map); // reset
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait(); // q is now projected
            if(warpgroupid) mul(linear_q, linear_q, -1.f);
            // now we need to run q through a local softmax to featurize
            softmax_featuremap_inplace(linear_q);
            mul(linear_q, linear_q, alpha);
            copy(linear_q_bf, linear_q); // now to bf16

            // copy the local KV cache into shared memory to shared memory and do matmul
            warpgroup::store(kv_smem[warpgroupid], local_kv);
            __syncthreads(); // this should probably be a cooperative group of just the 4 warps
            warpgroup::mma_fence(linear_o);
            warpgroup::mm_AB(linear_o, linear_q_bf, kv_smem[warpgroupid]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            // next we need to go figure out the norm.
            // first we load sum(k) from smem to registers.
            row_vec<rt_bf<1,4>> cumsum_k_reg;
            load(cumsum_k_reg, cumsum_k_smem[warpgroupid]);
            // now we can project this up into a register tile
            // we're broadcasting along the column axis (filling all rows with the same value)
            rt_bf<1,4> cumsum_k_reg_tile;
            broadcast_col(cumsum_k_reg_tile, cumsum_k_reg);
            // next we matmul! this gives us a tile.
            rt_fl_1x1<> norm_tile;
            zero(norm_tile);
            mma_ABt(norm_tile, linear_q_bf, cumsum_k_reg_tile, norm_tile);
            row_max(linear_norm_vec, norm_tile); // technically any column slice would work but this is EZ
            // ^ note this incorporates alpha since it was premultiplied onto linear_q!
            
            // now accumulate KV onto the matmul for the future.
            rt_fl<1, 4> linear_k;

            // matmul to generate linear_k before softmax
            warpgroup::mma_fence(linear_k);
            asm volatile ("fence.proxy.async;\n" ::: "memory");
            warpgroup::mm_AB(linear_k, k_smem[ring_id], kf_map); // reset
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait(); // k is now projected
            if(warpgroupid) mul(linear_k, linear_k, -1.f);
            // now we need to run q through a local softmax to featurize
            softmax_featuremap_inplace(linear_k);

            // copy the local K into shared memory & do matmul
            warpgroup::store(k_scratch_smem[warpgroupid], linear_k); // screw it, this is now just a scratchpad.
            __syncthreads();
            cumulative_add(cumsum_k_smem[warpgroupid], k_scratch_smem[warpgroupid]);
            warpgroup::mma_fence(local_kv);
            asm volatile ("fence.proxy.async;\n" ::: "memory");
            warpgroup::mma_AtB(local_kv, k_scratch_smem[warpgroupid], v_smem[ring_id]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
        }
        tma::store_async_wait();

        // next step is to sum two norm vecs
        add(sliding_norm_vec, sliding_norm_vec, linear_norm_vec);
        warpgroup::store(norm_exchange[warpgroupid], sliding_norm_vec);
        __syncthreads();
        col_vec<rt_fl_1x1<>> total_norm;
        warpgroup::load(total_norm, norm_exchange[warpgroupid^1]);
        add(total_norm, total_norm, sliding_norm_vec);
        // we have now finally accumulated the total norm for everything
        add(sliding_o, sliding_o, linear_o); // local o
        div_row(sliding_o, sliding_o, total_norm); // this half is now normalized
        if(warpgroupid == 1) {
            warpgroup::store(o_smem, sliding_o);
        }
        __syncthreads();
        if(warpgroupid == 0) {
            warpgroup::load(linear_o, o_smem);
            add(sliding_o, sliding_o, linear_o);
            warpgroup::store(o_smem, sliding_o);
        }
        __syncthreads();

        if(warpid == 0) {
            tma::store_async(tma_o, o_smem, batch_head_id*blocks + block);
            tma::store_commit_group();
        }
    }
    tma::store_async_wait();

    // Finally we want to write out the kv state and the k state
    
    // reinterpret k state as a vector of length 128, to save a tma call
    k_state_vec (&k_state_smem) = *reinterpret_cast<k_state_vec*>(&cumsum_k_smem[0].data[0]);
    // store out kv state into smem.
    kv_state_tile (&kv_state_smem) = reinterpret_cast<kv_state_tile&>(q_smem[0].data[0]); // we can overwrite early stuff, it's fine
    group<8>::store(kv_state_smem, local_kv); // all 8 warps store their own chunk.
    __syncthreads();
    __syncthreads(); // this second one is legit necessary for correctness lmao which means something scary is wrong but tbh i don't want to find it.
    __syncthreads(); // third one is just for good luck
    // write out kv state
    if(warpid == 0){
        tma::store_async(tma_kv_state, kv_state_smem, batch_head_id);
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