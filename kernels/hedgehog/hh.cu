#include "kittens.cuh"
#include <tuple>

#ifdef TORCH_COMPILE
#define TK_COMPILE_HEDGEHOG
#endif

static constexpr int NUM_WORKERS = (8);
static constexpr int NUM_THREADS = (NUM_WORKERS*kittens::WARP_THREADS);
static constexpr int NUM_WARPGROUPS = (NUM_WORKERS/kittens::WARPGROUP_WARPS);

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
    exp2(tile, tile);
    row_sum(sum_vec, tile);
    div_row(tile, tile, sum_vec);
}

#define CHUNK_SIZE 64
#define ATTN_D 128
#define ATTN_F 128
#define HALF_ATTN_F 64

struct hedgehog_globals { 
    // shapes    
    using q_tile = st_bf<CHUNK_SIZE, ATTN_F>;
    using k_tile = st_bf<CHUNK_SIZE, ATTN_F>;
    using v_tile = st_bf<CHUNK_SIZE, ATTN_D>;
    using o_tile = st_bf<CHUNK_SIZE, ATTN_D>;
    using kv_state_tile = st_fl<ATTN_F, ATTN_D>;
    using k_state_vec = sv_fl<ATTN_D>;
    using qk_map_tile = st_bf<ATTN_D, HALF_ATTN_F>;

    // global layouts
    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using qmap_gl = gl<bf16,  -1, -1, -1, -1, qk_map_tile>;
    using kmap_gl = gl<bf16,  -1, -1, -1, -1, qk_map_tile>;
    using k_state_gl = gl<float, -1, -1, -1, -1, k_state_vec>;
    using kv_state_gl = gl<float, -1, -1, -1, -1, kv_state_tile>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;

    // pointers
    q_gl q;
    k_gl k;
    v_gl v;
    qmap_gl qmap;
    kmap_gl kmap;
    k_state_gl k_state;
    kv_state_gl kv_state;
    o_gl o;

    float *alphas; 
    float *betas;
};


// should be launched with a grid of size (HEADS, BATCH) and blocks of 256 threads.
__global__ __launch_bounds__(NUM_THREADS, 1)
void hedgehog_linear_attention_smd (
    const __grid_constant__ hedgehog_globals g, int N
)  { // alpha is for linear component, beta is for sliding window component. Array, per head.

    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int batch = blockIdx.y;
    const int head  = blockIdx.x;
    // const int batch_head_id = batch*gridDim.x + head;

    float alpha = g.alphas[head];
    float beta  = g.betas [head];

    // smem
    using q_tile = st_bf<CHUNK_SIZE, ATTN_F>;
    using k_tile = st_bf<CHUNK_SIZE, ATTN_F>;
    using v_tile = st_bf<CHUNK_SIZE, ATTN_D>;
    using o_tile = st_bf<CHUNK_SIZE, ATTN_D>;
    using kv_state_tile = st_fl<ATTN_F, ATTN_D>;
    using k_state_vec = sv_fl<ATTN_D>;
    using qk_map_tile = st_bf<ATTN_D, HALF_ATTN_F>;
    q_tile (&q_smem)[2] = al.allocate<q_tile, 2>(); // 32k, (tic/toc)*16k
    k_tile (&k_smem)[3] = al.allocate<k_tile, 3>(); // 48k, (3-ring)*(64x128)
    v_tile (&v_smem)[3] = al.allocate<v_tile, 3>(); // 48k, (3-ring)*(64x128)
    o_tile (&o_smem)    = al.allocate<o_tile>   (); // 16k
    qk_map_tile (&qf_map) = al.allocate<qk_map_tile>(); // 16k, for fusing featuremap computation
    qk_map_tile (&kf_map) = al.allocate<qk_map_tile>(); // 16k, for fusing featuremap computation

    // norm stuff
    st_bf<CHUNK_SIZE, ATTN_F> (&kv_smem)[2] = al.allocate<st_bf<CHUNK_SIZE, ATTN_F>, 2>(); // 32k, 64x128 featurized 
    row_vec<st_fl<CHUNK_SIZE,4*16>> (&cumsum_k_smem)[2] = al.allocate<row_vec<st_fl<CHUNK_SIZE,4*16>>, 2>(); // smol
    col_vec<st_fl<CHUNK_SIZE,4*16>> (&norm_exchange)[2] = al.allocate<col_vec<st_fl<CHUNK_SIZE,4*16>>, 2>(); // smol
    st_bf<CHUNK_SIZE, 4*16> (*k_scratch_smem)     = reinterpret_cast<st_bf<CHUNK_SIZE, 4*16>*>(&kv_smem[0].data[0]);

    int warpid = kittens::warpid();
    int warpgroupid = warpid/4;
    int blocks = N / (q_tile::rows);

    int tic = 0, toc = 1;
    int ring_id = 0;

    __shared__ semaphore qkv_semaphore;
    if (warpid == 0) {
        init_semaphore(qkv_semaphore, 0, 1);
        tma::expect_bytes(qkv_semaphore, 
            size_bytes<typeof(q_smem[0])> + 
            size_bytes<typeof(k_smem[0])> + 
            size_bytes<typeof(v_smem[0])> +
            // we need qk maps to be loaded on this first iter, too.
            size_bytes<typeof(qf_map)> +
            size_bytes<typeof(kf_map)>
        );
        // first thing we need to do is load the QK map
        tma::load_async(qf_map, g.qmap, {0, head, 0, 0}, qkv_semaphore); // load the right head
        tma::load_async(kf_map, g.kmap, {0, head, 0, 0}, qkv_semaphore);
        // now we also load the first data we need
        tma::load_async(q_smem[tic],  g.q,      {batch, head, 0, 0}, qkv_semaphore);
        tma::load_async(k_smem[ring_id+1], g.k, {batch, head, 0, 0}, qkv_semaphore);
        tma::load_async(v_smem[ring_id+1], g.v, {batch, head, 0, 0}, qkv_semaphore);
    }

    rt_fl<1*16, 8*16> local_kv; // this is going to be split across the two warpgroups involved.

    zero(local_kv);
    warpgroup::zero(v_smem[ring_id]);
    warpgroup::zero(cumsum_k_smem[warpgroupid]);

    __syncthreads();

    for (int block = 0; block < blocks; block++, tic^=1, toc^=1, ring_id=(ring_id+1)%3) {

        wait(qkv_semaphore, tic);  // ding! memory arrived
        __syncthreads();

        if (warpid == 0 && block < blocks-1) {
            tma::expect_bytes(qkv_semaphore,
                size_bytes<typeof(q_smem[0])> + 
                size_bytes<typeof(k_smem[0])> + 
                size_bytes<typeof(v_smem[0])>
            );
            tma::load_async(q_smem[toc],           g.q, {batch, head, block+1, 0}, qkv_semaphore); 
            tma::load_async(k_smem[(ring_id+2)%3], g.k, {batch, head, block+1, 0}, qkv_semaphore); 
            tma::load_async(v_smem[(ring_id+2)%3], g.v, {batch, head, block+1, 0}, qkv_semaphore);
        }
        __syncthreads();

        // ----- let's do sliding window first -----
        // only warps 0-4 need to be involved in this

        rt_fl<1*16, 8*16> sliding_o;
        rt_fl<1*16, 4*16>::col_vec sliding_norm_vec;
        zero(sliding_o);
        zero(sliding_norm_vec);
        if(warpgroupid == 0) {

            // ******* sliding window attn ******* // 
            rt_fl<1*16, 4*16> att_block[2];
            rt_bf<1*16, 4*16> att_block_bf[2];
            rt_fl<1*16, 4*16>::col_vec max_vec;

            neg_infty(max_vec); // zero registers for the Q chunk

            for(int subtile = 0; subtile < 2; subtile++) {
                if (block + subtile >= 1) { // ensure tile has been loaded by now.
                    warpgroup::mm_ABt(att_block[subtile], q_smem[tic], k_smem[(ring_id+subtile)%3]);
                    warpgroup::mma_async_wait();
                }
                else {
                    neg_infty(att_block[subtile]); // initial blocks must be zero
                }
            }
            // make last block causal
            #pragma unroll
            for(int j = 0; j < 4; j++) {
                auto &attn_subtile = reinterpret_cast<rt_fl<1*16,1*16>&>(att_block[1].tiles[0][j]);
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
            for(int subtile = 0; subtile < 2; subtile++) {
                warpgroup::mma_AB(sliding_o, att_block_bf[subtile], v_smem[(ring_id+subtile)%3]);
            }
            warpgroup::mma_async_wait();
        }
        __syncthreads();

        rt_fl<1*16, 8*16> linear_o; // this is partitioned across the two warpgroups.
        rt_fl<1*16, 4*16>::col_vec linear_norm_vec;
        zero(linear_norm_vec);
        if(block == 0) {
            zero(linear_o);
        }
        else { // if not in at least the second block, no need for linear attention.
            // ******* linear attn ******** // 

            // matmul to generate linear_q before softmax
            rt_fl<1*16, 4*16> linear_q;
            rt_bf<1*16, 4*16> linear_q_bf;

            warpgroup::mm_AB(linear_q, q_smem[tic], qf_map); // reset
            warpgroup::mma_async_wait(); // q is now projected
            if(warpgroupid) mul(linear_q, linear_q, -1.44269504089f);
            else            mul(linear_q, linear_q,  1.44269504089f);
            // now we need to run q through a local softmax to featurize
            softmax_featuremap_inplace(linear_q);
            mul(linear_q, linear_q, alpha);
            copy(linear_q_bf, linear_q); // now to bf16

            // copy the local KV cache into shared memory to shared memory and do matmul
            warpgroup::store(kv_smem[warpgroupid], local_kv);
            __syncthreads(); // this should probably be a cooperative group of just the 4 warps
            warpgroup::mm_AB(linear_o, linear_q_bf, kv_smem[warpgroupid]);
            warpgroup::mma_async_wait();

            // next we need to go figure out the norm.
            // first we load sum(k) from smem to registers.
            row_vec<rt_bf<1*16,4*16>> cumsum_k_reg;
            load(cumsum_k_reg, cumsum_k_smem[warpgroupid]);
            // now we can project this up into a register tile
            // we're broadcasting along the column axis (filling all rows with the same value)
            rt_bf<1*16,4*16> cumsum_k_reg_tile;
            broadcast_col(cumsum_k_reg_tile, cumsum_k_reg);
            // next we matmul! this gives us a tile.
            rt_fl<1*16,1*16> norm_tile;
            zero(norm_tile);
            mma_ABt(norm_tile, linear_q_bf, cumsum_k_reg_tile, norm_tile);
            row_max(linear_norm_vec, norm_tile); // technically any column slice would work but this is EZ
            // ^ note this incorporates alpha since it was premultiplied onto linear_q!
            
            // now accumulate KV onto the matmul for the future.
            rt_fl<1*16, 4*16> linear_k;

            // matmul to generate linear_k before softmax
            warpgroup::mm_AB(linear_k, k_smem[ring_id], kf_map); // reset
            warpgroup::mma_async_wait(); // k is now projected
            if(warpgroupid) mul(linear_k, linear_k, -1.44269504089f);
            else            mul(linear_k, linear_k,  1.44269504089f);
            // now we need to run q through a local softmax to featurize
            softmax_featuremap_inplace(linear_k);

            // copy the local K into shared memory & do matmul
            warpgroup::store(k_scratch_smem[warpgroupid], linear_k); // screw it, this is now just a scratchpad.
            __syncthreads();
            cumulative_add(cumsum_k_smem[warpgroupid], k_scratch_smem[warpgroupid]);
            warpgroup::mma_AtB(local_kv, k_scratch_smem[warpgroupid], v_smem[ring_id]);
            warpgroup::mma_async_wait();
        }
        tma::store_async_wait();

        // next step is to sum two norm vecs
        if(warpgroupid == 1) {
            warpgroup::store(o_smem, linear_o);
            warpgroup::store(norm_exchange[0], linear_norm_vec);
        }
        else {
            add(sliding_norm_vec, sliding_norm_vec, linear_norm_vec);
            add(sliding_o, sliding_o, linear_o);
        }
        __syncthreads();
        if(warpgroupid == 0) {
            warpgroup::load(linear_o, o_smem);
            warpgroup::load(linear_norm_vec, norm_exchange[0]);
            add(sliding_o, sliding_o, linear_o);
            add(sliding_norm_vec, sliding_norm_vec, linear_norm_vec);
            div_row(sliding_o, sliding_o, sliding_norm_vec); // this half is now normalized
            warpgroup::store(o_smem, sliding_o);
        }
        __syncthreads();

        if(warpid == 0) {
            tma::store_async(g.o, o_smem, {batch, head, block, 0});
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
    // write out kv state
    if(warpid == 0){
        tma::store_async(g.kv_state, kv_state_smem, {batch, head, 0, 0});
        tma::store_async(g.k_state, k_state_smem, {batch, head, 0, 0});
        tma::store_commit_group();
    }
    __syncthreads();
    tma::store_async_wait();
}

hedgehog_globals hedgehog_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    bf16 *d_qmap, bf16 *d_kmap,
    float *d_k_state, float *d_kv_state,
    float *d_alphas, float *d_betas,
    int ATTN_B, int ATTN_H, int ATTN_N
) {
    // global pointers. 
    using q_tile = st_bf<CHUNK_SIZE, ATTN_F>;
    using k_tile = st_bf<CHUNK_SIZE, ATTN_F>;
    using v_tile = st_bf<CHUNK_SIZE, ATTN_D>;
    using o_tile = st_bf<CHUNK_SIZE, ATTN_D>;
    using kv_state_tile = st_fl<ATTN_F, ATTN_D>;
    using k_state_vec = sv_fl<ATTN_D>;
    using qk_map_tile = st_bf<ATTN_D, HALF_ATTN_F>;
    
    using q_global = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_global = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_global = gl<bf16, -1, -1, -1, -1, v_tile>;
    using o_global = gl<bf16, -1, -1, -1, -1, o_tile>;
    using kv_state_global = gl<float, -1, -1, -1, -1, kv_state_tile>;
    using k_state_global = gl<float, -1, -1, -1, -1, k_state_vec>;
    using qmap_global = gl<bf16, -1, -1, -1, -1, qk_map_tile>;
    using kmap_global = gl<bf16, -1, -1, -1, -1, qk_map_tile>;
    
    using globals = hedgehog_globals;
    q_global q_arg{d_q, ATTN_B, ATTN_H, ATTN_N, ATTN_F};
    k_global k_arg{d_k, ATTN_B, ATTN_H, ATTN_N, ATTN_F};
    v_global v_arg{d_v, ATTN_B, ATTN_H, ATTN_N, ATTN_D};
    o_global o_arg{d_o, ATTN_B, ATTN_H, ATTN_N, ATTN_D};
    qmap_global qmap_arg{d_qmap, 1, ATTN_H, ATTN_F, HALF_ATTN_F};
    kmap_global kmap_arg{d_kmap, 1, ATTN_H, ATTN_F, HALF_ATTN_F};
    kv_state_global kv_state_arg{d_kv_state, ATTN_B, ATTN_H, ATTN_F, ATTN_D};
    k_state_global k_state_arg{d_k_state, ATTN_B, ATTN_H, 1, ATTN_D};

    globals g{
        q_arg, k_arg, v_arg, 
        qmap_arg, kmap_arg,
        k_state_arg, kv_state_arg,
        o_arg, d_alphas, d_betas
    };
    return g;
}

#ifdef TK_COMPILE_HEDGEHOG
#include "pyutils/torchutils.cuh"
#include <iostream>
void dispatch_hedgehog( 
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    bf16 *d_qmap, bf16 *d_kmap,
    float *d_k_state, float *d_kv_state,
    float *d_alphas, float *d_betas,
    int ATTN_B, int ATTN_H, int ATTN_N
){
    hedgehog_globals g = hedgehog_init(
        d_q, d_k, d_v, d_o,
        d_qmap, d_kmap,
        d_k_state, d_kv_state,
        d_alphas, d_betas,
        ATTN_B, ATTN_H, ATTN_N
    );

    // launch
    unsigned long mem_size = kittens::MAX_SHARED_MEMORY;
    cudaFuncSetAttribute(
        hedgehog_linear_attention_smd,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(ATTN_H, ATTN_B);
    hedgehog_linear_attention_smd<<<grid,NUM_THREADS,mem_size>>>(g, ATTN_N);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hedgehog(
    const torch::Tensor q, 
    const torch::Tensor k,
    const torch::Tensor v,
    const torch::Tensor qmap,
    const torch::Tensor kmap,
    const torch::Tensor d_alphas,
    const torch::Tensor d_betas
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(qmap);
    CHECK_INPUT(kmap);
    CHECK_INPUT(d_alphas);
    CHECK_INPUT(d_betas);

    int B = q.size(0);
    int H = q.size(1);
    int N = q.size(2);
    int DV = v.size(3);
    int FD = qmap.size(1);

    // checks
    TORCH_CHECK(k.size(0) == B, "k batch?");
    TORCH_CHECK(k.size(1) == H, "k heads?");
    TORCH_CHECK(k.size(2) == N, "k length?");

    TORCH_CHECK(v.size(0) == B, "v batch?");
    TORCH_CHECK(v.size(1) == H, "v heads?");
    TORCH_CHECK(v.size(2) == N, "v length?");

    TORCH_CHECK(qmap.size(0) == H, "qmap heads?");
    TORCH_CHECK(qmap.size(1) == FD, "qmap length?");
    TORCH_CHECK(qmap.size(2) == 64, "qmap length?");

    TORCH_CHECK(kmap.size(0) == H, "kmap heads?");
    TORCH_CHECK(kmap.size(1) == FD, "kmap length?");
    TORCH_CHECK(kmap.size(2) == 64, "kmap length?");

    TORCH_CHECK(d_alphas.size(0) == H, "alphas heads?");
    TORCH_CHECK(d_betas.size(0) == H, "betas heads?");

    // allocate outputs
    torch::Tensor out = torch::empty({B, H, N, DV}, v.options());
    torch::Tensor kv_state = torch::empty({B, H, FD, DV}, torch::dtype(torch::kFloat32).device(v.device()));
    torch::Tensor k_state = torch::empty({B, H, 1, DV}, torch::dtype(torch::kFloat32).device(v.device()));

    // convert to bf16
    c10::BFloat16 *q_bf16 = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_bf16 = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_bf16 = v.data_ptr<c10::BFloat16>();
    c10::BFloat16 *qmap_bf16 = qmap.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kmap_bf16 = kmap.data_ptr<c10::BFloat16>();
    float *d_alphas_float = d_alphas.data_ptr<float>(); // alpha and beta are already float32
    float *d_betas_float = d_betas.data_ptr<float>();
    // alpha and beta are already float32
    
    bf16 *d_q = reinterpret_cast<bf16*>(q_bf16);
    bf16 *d_k = reinterpret_cast<bf16*>(k_bf16);
    bf16 *d_v = reinterpret_cast<bf16*>(v_bf16);
    bf16 *d_qmap = reinterpret_cast<bf16*>(qmap_bf16);
    bf16 *d_kmap = reinterpret_cast<bf16*>(kmap_bf16);
    bf16 *d_o = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());
    float *d_k_state = k_state.data_ptr<float>();
    float *d_kv_state = kv_state.data_ptr<float>();

    dispatch_hedgehog(
        d_q, d_k, d_v, d_o, 
        d_qmap, d_kmap,
        d_k_state, d_kv_state,
        d_alphas_float, d_betas_float,
        B, H, N
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    return std::make_tuple(out, kv_state, k_state);
}
#else
#include "harness.impl"
#endif

