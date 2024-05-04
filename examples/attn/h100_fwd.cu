#include "src/kittens.cuh"
// #include "../../src/kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

#define NUM_WORKERS (8)
#define NUM_WARPGROUPS (NUM_WORKERS/(kittens::WARPGROUP_WARPS))

#define qo_height (4)
#define kv_height (8)
#define NUM_WORKERS_KV (1)
#define tile_width (64/16)

using namespace kittens;

using layout_q = kittens::ducks::st_layout::wgmma_swizzle; // need to make this 128b
using layout_k = kittens::ducks::st_layout::wgmma_swizzle; // need to make this 128b
using layout_v = kittens::ducks::st_layout::wgmma_interleave; // need to make this 128b
using layout_o = kittens::ducks::st_layout::swizzle;

__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 2)
void fwd_attend_ker(const int N, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, CUtensorMap* tma_o) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    st_bf<qo_height, tile_width, layout_q>          (&q_smem)   [NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_q>,          NUM_WARPGROUPS>();
    st_bf<kv_height, tile_width, layout_k>          (&k_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, tile_width, layout_k>, 2,       NUM_WORKERS_KV>();
    st_bf<kv_height, tile_width, layout_v>          (&v_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, tile_width, layout_v>, 2,       NUM_WORKERS_KV>();

    int tic = 0, toc = 1;
 
    rt_fl<1, kv_height> att_block;
    rt_bf<1, kv_height> att_block_mma;
    rt_fl<1, tile_width> o_prev;
    rt_fl<1, kv_height>::col_vec max_vec_last, max_vec;
    rt_fl<1, kv_height>::col_vec norm_vec_last, norm_vec;

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    int kv_blocks = N / (NUM_WORKERS_KV*k_smem[0][0].rows);

    __shared__ uint64_t qsmem_barrier, kvsmem_barrier;//, vsmem_barrier;

    int q_phasebit = 0;
    int kv_phasebit = 0;

    if (threadIdx.x == 0) {
        tma::init_barrier<st_bf<qo_height, tile_width, layout_q>, NUM_WARPGROUPS>(qsmem_barrier, 1);
        tma::init_barrier<st_bf<kv_height, tile_width, layout_k>, NUM_WORKERS_KV*2>(kvsmem_barrier, 1); 
    }

    if (warpid == 0) {
        for (int wg = 0; wg < NUM_WORKERS/kittens::WARPGROUP_WARPS; wg++) { // load q
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + wg;
            tma::load_async((q_smem[wg]), tma_q, qsmem_barrier, tile_idx); 
        }
        for (int w = 0; w < NUM_WORKERS_KV; w++) { // load k, v      
            int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + (0 * NUM_WORKERS_KV) + w; 
            tma::load_async((k_smem[tic][w]), tma_k, kvsmem_barrier, tile_idx); 
            tma::load_async((v_smem[tic][w]), tma_v, kvsmem_barrier, tile_idx); 
        }
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);
    __syncthreads();

    tma::arrive_and_wait(qsmem_barrier, q_phasebit);
    q_phasebit ^= 1;

    warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f));

    for (auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic ^= 1, toc ^= 1) {
        tma::arrive_and_wait(kvsmem_barrier, kv_phasebit);
        kv_phasebit ^= 1;

        __syncthreads();
        if (warpid == 0) {
            tma::set_bytes(kvsmem_barrier, 2 * NUM_WORKERS_KV * k_smem[0][0].num_elements * sizeof(bf16));

            if (kv_idx + 1 < kv_blocks) {
                for (int w = 0; w < NUM_WORKERS_KV; w++) {        
                    int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS_KV) + w; 
                    tma::load_async((k_smem[toc][w]), tma_k, kvsmem_barrier, tile_idx); 
                    tma::load_async((v_smem[toc][w]), tma_v, kvsmem_barrier, tile_idx);
                }
            }
        }

        warpgroup::mma_fence(att_block);
        warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[tic][0]);
        warpgroup::mma_commit_group();

        copy(norm_vec_last, norm_vec);
        copy(max_vec_last,  max_vec);

        warpgroup::mma_async_wait();

        row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
        sub_row(att_block, att_block, max_vec);
        exp(att_block, att_block);

        sub(max_vec_last, max_vec_last, max_vec);
        exp(max_vec_last, max_vec_last);
        mul(norm_vec, norm_vec, max_vec_last);

        row_sum(norm_vec, att_block, norm_vec); // accumulate onto the norm_vec
        div_row(att_block, att_block, norm_vec);

        mul(norm_vec_last, norm_vec_last, max_vec_last);
        div(norm_vec_last, norm_vec_last, norm_vec);

        copy(att_block_mma, att_block); // convert to bf16 for mma
        mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma'ing onto it

        warpgroup::mma_fence(o_prev);
        warpgroup::mma_AB(o_prev, att_block_mma, v_smem[tic][0]);
        warpgroup::mma_commit_group();
    }

    auto (*o_smem) = reinterpret_cast<st_bf<qo_height, tile_width, layout_o>(*)>(q_smem); // reuse q memory
    warpgroup::store(o_smem[warpgroupid], o_prev); 
    __syncthreads();
    
    if (warpid % 4 == 0) { // store o
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid;
        tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    tma::store_async_wait();
}

// #include "harness_h100_fwd.impl"

#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>
void fwd_attend_ker_tk(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {
    std::cout << "Entered forward attention kernel handler" << std::endl;

    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);

    auto batch   = q.size(0);
    auto heads   = q.size(1);
    auto threads = NUM_WORKERS * kittens::WARP_THREADS;
    auto n       = q.size(2);
    auto d       = q.size(3);

    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "k must be bf16");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "v must be bf16");
    TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "o must be bf16");

    // b, h, n -> 16, 16, 4096
    // need to hard code TMA descriptor below

    TORCH_CHECK(batch == 16, "Batch size is TMA hard coded to 16");
    TORCH_CHECK(heads == 16, "Number of heads is TMA hard coded to 16");
    TORCH_CHECK(n == 4096, "Seq len is TMA hard coded to 4096");
    TORCH_CHECK(d == 64, "Head dim is TMA hard coded to 64"); 

    TORCH_CHECK(n % (NUM_WORKERS * kittens::TILE_DIM) == 0, "The number of elements should be divisible the number of workers times the tile dimension");

    // convert to bf16
    c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();

    bf16* q_bf = reinterpret_cast<bf16*>(q_ptr);
    bf16* k_bf = reinterpret_cast<bf16*>(k_ptr);
    bf16* v_bf = reinterpret_cast<bf16*>(v_ptr);
    bf16* o_bf = reinterpret_cast<bf16*>(o_ptr);

    CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<qo_height, tile_width, layout_q>, (16 * 16 * 4096)/(qo_height * 16)>(q_bf);
    CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<kv_height, tile_width, layout_k>, (16 * 16 * 4096)/(kv_height * 16)>(k_bf);
    CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<kv_height, tile_width, layout_v>, (16 * 16 * 4096)/(kv_height * 16)>(v_bf);
    CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<qo_height, tile_width, layout_o>, (16 * 16 * 4096)/(qo_height * 16)>(o_bf);

    std::cout << "Check and casts" << std::endl;
    unsigned long mem_size = 227000;
    cudaFuncSetAttribute(fwd_attend_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    std::cout << "Set dynamic shared memory" << std::endl;

    dim3 grid(n/(NUM_WORKERS*kittens::TILE_DIM), batch*heads, 1);
    fwd_attend_ker<<<grid, threads, mem_size>>>(n, tma_q_d, tma_k_d, tma_v_d, tma_o_d);

    std::cout << "Launched kernel" << std::endl;
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "Exiting forward attention kernel handler" << std::endl;
}