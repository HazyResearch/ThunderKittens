#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

#define CONSUMER_WARPGROUPS (3) // hardcoded
#define PRODUCER_WARPGROUPS (1) // hardcoded
#define NUM_WARPGROUPS (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS)
#define NUM_WORKERS (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS)

using namespace kittens;
namespace cg = cooperative_groups;

template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = 64/kittens::TILE_DIM;
    constexpr static int qo_height  = 4;
    constexpr static int kv_height  = 12;
};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = 128/kittens::TILE_DIM;
    constexpr static int qo_height  = 4;
    constexpr static int kv_height  = 8;
};

template<int D>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1) __cluster_dims__(2,1,1)
void fwd_attend_ker_dim(int N, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, CUtensorMap* tma_o) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);
    cg::cluster_group cluster = cg::this_cluster();
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS, block_rank = cluster.block_rank();

    using K = fwd_attend_ker_tile_dims<D>;
    st_bf<K::qo_height, K::tile_width> (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<st_bf<K::qo_height, K::tile_width>, CONSUMER_WARPGROUPS>();
    st_bf<K::kv_height, K::tile_width> (&k_smem)[2]                   = al.allocate<st_bf<K::kv_height, K::tile_width>, 2                  >();
    st_bf<K::kv_height, K::tile_width> (&v_smem)[2]                   = al.allocate<st_bf<K::kv_height, K::tile_width>, 2                  >();
    int kv_blocks = N / (K::kv_height*TILE_DIM);

    __shared__ kittens::barrier qsmem_barrier, k_smem_arrived[2], v_smem_arrived[2], compute_done[2];
    if (threadIdx.x == 0) { // initialize barriers and initial loads
        init_barrier(qsmem_barrier, 0, 1); // no threads, one transaction
        for(int j = 0; j < 2; j++) {
            init_barrier(k_smem_arrived[j], 0, 1); // no threads, one transaction
            init_barrier(v_smem_arrived[j], 0, 1); // no threads, one transaction
            init_barrier(compute_done[j], 2*CONSUMER_WARPGROUPS, 0); // all the consumer threads across both blocks, no transactions
        }
        tma::expect_bytes(qsmem_barrier, sizeof(q_smem));
        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) { // issue async loads for Q chunks
            int q_tile_idx = (blockIdx.y * CONSUMER_WARPGROUPS * gridDim.x) + (blockIdx.x * CONSUMER_WARPGROUPS) + wg;
            tma::load_async(q_smem[wg], tma_q, qsmem_barrier, q_tile_idx); 
        }
        int kv_tile_idx = (blockIdx.y * kv_blocks) + 0; 
        tma::expect<typeof(k_smem[0])>(k_smem_arrived[0]);
        tma::expect<typeof(v_smem[0])>(v_smem_arrived[0]);
        tma::load_async(k_smem[0], tma_k, k_smem_arrived[0], kv_tile_idx); 
        tma::load_async(v_smem[0], tma_v, v_smem_arrived[0], kv_tile_idx);
    }
    cluster.sync();

    int tic = 0, toc = 1;
    if(warpgroupid == NUM_WARPGROUPS-1) { // producer warpgroup
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));        
        if(block_rank == 0 && warpid == NUM_WORKERS-4) {
            for (auto kv_idx = 0; kv_idx < kv_blocks-1; kv_idx++, tic=tic^1, toc=toc^1) {
                int kv_tile_idx = (blockIdx.y * kv_blocks) + (kv_idx + 1);
                tma::cluster::expect<typeof(k_smem[0])>(k_smem_arrived[toc], 0);
                tma::cluster::expect<typeof(k_smem[0])>(k_smem_arrived[toc], 1);
                tma::cluster::load_async(k_smem[toc], tma_k, k_smem_arrived[toc], kv_tile_idx, 0, uint16_t(0x0003));
                tma::cluster::expect<typeof(v_smem[0])>(v_smem_arrived[toc], 0);
                tma::cluster::expect<typeof(v_smem[0])>(v_smem_arrived[toc], 1);
                tma::cluster::load_async(v_smem[toc], tma_v, v_smem_arrived[toc], kv_tile_idx, 0, uint16_t(0x0003));
                tma::cluster::wait(compute_done[tic], (kv_idx/2)%2);
            }
        }
        __syncthreads();
    }
    else { // consumer warpgroup
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(160));

        // premultiply by temperature and lg(e)
        wait(qsmem_barrier, 0);
        if constexpr (D == 64) { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f * 1.44269504089f)); }
        else { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.08838834764f * 1.44269504089f)); }
    
        rt_fl<1, K::kv_height> att_block;
        rt_bf<1, K::kv_height> att_block_mma;
        rt_fl<1, K::tile_width> o_reg;
        col_vec<rt_fl<1, K::kv_height>> max_vec_last, max_vec;
        col_vec<rt_fl<1, K::kv_height>> norm_vec_last, norm_vec;
        neg_infty(max_vec); // clear registers for the Q chunk
        zero(norm_vec);
        zero(o_reg);

        for (auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic=tic^1, toc=toc^1) {
        
            tma::cluster::wait(k_smem_arrived[tic], (kv_idx/2)%2); // wait on k memory
            
            warpgroup::mma_fence(att_block);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[tic]);
            warpgroup::mma_commit_group();
            tma::cluster::wait(v_smem_arrived[tic], (kv_idx/2)%2); // wait on v memory, during the matmul
            copy(norm_vec_last, norm_vec);
            copy(max_vec_last,  max_vec);
            warpgroup::mma_async_wait();

            row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
            sub_row(att_block, att_block, max_vec);
            exp2(att_block, att_block);
            sub(max_vec_last, max_vec_last, max_vec);
            exp2(max_vec_last, max_vec_last);
            mul(norm_vec, norm_vec, max_vec_last);
            row_sum(norm_vec, att_block, norm_vec); // accumulate onto the norm_vec
            div_row(att_block, att_block, norm_vec);
            mul(norm_vec_last, norm_vec_last, max_vec_last);
            div(norm_vec_last, norm_vec_last, norm_vec);
            copy(att_block_mma, att_block); // convert to bf16 for mma
            mul_row(o_reg, o_reg, norm_vec_last); // normalize o_prev in advance of mma'ing onto it

            warpgroup::mma_fence(o_reg);
            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[tic]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
            if(warpgroup::laneid() == 0) tma::cluster::arrive(compute_done[tic], 0); // signal to producer that we are ready for more
        }
        auto (*o_smem) = reinterpret_cast<st_bf<K::qo_height, K::tile_width>(*)>(q_smem); // reuse q memory
        warpgroup::store(o_smem[warpgroupid], o_reg); 
        __syncthreads();
        if (warpid % 4 == 0) { // store o
            int tile_idx = (blockIdx.y * CONSUMER_WARPGROUPS * gridDim.x) + (blockIdx.x * CONSUMER_WARPGROUPS) + warpgroupid;
            tma::store_async(tma_o, o_smem[warpgroupid], tile_idx); 
            tma::store_commit_group(); 
        }
        tma::store_async_wait();
    }
    cluster.sync();
}

// #include "common/pyutils/torch_helpers.cuh"
// void attention_inference_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {

//     // get general parameters to check
//     TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q,k,v must have 4 dimensions (B,H,N,D)");
//     auto batch = q.size(0);
//     auto heads = q.size(1);
//     auto N = q.size(2);
//     TORCH_CHECK(N>0 && N%128 == 0, "N must be a multiple of 128");
//     auto D = q.size(3);
//     TORCH_CHECK(D == 128 || D == 64, "Only head dims of 64 or 128 are supported");

//     // check K, V, O dimensions, too.
//     TORCH_CHECK(k.size(0) == batch && k.size(1) == heads && v.size(2) == N && k.size(3) == D, "k must be (B,H,N,128)");
//     TORCH_CHECK(v.size(0) == batch && v.size(1) == heads && v.size(2) == N && v.size(3) == D, "v must be (B,H,N,128)");
//     TORCH_CHECK(o.size(0) == batch && o.size(1) == heads && o.size(2) == N && o.size(3) == D, "o must be (B,H,N,128)");

//     CHECK_INPUT(q);
//     CHECK_INPUT(k);
//     CHECK_INPUT(v);
//     CHECK_INPUT(o);
//     TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
//     TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bf16");
//     TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bf16");
//     TORCH_CHECK(o.scalar_type() == torch::kBFloat16, "o must be bf16");

//     // make sure D = 64 or 128
//     TORCH_CHECK(D == 64 | D == 128, "Currently, only D = 64 or 128 is supported");

//     // convert to bf16
//     c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();

//     const bf16* q_bf = reinterpret_cast<const bf16*>(q_ptr);
//     const bf16* k_bf = reinterpret_cast<const bf16*>(k_ptr);
//     const bf16* v_bf = reinterpret_cast<const bf16*>(v_ptr);
//     bf16* o_bf = reinterpret_cast<bf16*>(o_ptr);

//     auto threads = NUM_WORKERS * kittens::WARP_THREADS;
//     if (D == 64) {

//         CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width, layout_q>>(q_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<64>::qo_height * 16));
//         CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width, layout_k>>(k_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<64>::kv_height * 16));
//         CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width, layout_v>>(v_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<64>::kv_height * 16));
//         CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width, layout_o>>(o_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<64>::qo_height * 16));

//         unsigned long mem_size = 112000;
//         cudaFuncSetAttribute(fwd_attend_ker_dim<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

//         dim3 grid(N/(NUM_WORKERS*kittens::TILE_DIM), batch*heads, 1);

//         fwd_attend_ker_dim<64><<<grid, threads, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d);
//     }
//     else {
//         CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width, layout_q>>(q_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<128>::qo_height * 16));
//         CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width, layout_k>>(k_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<128>::kv_height * 16));
//         CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width, layout_v>>(v_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<128>::kv_height * 16));
//         CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width, layout_o>>(o_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<128>::qo_height * 16));

//         unsigned long mem_size = 112000;
//         cudaFuncSetAttribute(fwd_attend_ker_dim<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

//         dim3 grid(N/(NUM_WORKERS*kittens::TILE_DIM), batch*heads, 1);

//         fwd_attend_ker_dim<128><<<grid, threads, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d);
//     }
    
//     CHECK_CUDA_ERROR(cudaGetLastError());
// }

#include "harness.impl"