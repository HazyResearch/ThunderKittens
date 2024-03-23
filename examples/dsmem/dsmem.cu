#include <cooperative_groups.h>

#include "../../src/kittens.cuh"

#define ATTN_B 16
#define ATTN_H 16
#define ATTN_N 4096
#define ATTN_D 64

#define NUM_WORKERS 8
#define BLOCK_SIZE (32*NUM_WORKERS)

#define CLUSTER_SIZE 16

using namespace kittens;

using q_layout = st_wgmma_row_0b_layout; // may need to swap for head dim 128
using k_layout = st_wgmma_row_0b_layout;
using v_layout = st_wgmma_col_t_0b_layout;
using o_layout = st_wgmma_row_0b_layout;

template<typename T> __device__ inline void swap(T & a, T & b) { T tmp = a; a = b; b = tmp; }

template<int D, int TH, int B, int N>
struct attn_tma_descriptors {
    // int N, B; // N sequence length, B is batch (incl. heads)
    CUtensorMap *q_desc, *k_desc, *v_desc, *o_desc;
    attn_tma_descriptors(bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o) { // }: N(_N), B(_B) {
        CUtensorMap q_desc_host = {}; 
        CUtensorMap k_desc_host = {};
        CUtensorMap v_desc_host = {};
        CUtensorMap o_desc_host = {};

        constexpr int TW = D/16;
        tma::create_tensor_map<st_bf<TH,TW,q_layout>, (B*N)/(TH * 16)>(&q_desc_host, d_q); 
        tma::create_tensor_map<st_bf<TH,TW,k_layout>, (B*N)/(TH * 16)>(&k_desc_host, d_k);
        tma::create_tensor_map<st_bf<TH,TW,v_layout>, (B*N)/(TH * 16)>(&v_desc_host, d_v);
        tma::create_tensor_map<st_bf<TH,TW,o_layout>, (B*N)/(TH * 16)>(&o_desc_host, d_o);

        cudaMalloc(&q_desc, sizeof(CUtensorMap));
        cudaMalloc(&k_desc, sizeof(CUtensorMap));
        cudaMalloc(&v_desc, sizeof(CUtensorMap));
        cudaMalloc(&o_desc, sizeof(CUtensorMap));

        cudaMemcpy(q_desc, &q_desc_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
        cudaMemcpy(k_desc, &k_desc_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
        cudaMemcpy(v_desc, &v_desc_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
        cudaMemcpy(o_desc, &o_desc_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    }
    ~attn_tma_descriptors() {
        cudaFree(q_desc);
        cudaFree(k_desc);
        cudaFree(v_desc);
        cudaFree(o_desc);
    }
};

template<int D, int TH, int B, int N>
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) 
attend_ker(CUtensorMap *q_desc, CUtensorMap *k_desc, CUtensorMap *v_desc, CUtensorMap *o_desc) {
            // const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__) {
    
                // Unpack args
    // const int N = desc.N;
    auto warpid        = threadIdx.x / 32;

    static_assert(D % 16 == 0, "head dim must be a multiple of 16!");
    constexpr int TW = D/16;
    static_assert(6*2*TW*TH*256*NUM_WORKERS <= 227000, "dimensions chosen will not fit in shared memory");
    
    namespace cg = cooperative_groups;

    auto grid = cg::this_grid();
    auto cluster = cg::this_cluster();
    constexpr int cluster_size = CLUSTER_SIZE; // cluster.num_blocks(); but i want this at compile time
    int block_idx    = cluster.block_rank();
    int cluster_idx  = grid.cluster_rank();

    auto block = cg::this_thread_block();

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al = shared_allocator::create_allocator((int*)&__shm[0]); 

    // layout:
    // index 0: which part of the cache is this? (0,1) are used as a tic-toc for message passing, 2 is async load.
    // index 1: which worker is responsible
    st_bf<TH,TW,q_layout> (&q_smem)[1][NUM_WORKERS] = al.allocate<st_bf<TH,TW,q_layout>, 1, NUM_WORKERS>();
    st_bf<TH,TW,k_layout> (&k_smem)[2][NUM_WORKERS] = al.allocate<st_bf<TH,TW,k_layout>, 2, NUM_WORKERS>();
    st_bf<TH,TW,v_layout> (&v_smem)[2][NUM_WORKERS] = al.allocate<st_bf<TH,TW,v_layout>, 2, NUM_WORKERS>();

    rt_bf<TH,TW> q_reg;
    rt_fl<TH,TH> att_block;
    rt_bf<TH,TH> att_block_mma;
    rt_fl<TH,TW> o_prev;
    typename rt_fl<TH,TH>::col_vec max_vec_last, max_vec;
    typename rt_fl<TH,TH>::col_vec norm_vec_last, norm_vec;

    constexpr int rows_per_block       = q_reg.rows * NUM_WORKERS;
    constexpr int rows_per_cluster     = rows_per_block * cluster_size;
    constexpr int elements_per_block   = rows_per_block * ATTN_D;
    constexpr int elements_per_cluster = rows_per_cluster * ATTN_D;

    int q_iters = N / elements_per_cluster;
    int kv_iters = N / elements_per_cluster;
    int dsmem_iters = cluster_size;

    // barriers for TMA
    __shared__ uint64_t q_tma_barrier[NUM_WORKERS];
    __shared__ uint64_t k_tma_barrier[NUM_WORKERS];
    __shared__ uint64_t v_tma_barrier[NUM_WORKERS];
    // STORE has its own completion mechanism for the output

    constexpr int tile_bytes = sizeof(bf16) * k_smem[0][0].num_elements;

    tma::init_barrier(q_tma_barrier[warpid], block.size());
    tma::set_barrier_bytes(q_tma_barrier[warpid], tile_bytes);

    tma::init_barrier(k_tma_barrier[warpid], block.size());
    tma::set_barrier_bytes(k_tma_barrier[warpid], tile_bytes);

    tma::init_barrier(v_tma_barrier[warpid], block.size());
    tma::set_barrier_bytes(v_tma_barrier[warpid], tile_bytes);

    block.sync();

    // barriers for DSMEM
    __shared__ uint64_t k_dsmem_barrier[2];
    __shared__ uint64_t v_dsmem_barrier[2];
    
    constexpr int size_bytes = sizeof(bf16) * k_smem[0][0].num_elements * NUM_WORKERS;

    // dsmem works at threadblock level (not using warp)
    for(int i = 0; i < 2; i++) {
        dsmem::init_barrier(k_dsmem_barrier[i], block.size());
        dsmem::set_barrier_bytes(k_dsmem_barrier[i], size_bytes);
        dsmem::init_barrier(v_dsmem_barrier[i], block.size());
        dsmem::set_barrier_bytes(v_dsmem_barrier[i], size_bytes);
    }

    block.sync();

    int global_warp_idx = (cluster_idx * cluster_size + block_idx) * NUM_WORKERS + warpid;

    constexpr int kPhaseBit_dsmem_kv = 1;
    constexpr int kPhaseBit_tma_q = 1;
    constexpr int kPhaseBit_tma_k = 1;
    constexpr int kPhaseBit_tma_v = 1;
    constexpr int kPhaseBit_tma_o = 1;

    int tic = 0, toc = 1; int async=2;

    // load(q_reg, __q__ + global_warp_idx*q_reg.num_elements, D);
    tma::load_async(q_smem[0][warpid], q_desc, global_warp_idx, q_tma_barrier[warpid]);
    tma::arrive_wait(q_tma_barrier[warpid], kPhaseBit_tma_q);
    cluster.sync();
    load(q_reg, q_smem[0][warpid]);

    if constexpr (D == 64) {
        mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment head 64
    }
    else if constexpr (D == 128) {
        mul(q_reg, q_reg, __float2bfloat16(0.08838834764831843f)); // temperature adjustment head 128
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);

    // load(k_smem[tic][warpid], __k__ + global_warp_idx*q_reg.num_elements, ATTN_D);
    // load(v_smem[tic][warpid], __v__ + global_warp_idx*q_reg.num_elements, ATTN_D);
    tma::load_async(k_smem[tic][warpid], k_desc, global_warp_idx, k_tma_barrier[warpid]);
    tma::load_async(v_smem[tic][warpid], v_desc, global_warp_idx, v_tma_barrier[warpid]);
    tma::arrive_wait(k_tma_barrier[warpid], kPhaseBit_tma_k);
    tma::arrive_wait(v_tma_barrier[warpid], kPhaseBit_tma_v);

    __syncthreads(); 
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // print out k and v
        printf("k \n");
        for (int w = 0; w < 8; w++) {
            for (int r = 0; r < k_smem[tic][w].rows; r++) {
                for (int c = 0; c < k_smem[tic][w].cols; c++) {
                    printf("%f ", __bfloat162float(k_smem[tic][w].data[r * k_smem[tic][w].cols + c]));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
        printf("v \n");
        for (int w = 0; w < 8; w++) {
            for (int r = 0; r < v_smem[tic][w].rows; r++) {
                for (int c = 0; c < v_smem[tic][w].cols; c++) {
                    printf("%f ", __bfloat162float(v_smem[tic][w].data[r * v_smem[tic][w].cols + c]));
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    __syncthreads(); 


    cluster.sync(); // make sure all the memory has arrived!

    // for(int q_itr = 0; q_itr < q_iters; q_itr++) {

    //     for(int kv_itr = 0; kv_itr < kv_iters; kv_itr++) {
    
            for(auto dsmem_itr = 0; dsmem_itr < dsmem_iters; dsmem_itr++) {

                if(dsmem_itr > 0) {
                    dsmem::distribution_wait(k_dsmem_barrier[tic], kPhaseBit_dsmem_kv);
                    dsmem::distribution_wait(v_dsmem_barrier[tic], kPhaseBit_dsmem_kv);
                }

                if(dsmem_itr+1 < dsmem_iters) {
                    int neighbor_idx = (block_idx+1) % cluster_size; // pass down by 1
                    dsmem::tile_distribute_smem(k_smem[toc][0], k_smem[tic][0], cluster_size, neighbor_idx, size_bytes, k_dsmem_barrier[toc]);
                    dsmem::tile_distribute_smem(v_smem[toc][0], v_smem[tic][0], cluster_size, neighbor_idx, size_bytes, v_dsmem_barrier[toc]);
                }

                for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {

                    warpgroup::fence(att_block); 
                    warpgroup::dot_reset(att_block, q_reg, k_smem[tic][subtile]); 
                    warpgroup::commit_group();

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

                    warpgroup::fence(o_prev); 
                    warpgroup::mma_accum(o_prev, att_block_mma, v_smem[tic][subtile]); 
                    warpgroup::commit_group();

                    warpgroup::mma_async_wait();
                }

                swap(tic, toc);
                
                __syncthreads(); 
                // cluster.sync(); // I would think this is necessary but seems to work without it? Saves a lot of time too.
        //     }
        }
        cluster.sync(); 

        // store(__o__ + global_warp_idx*q_reg.num_elements, o_prev, ATTN_D);
        st_bf<TH,TW,o_layout> &o_smem = reinterpret_cast<st_bf<TH,TW,o_layout>&>(k_smem[toc][warpid]); // use toc memory to load q first.
        store(o_smem, o_prev);
        // __syncthreads();
        tma::store_async(o_desc, o_smem, global_warp_idx);
        tma::commit_group();
        tma::wait_for_store_complete<0>();
        // __syncthreads();

    // }
}

#include "harness.impl"