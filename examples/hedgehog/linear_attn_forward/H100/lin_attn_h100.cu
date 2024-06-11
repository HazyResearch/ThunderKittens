// #define TORCH_COMPILE // defined by default for PyTorch bindings - to use cpp harness, comment this out

#ifdef TORCH_COMPILE
#include "src/kittens.cuh"
#else
#include "../../../../src/kittens.cuh"
#endif

#include <cooperative_groups.h>
#include <cuda/pipeline>

#define NUM_WORKERS (4)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define NUM_WARPGROUPS (NUM_WORKERS/kittens::WARPGROUP_WARPS)

using namespace kittens;

template<ducks::st::all ST>
__device__ inline void cumulative_add(ST &dst, const ST &inc) {
    // first do a reduction for each col
    constexpr int responsible_elements = (ST::cols + kittens::WARP_THREADS - 1) / kittens::WARP_THREADS;
    float acc[responsible_elements];

    // acc equal to the last row of dst
    for (auto i = 0; i < responsible_elements; i++) {
        auto col = (kittens::laneid() + (i * kittens::WARP_THREADS));
        if (col < dst.cols) {
            acc[i] = __bfloat162float(dst.data[(dst.rows - 1) * dst.cols + col]);
            __syncwarp();
            for (auto row = 0; row < dst.rows; row++) {
                acc[i] += __bfloat162float(inc.data[row * inc.cols + col]);
                dst.data[row * dst.cols + col] = __float2bfloat16(acc[i]);
            }
        }
        __syncwarp();
    }
}

template<ducks::sv::all SV, ducks::st::all ST, int N>
__device__ inline void get_last_row(SV &dst, const ST (&src)[N]) {
    // src is of shape (H x W)[N]
    // dst is of shape 1 X (W * N)
    static_assert(SV::length == ST::cols * N, "Destination vector must have W * N columns");

    for (auto i = 0; i < N; i++) {
        for (auto j = 0; j < ST::cols; j++) {
            dst.data[i * ST::cols + j] = src[i].data[(ST::rows - 1) * ST::cols + j];
        }
    }
}

#define ATTN_D 128
#define ATTN_F 256

// __global__ __launch_bounds__(NUM_THREADS, 1)
// void hedgehog_linear_attention_exp(int n, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, 
//                                                 CUtensorMap* tma_o,       CUtensorMap* tma_kv)
// {
//     extern __shared__ int __shm[]; // this is the CUDA shared memory
//     tma_swizzle_allocator al((int*)&__shm[0]);

//     st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>    (&q_smem)   [2][4] = al.allocate<st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>,  2, 4>(); // 32k
//     st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave> (&k_smem)   [2][4] = al.allocate<st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave>,  2, 4>(); // 32k 
//     st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave> (&v_smem)   [2][1] = al.allocate<st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave>, 2, 1>(); // 16k
//     st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave> (&kv_smem)         = al.allocate<st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave>   >();
//     st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave> (&k_c_smem) [4]    = al.allocate<st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave>,  4>(); // 32k

//     int warpid      = kittens::warpid();

//     int tic = 0, toc = 1; 
//     __shared__ uint64_t qkv_barrier; 

//     int blocks = n / (kittens::TILE_DIM * 4); 

//     if (warpid == 0) {
//         tma::init_barrier(qkv_barrier, 1);
//         tma::set_bytes(qkv_barrier, 
//             size_bytes<st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>>*2 + 
//             size_bytes<st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave>>*2 + 
//             size_bytes<st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave>>
//         );

//         int tile_idx = (blockIdx.x * blocks) + 0; 
//         for (int i = 0; i < 2; i++) {
//             tma::load_async(q_smem[tic][i], tma_q, qkv_barrier, tile_idx, i); 
//             tma::load_async(k_smem[tic][i], tma_k, qkv_barrier, tile_idx, i); 
//         }
//         tma::load_async(v_smem[tic][0], tma_v, qkv_barrier, tile_idx);
//     }

//     rt_fl<1, 8> local_kv[4];

//     for (int rt = 0; rt < 4; rt++) { zero(local_kv[rt]); } 
//     for (int rt = 0; rt < 4; rt++) { warpgroup::zero(k_c_smem[rt]); }

//     for (int block = 0; block < blocks; block++, tic^=1, toc^=1) {
//         rt_fl<1, 4>          local_attn; 
//         rt_bf<1, 4>          local_attn_bf; 

//         rt_bf<1, 8>          kv_bf[4];

//         rt_fl<1, 4>          q_reg; 
//         rt_fl<1, 4>          k_c_reg;

//         rt_fl<1, 4>          q_fm_reg; 
//         rt_fl<1, 4>          k_fm_reg; 

//         col_vec<rt_fl<1, 4>> max_fm_vec; 
//         col_vec<rt_fl<1, 4>> min_fm_vec;

//         rt_fl<1, 8>          local_o; 
//         col_vec<rt_fl<1, 4>> den_vec;

//         neg_infty(max_fm_vec); 
//         pos_infty(min_fm_vec);
//         zero(den_vec);
        
//         tma::arrive_and_wait(qkv_barrier, tic); 
//         __syncthreads();

//         if (warpid == 0 && block < blocks - 1) {
//             tma::set_bytes(qkv_barrier, 
//                 size_bytes<st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>>*2 + 
//                 size_bytes<st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave>>*2 + 
//                 size_bytes<st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave>>
//             );

//             int tile_idx = (blockIdx.x * blocks) + block + 1;
            
//             for (int i = 0; i < 2; i++) {
//                 tma::load_async(q_smem[toc][i], tma_q, qkv_barrier, tile_idx, i); 
//                 tma::load_async(k_smem[toc][i], tma_k, qkv_barrier, tile_idx, i); 
//             }
//             tma::load_async(v_smem[toc][0], tma_v, qkv_barrier, tile_idx);
//         }

//         // ******* apply feature map ******** // 
//         // do q first
        
//         for (int rt = 0; rt < 2; rt++) {
//             warpgroup::load(q_fm_reg, q_smem[tic][rt]);

//             row_max(max_fm_vec, q_fm_reg, max_fm_vec);
//             row_min(min_fm_vec, q_fm_reg, min_fm_vec);

//             warpgroup::mul(q_smem[tic][rt + 2], q_smem[tic][rt], __float2bfloat16(-1.0f));
//         }

//         sub_row(q_fm_reg, q_fm_reg, max_fm_vec);
//         exp(q_fm_reg, q_fm_reg);
//         warpgroup::store(q_smem[tic][1], q_fm_reg);

//         // now do 0
//         warpgroup::load(q_fm_reg, q_smem[tic][0]);
//         sub_row(q_fm_reg, q_fm_reg, max_fm_vec);
//         exp(q_fm_reg, q_fm_reg);
//         warpgroup::store(q_smem[tic][0], q_fm_reg);

        
//         for (int rt = 2; rt < 4; rt++) {
//             warpgroup::load(q_fm_reg, q_smem[tic][rt]);
//             add_row(q_fm_reg, q_fm_reg, min_fm_vec);
//             exp(q_fm_reg, q_fm_reg);
//             warpgroup::store(q_smem[tic][rt], q_fm_reg);
//         }
        
//         neg_infty(max_fm_vec);
//         pos_infty(min_fm_vec);
//         __syncthreads();

//         // now do exactly the same for k
//         for (int rt = 0; rt < 2; rt++) {
//             warpgroup::load(k_fm_reg, k_smem[tic][rt]);
//             row_max(max_fm_vec, k_fm_reg, max_fm_vec);
//             row_min(min_fm_vec, k_fm_reg, min_fm_vec);

//             warpgroup::mul(k_smem[tic][rt + 2], k_smem[tic][rt], __float2bfloat16(-1.0f));
//         }

//         sub_row(k_fm_reg, k_fm_reg, max_fm_vec);
//         exp(k_fm_reg, k_fm_reg);
//         warpgroup::store(k_smem[tic][1], k_fm_reg);

//         // now do 0
//         warpgroup::load(k_fm_reg, k_smem[tic][0]);
//         sub_row(k_fm_reg, k_fm_reg, max_fm_vec);
//         exp(k_fm_reg, k_fm_reg);
//         warpgroup::store(k_smem[tic][0], k_fm_reg);

//         for (int rt = 2; rt < 4; rt++) {
//             warpgroup::load(k_fm_reg, k_smem[tic][rt]);
//             add_row(k_fm_reg, k_fm_reg, min_fm_vec);
//             exp(k_fm_reg, k_fm_reg);
//             warpgroup::store(k_smem[tic][rt], k_fm_reg);
//         }
//         __syncthreads();

//         // ******* feature map done ******** //  
//         zero(local_attn); 
//         for (int j = 0; j < 4; j++) {
//             warpgroup::mma_fence(local_attn); 
//             warpgroup::mma_ABt(local_attn, q_smem[tic][j], k_smem[tic][j]); 
//             warpgroup::mma_commit_group(); 
//             warpgroup::mma_async_wait();
//         }

//         // now make causal
//         for (int j = 0; j < 4; j++) {
//             auto &attn_subtile = reinterpret_cast<rt_fl_1x1<>&>(local_attn.tiles[0][j]);
//             if (j > warpid) zero(attn_subtile);
//             else if (j == warpid) make_causal(attn_subtile, attn_subtile, 0.0f);
//         }
//         __syncthreads();

//         copy(local_attn_bf, local_attn); 
//         warpgroup::mma_fence(local_o); 
//         warpgroup::mm_AB(local_o, local_attn_bf, v_smem[tic][0]); 
//         warpgroup::mma_commit_group(); 
//         warpgroup::mma_async_wait();

//         for (auto rt = 0; rt < 4; rt++) {
//             copy(kv_bf[rt], local_kv[rt]); 
//             warpgroup::store(kv_smem, kv_bf[rt]); 
//             __syncthreads();

//             warpgroup::mma_fence(local_o); 
//             warpgroup::mma_AB(local_o, q_smem[tic][rt], kv_smem); 
//             warpgroup::mma_commit_group();

//             warpgroup::mma_fence(local_kv[rt]); 
//             warpgroup::mma_AtB(local_kv[rt], k_smem[tic][rt], v_smem[tic][0]); 
//             warpgroup::mma_commit_group(); 
//             warpgroup::mma_async_wait();
//         }

//         __syncthreads();
//         cumulative_add(k_c_smem[warpid], k_smem[tic][warpid]);
//         __syncthreads(); 

//         if (block > 0) {
//             zero(den_vec);
//             add(den_vec, den_vec, 1e-6f);
//             for (auto rt = 0; rt < 4; rt++) {
//                 auto &k_c_2_smem = reinterpret_cast<st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>&>(k_c_smem[rt]);
            
//                 warpgroup::load(q_reg, q_smem[tic][rt]);
//                 warpgroup::load(k_c_reg, k_c_2_smem);

//                 mul(q_reg, q_reg, k_c_reg);
//                 row_sum(den_vec, q_reg, den_vec);
//             }
//         }
//         else {
//             row_sum(den_vec, local_attn, den_vec);
//         }
//         div_row(local_o, local_o, den_vec);

//         auto &o_smem = reinterpret_cast<st_bf<4, 8, kittens::ducks::st_layout::wgmma_swizzle>&>(v_smem[tic][0]);
//         copy(kv_bf[0], local_o);
//         warpgroup::store(o_smem, kv_bf[0]); 
//         __syncthreads();

//         if (warpid == 0) {
//             int sidx = (blockIdx.x * blocks) + block; 
//             tma::store_async(tma_o, o_smem, sidx); 
//             tma::store_commit_group(); 
//         }
//         tma::store_async_wait();
//     }

//     for (int rt = 0; rt < 4; rt++) {
//         auto &kv_smem_2 = reinterpret_cast<st_bf<4, 8, kittens::ducks::st_layout::wgmma_swizzle>&>(kv_smem);
//         warpgroup::store(kv_smem_2, local_kv[rt]); 
//         __syncthreads();

//         if (warpid == 0) {
//             int tile_idx = (blockIdx.x * 4) + rt; 
//             tma::store_async(tma_kv, kv_smem_2, tile_idx); 
//             tma::store_commit_group(); 
//         }
//         tma::store_async_wait();
//     }
// }

__global__ __launch_bounds__(NUM_THREADS, 1)
void hedgehog_linear_attention_smd(int n, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, 
                                                CUtensorMap* tma_o)
{
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>    (&q_smem)   [2][4] = al.allocate<st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>,    2, 4>(); // 32k
    st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave> (&k_smem)   [2][4] = al.allocate<st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave>, 2, 4>(); // 32k 
    st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave> (&v_smem)   [2][1] = al.allocate<st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave>, 2, 1>(); // 16k
    st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave> (&kv_smem)         = al.allocate<st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave>      >();
    
    st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave> (&k_c_smem) [4]    = al.allocate<st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave>, 4   >(); // 32k
    row_vec<st_bf<4, 4*4>>                                   (&k_state)         = al.allocate<row_vec<st_bf<4, 4*4>>>();

    int warpid      = kittens::warpid();

    int tic = 0, toc = 1; 
    __shared__ uint64_t qkv_barrier; 

    int blocks = n / (kittens::TILE_DIM * 4); 

    if (warpid == 0) {
        tma::init_barrier(qkv_barrier, 1);
        tma::set_bytes(qkv_barrier, 
            size_bytes<st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>>*2 + 
            size_bytes<st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave>>*2 + 
            size_bytes<st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave>>
        );

        int tile_idx = (blockIdx.x * blocks) + 0; 
        for (int i = 0; i < 2; i++) {
            tma::load_async(q_smem[tic][i], tma_q, qkv_barrier, tile_idx, i); 
            tma::load_async(k_smem[tic][i], tma_k, qkv_barrier, tile_idx, i); 
        }
        tma::load_async(v_smem[tic][0], tma_v, qkv_barrier, tile_idx);
    }

    rt_fl<1, 8> local_kv[4];

    for (int rt = 0; rt < 4; rt++) { zero(local_kv[rt]); } 
    for (int rt = 0; rt < 4; rt++) { warpgroup::zero(k_c_smem[rt]); }

    for (int block = 0; block < blocks; block++, tic^=1, toc^=1) {
        rt_fl<1, 4>          local_attn; 
        rt_bf<1, 4>          local_attn_bf; 

        rt_bf<1, 8>          kv_bf[4];

        rt_fl<1, 4>          q_reg; 
        rt_fl<1, 4>          k_c_reg;

        rt_fl<1, 4>          q_fm_reg[2]; 
        rt_fl<1, 4>          k_fm_reg[2];

        col_vec<rt_fl<1, 4>> max_fm_vec; 
        col_vec<rt_fl<1, 4>> min_fm_vec;
        col_vec<rt_fl<1, 4>> sum_fm_vec;

        rt_fl<1, 8>          local_o; 
        col_vec<rt_fl<1, 4>> den_vec;

        neg_infty(max_fm_vec); 
        pos_infty(min_fm_vec);
        zero(den_vec);

        tma::arrive_and_wait(qkv_barrier, tic); 
        __syncthreads();

        if (warpid == 0 && block < blocks - 1) {
            tma::set_bytes(qkv_barrier, 
                size_bytes<st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>>*2 + 
                size_bytes<st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave>>*2 + 
                size_bytes<st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave>>
            );

            int tile_idx = (blockIdx.x * blocks) + block + 1;

            for (int i = 0; i < 2; i++) {
                tma::load_async(q_smem[toc][i], tma_q, qkv_barrier, tile_idx, i); 
                tma::load_async(k_smem[toc][i], tma_k, qkv_barrier, tile_idx, i); 
            }
            tma::load_async(v_smem[toc][0], tma_v, qkv_barrier, tile_idx);
        }

        // ******* apply feature map ******** // 
        
        // do q first
        warpgroup::mul(q_smem[tic][2], q_smem[tic][0], __float2bfloat16(-1.0f));
        warpgroup::mul(q_smem[tic][3], q_smem[tic][1], __float2bfloat16(-1.0f));
        __syncthreads();
        
        #pragma unroll
        for (int rt = 0; rt < 2; rt++) {
            warpgroup::load(q_fm_reg[rt], q_smem[tic][rt]);
            row_max(max_fm_vec, q_fm_reg[rt], max_fm_vec);
            row_min(min_fm_vec, q_fm_reg[rt], min_fm_vec);
        }

        zero(sum_fm_vec);

        #pragma unroll
        for (int rt = 0; rt < 2; rt++) {
            sub_row(q_fm_reg[rt], q_fm_reg[rt], max_fm_vec);
            exp(q_fm_reg[rt], q_fm_reg[rt]);
            row_sum(sum_fm_vec, q_fm_reg[rt], sum_fm_vec);
        }

        #pragma unroll
        for (int rt = 0; rt < 2; rt++) {
            div_row(q_fm_reg[rt], q_fm_reg[rt], sum_fm_vec);
            warpgroup::store(q_smem[tic][rt], q_fm_reg[rt]);
        }
        __syncthreads();

        zero(sum_fm_vec);

        #pragma unroll
        for (int rt = 2; rt < 4; rt++) {
            warpgroup::load(q_fm_reg[rt - 2], q_smem[tic][rt]);
            add_row(q_fm_reg[rt - 2], q_fm_reg[rt - 2], min_fm_vec);
            exp(q_fm_reg[rt - 2], q_fm_reg[rt - 2]);
            row_sum(sum_fm_vec, q_fm_reg[rt - 2], sum_fm_vec);
        }

        #pragma unroll
        for (int rt = 2; rt < 4; rt++) {
            div_row(q_fm_reg[rt - 2], q_fm_reg[rt - 2], sum_fm_vec);
            warpgroup::store(q_smem[tic][rt], q_fm_reg[rt - 2]);
        }
        __syncthreads();

        // now do exactly the same for k
        neg_infty(max_fm_vec);
        pos_infty(min_fm_vec);

        // do q first
        warpgroup::mul(k_smem[tic][2], k_smem[tic][0], __float2bfloat16(-1.0f));
        warpgroup::mul(k_smem[tic][3], k_smem[tic][1], __float2bfloat16(-1.0f));
        __syncthreads();

        #pragma unroll
        for (int rt = 0; rt < 2; rt++) {
            warpgroup::load(k_fm_reg[rt], k_smem[tic][rt]);
            row_max(max_fm_vec, k_fm_reg[rt], max_fm_vec);
            row_min(min_fm_vec, k_fm_reg[rt], min_fm_vec);
        }

        zero(sum_fm_vec);

        #pragma unroll
        for (int rt = 0; rt < 2; rt++) {
            sub_row(k_fm_reg[rt], k_fm_reg[rt], max_fm_vec);
            exp(k_fm_reg[rt], k_fm_reg[rt]);
            row_sum(sum_fm_vec, k_fm_reg[rt], sum_fm_vec);
        }

        #pragma unroll
        for (int rt = 0; rt < 2; rt++) {
            div_row(k_fm_reg[rt], k_fm_reg[rt], sum_fm_vec);
            warpgroup::store(k_smem[tic][rt], k_fm_reg[rt]);
        }
        __syncthreads();

        zero(sum_fm_vec);

        #pragma unroll
        for (int rt = 2; rt < 4; rt++) {
            warpgroup::load(k_fm_reg[rt - 2], k_smem[tic][rt]);
            add_row(k_fm_reg[rt - 2], k_fm_reg[rt - 2], min_fm_vec);
            exp(k_fm_reg[rt - 2], k_fm_reg[rt - 2]);
            row_sum(sum_fm_vec, k_fm_reg[rt - 2], sum_fm_vec);
        }

        #pragma unroll
        for (int rt = 2; rt < 4; rt++) {
            div_row(k_fm_reg[rt - 2], k_fm_reg[rt - 2], sum_fm_vec);
            warpgroup::store(k_smem[tic][rt], k_fm_reg[rt - 2]);
        }
        __syncthreads();
        // ******* feature map done ******** // 
        zero(local_attn); 
        for (int j = 0; j < 4; j++) {
            warpgroup::mma_fence(local_attn); 
            warpgroup::mma_ABt(local_attn, q_smem[tic][j], k_smem[tic][j]); 
            warpgroup::mma_commit_group(); 
            warpgroup::mma_async_wait();
        }

        // now make causal
        for (int j = 0; j < 4; j++) {
            auto &attn_subtile = reinterpret_cast<rt_fl_1x1<>&>(local_attn.tiles[0][j]);
            if (j > warpid) zero(attn_subtile);
            else if (j == warpid) make_causal(attn_subtile, attn_subtile, 0.0f);
        }
        __syncthreads();

        copy(local_attn_bf, local_attn); 
        warpgroup::mma_fence(local_o); 
        warpgroup::mm_AB(local_o, local_attn_bf, v_smem[tic][0]); 
        warpgroup::mma_commit_group(); 
        warpgroup::mma_async_wait();

        #pragma unroll
        for (auto rt = 0; rt < 4; rt++) {
            warpgroup::store(kv_smem, local_kv[rt]); 
            __syncthreads();

            warpgroup::mma_fence(local_o); 
            warpgroup::mma_AB(local_o, q_smem[tic][rt], kv_smem); 
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            warpgroup::mma_fence(local_kv[rt]); 
            warpgroup::mma_AtB(local_kv[rt], k_smem[tic][rt], v_smem[tic][0]); 
            warpgroup::mma_commit_group(); 
            warpgroup::mma_async_wait();
        }

        __syncthreads();
        cumulative_add(k_c_smem[warpid], k_smem[tic][warpid]);
        __syncthreads();

        if (block > 0) {
            zero(den_vec);
            add(den_vec, den_vec, 1e-6f);
            for (auto rt = 0; rt < 4; rt++) {
                auto &k_c_2_smem = reinterpret_cast<st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>&>(k_c_smem[rt]);
            
                warpgroup::load(q_reg, q_smem[tic][rt]);
                warpgroup::load(k_c_reg, k_c_2_smem);

                mul(q_reg, q_reg, k_c_reg);
                row_sum(den_vec, q_reg, den_vec);
            }
        }
        else {
            row_sum(den_vec, local_attn, den_vec);
        }
        div_row(local_o, local_o, den_vec);

        auto &o_smem = reinterpret_cast<st_bf<4, 8, kittens::ducks::st_layout::wgmma_swizzle>&>(v_smem[tic][0]);
        warpgroup::store(o_smem, local_o); 
        __syncthreads(); 

        if (warpid == 0) {
            tma::store_async(tma_o, o_smem, (blockIdx.x * blocks) + block); 
            tma::store_commit_group(); 
        }
        tma::store_async_wait(); 
    }

    // __syncthreads(); 
    // get_last_row(k_state, k_c_smem);
    // __syncthreads(); 

    // if (warpid == 0) {
    //     tma::store_async(tma_ks, k_state, blockIdx.x); 
    //     tma::store_commit_group(); 
    // }

    // for (int rt = 0; rt < 4; rt++) {
    //     auto &kv_smem_2 = reinterpret_cast<st_bf<4, 8, kittens::ducks::st_layout::wgmma_swizzle>&>(kv_smem);
    //     warpgroup::store(kv_smem_2, local_kv[rt]); 
    //     __syncthreads();

    //     if (warpid == 0) {
    //         int tile_idx = (blockIdx.x * 4) + rt; 
    //         tma::store_async(tma_kv, kv_smem_2, tile_idx); 
    //         tma::store_commit_group(); 
    //     }
    //     tma::store_async_wait();
    // }
}

#ifdef TORCH_COMPILE
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>

// void hh_lin_tk_exp(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor kv) {

//     CHECK_INPUT(q); 
//     CHECK_INPUT(k); 
//     CHECK_INPUT(v); 
//     CHECK_INPUT(kv); 
//     CHECK_INPUT(o); 

//     auto batch = q.size(0); 
//     auto heads = q.size(1); 
//     auto N  = q.size(2); 

//     // N must be >= 64 and a multiple of 64
//     TORCH_CHECK(N >= 64, "N must be >= 64");
//     TORCH_CHECK(N % 64 == 0, "N must be a multiple of 64");

//     auto q_d  = q.size(3);
//     auto k_d  = k.size(3);
//     auto v_d  = v.size(3);
//     auto o_d  = o.size(3);

//     // all must be == 128
//     TORCH_CHECK(q_d == k_d, "q and k must have the same dimension");
//     TORCH_CHECK(q_d == v_d, "q and v must have the same dimension");
//     TORCH_CHECK(q_d == o_d, "q and o must have the same dimension");
//     TORCH_CHECK(q_d == 128, "q, k, v must have dimension 128");

//     auto kv_d_1 = kv.size(2);
//     auto kv_d_2 = kv.size(3);

//     // kv must be 256x128
//     TORCH_CHECK(kv_d_1 == 256, "kv must have dimension 256");
//     TORCH_CHECK(kv_d_2 == 128, "kv must have dimension 128");

//     TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "q must be bf16");
//     TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "k must be bf16");
//     TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "v must be bf16");
//     TORCH_CHECK(kv.scalar_type() == c10::ScalarType::BFloat16, "kv must be bf16");
//     TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "o must be bf16");

//     c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *kv_ptr = kv.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();

//     const bf16* d_q = reinterpret_cast<const bf16*>(q_ptr); 
//     const bf16* d_k = reinterpret_cast<const bf16*>(k_ptr);  
//     const bf16* d_v = reinterpret_cast<const bf16*>(v_ptr);  
//     bf16* d_kv_state = reinterpret_cast<bf16*>(kv_ptr);  
//     bf16* d_o = reinterpret_cast<bf16*>(o_ptr);

//     CUtensorMap* tma_q_d  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>>   (d_q,        (batch*heads*N/(16 * 4)),    128/(16 * 4) ); 
//     CUtensorMap* tma_k_d  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave>>(d_k,        (batch*heads*N/(16 * 4)),    128/(16 * 4) );
//     CUtensorMap* tma_v_d  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave>>(d_v,        (batch*heads*N/(16 * 4)),    128/(16 * 8) );
//     CUtensorMap* tma_o_d  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 8, kittens::ducks::st_layout::wgmma_swizzle>>   (d_o,        (batch*heads*N/(16 * 4)),    128/(16 * 8) );
//     CUtensorMap* tma_kv_d = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 8, kittens::ducks::st_layout::wgmma_swizzle>>   (d_kv_state, (batch*heads*256/(16 * 4)),  128/(16 * 8) ); 

//     unsigned long mem_size = kittens::MAX_SHARED_MEMORY;

//     cudaFuncSetAttribute(
//         hedgehog_linear_attention_exp,
//         cudaFuncAttributeMaxDynamicSharedMemorySize,
//         mem_size
//     );

//     dim3 grid(batch*heads, 1, 1);

//     hedgehog_linear_attention_exp<<<grid, 128, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d, tma_kv_d);

//     CHECK_CUDA_ERROR(cudaGetLastError());
// }

void hh_lin_tk_smd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {

    CHECK_INPUT(q); 
    CHECK_INPUT(k); 
    CHECK_INPUT(v); 
    CHECK_INPUT(o); 

    auto batch = q.size(0); 
    auto heads = q.size(1); 
    auto N  = q.size(2); 

    // N must be >= 64 and a multiple of 64
    TORCH_CHECK(N >= 64, "N must be >= 64");
    TORCH_CHECK(N % 64 == 0, "N must be a multiple of 64");

    auto q_d  = q.size(3);
    auto k_d  = k.size(3);
    auto v_d  = v.size(3);
    auto o_d  = o.size(3);

    // all must be == 128
    TORCH_CHECK(q_d == k_d, "q and k must have the same dimension");
    TORCH_CHECK(q_d == v_d, "q and v must have the same dimension");
    TORCH_CHECK(q_d == o_d, "q and o must have the same dimension");
    TORCH_CHECK(q_d == 128, "q, k, v must have dimension 128");

    // auto kv_d_1 = kv.size(2);
    // auto kv_d_2 = kv.size(3);

    // // kv must be 256x128
    // TORCH_CHECK(kv_d_1 == 256, "kv must have dimension 256");
    // TORCH_CHECK(kv_d_2 == 128, "kv must have dimension 128");

    // // k must be 1 x 256
    // TORCH_CHECK(ks.size(2) == 1, "ks must have sequence length 1");
    // TORCH_CHECK(ks.size(3) == 256, "ks must have dimension 256");

    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "k must be bf16");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "v must be bf16");
    // TORCH_CHECK(kv.scalar_type() == c10::ScalarType::BFloat16, "kv must be bf16");
    // TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "ks must be bf16");
    TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "o must be bf16");

    c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
    // c10::BFloat16 *kv_ptr = kv.data_ptr<c10::BFloat16>();
    // c10::BFloat16 *ks_ptr = ks.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();

    const bf16* d_q = reinterpret_cast<const bf16*>(q_ptr); 
    const bf16* d_k = reinterpret_cast<const bf16*>(k_ptr);  
    const bf16* d_v = reinterpret_cast<const bf16*>(v_ptr);  
    // bf16* d_kv_state = reinterpret_cast<bf16*>(kv_ptr);  
    // bf16* d_k_state  = reinterpret_cast<bf16*>(ks_ptr);
    bf16* d_o = reinterpret_cast<bf16*>(o_ptr);

    CUtensorMap* tma_q_d  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, kittens::ducks::st_layout::wgmma_swizzle>>   (d_q,        (batch*heads*N/(16 * 4)),    128/(16 * 4) ); 
    CUtensorMap* tma_k_d  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 4, kittens::ducks::st_layout::wgmma_interleave>>(d_k,        (batch*heads*N/(16 * 4)),    128/(16 * 4) );
    CUtensorMap* tma_v_d  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 8, kittens::ducks::st_layout::wgmma_interleave>>(d_v,        (batch*heads*N/(16 * 4)),    128/(16 * 8) );
    CUtensorMap* tma_o_d  = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 8, kittens::ducks::st_layout::wgmma_swizzle>>   (d_o,        (batch*heads*N/(16 * 4)),    128/(16 * 8) );
    // CUtensorMap* tma_kv_d = tma::allocate_and_create_tensor_map<kittens::st_bf<4, 8, kittens::ducks::st_layout::wgmma_swizzle>>   (d_kv_state, (batch*heads*256/(16 * 4)),  128/(16 * 8) );
    // CUtensorMap* tma_ks_d = tma::allocate_and_create_tensor_map<row_vec<st_bf<4, 4*4>>>                                           (d_k_state,  (batch*heads*  1/(   1  )));  

    unsigned long mem_size = kittens::MAX_SHARED_MEMORY;

    cudaFuncSetAttribute(
        hedgehog_linear_attention_smd,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    dim3 grid(batch*heads, 1, 1);

    hedgehog_linear_attention_smd<<<grid, 128, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d); 

    CHECK_CUDA_ERROR(cudaGetLastError());
}

#else
#include "harness.impl"
#endif