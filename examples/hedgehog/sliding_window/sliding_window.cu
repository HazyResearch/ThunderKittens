#include "src/kittens.cuh"

#define NUM_WORKERS (4) // this comes from the fact that we want a 64-long sliding window
using namespace kittens;

#define WINDOW_WIDTH (64)
static_assert(WINDOW_WIDTH%64==0 && WINDOW_WIDTH<=256);
#define WINDOW_TILES ((WINDOW_WIDTH/64)+1) // first 1 is for ensuring sufficient width
#define WINDOW_TILES_MEM (WINDOW_TILES+1) // second 1 is for async

using layout_q = kittens::ducks::st_layout::wgmma_swizzle; 
using layout_k = kittens::ducks::st_layout::wgmma_swizzle; 
using layout_v = kittens::ducks::st_layout::wgmma_interleave; 
using layout_o = kittens::ducks::st_layout::swizzle;

constexpr int qo_height = 4;
constexpr int kv_height = qo_height; // must be for this kernel.

// we need a function to make anticausal for the first chunk of the sliding window.
template<ducks::rt::row_layout RT>
__device__ static inline void make_anticausal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j < i) { // below the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else if(j > i) { // above the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint32_t MASK_X = 0xFF773311, MASK_Y = 0xF7733110; // magic numbers for on-diagonal core matrices
                dst.tiles[i][j].data[1] = packed_val; // below diagonal, zero
                dst.tiles[i][j].data[2] = src.tiles[i][j].data[2]; // above diagonal, copy
                if((MASK_X >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                }
                else {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }
                if((MASK_Y >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                }
                else {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
            }
        }
    }
}

template<int D>
__global__ __launch_bounds__(NUM_WORKERS*kittens::WARP_THREADS, 2)
void sliding_window(int N, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, CUtensorMap* tma_o, float beta) {

    constexpr int D_T = D/kittens::TILE_DIM;

    auto warpid        = kittens::warpid();

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    st_bf<qo_height, D_T, layout_q> (&q_smem)[2]                = al.allocate<st_bf<qo_height, D_T, layout_q>, 2               >();
    st_bf<kv_height, D_T, layout_k> (&k_smem)[WINDOW_TILES_MEM] = al.allocate<st_bf<kv_height, D_T, layout_k>, WINDOW_TILES_MEM>();
    st_bf<kv_height, D_T, layout_v> (&v_smem)[WINDOW_TILES_MEM] = al.allocate<st_bf<kv_height, D_T, layout_v>, WINDOW_TILES_MEM>();
    st_bf<qo_height, D_T, layout_o> (&o_smem)[2]                = al.allocate<st_bf<qo_height, D_T, layout_o>, 2               >();

    rt_fl<1, kv_height> att_block[WINDOW_TILES];
    rt_bf<1, kv_height> att_block_bf[WINDOW_TILES];
    rt_fl<1, D_T> o_reg;
    rt_fl<1, kv_height>::col_vec max_vec, norm_vec;

    int qo_blocks = N / q_smem[0].rows;

    __shared__ uint64_t smem_barrier;
    if (threadIdx.x == 0) {
        tma::init_barrier(smem_barrier, 1);
        tma::set_bytes(smem_barrier, size_bytes<typeof(q_smem[0])> + size_bytes<typeof(k_smem[0])> + size_bytes<typeof(v_smem[0])>);
        int tile_idx = blockIdx.x * qo_blocks;
        tma::load_async((q_smem[0]), tma_q, smem_barrier, tile_idx); // initial load is into what will become "tic" on the first iter.
        tma::load_async((k_smem[WINDOW_TILES-1]), tma_k, smem_barrier, tile_idx); // initial load is into what will become "last block" on the first iter.
        tma::load_async((v_smem[WINDOW_TILES-1]), tma_v, smem_barrier, tile_idx);
    }
    
    int tic = 0, toc = 1;

    int start_block = 0, last_block = WINDOW_TILES-1, load_block = WINDOW_TILES_MEM-1;
    for(auto qo_blk = 0; qo_blk < qo_blocks; qo_blk++, start_block=(start_block+1)%WINDOW_TILES_MEM, last_block=(last_block+1)%WINDOW_TILES_MEM, load_block=(load_block+1)%WINDOW_TILES_MEM, tic^=1, toc^=1) {

        tma::arrive_and_wait(smem_barrier, tic);

        __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk

        // load the curent k, v blocks into load_block. If qo_blk > 0, then the previous tiles stick around.
        if (warpid == 0) {
            if (qo_blk + 1 < qo_blocks) {
                tma::set_bytes(smem_barrier, size_bytes<typeof(q_smem[0])> + size_bytes<typeof(k_smem[0])> + size_bytes<typeof(v_smem[0])>);
                int tile_idx = blockIdx.x * qo_blocks + qo_blk+1;
                tma::load_async((q_smem[toc]), tma_q, smem_barrier, tile_idx);
                tma::load_async((k_smem[load_block]), tma_k, smem_barrier, tile_idx);
                tma::load_async((v_smem[load_block]), tma_v, smem_barrier, tile_idx);
            }
        }

        if constexpr (D == 64) { warpgroup::mul(q_smem[tic], q_smem[tic], __float2bfloat16(0.125f)); }
        else { warpgroup::mul(q_smem[tic], q_smem[tic], __float2bfloat16(0.08838834764f)); }

        neg_infty(max_vec); // zero registers for the Q chunk
        zero(norm_vec);
        zero(o_reg);

        __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

        for(int subtile = 0; subtile < WINDOW_TILES; subtile++) {
            if (qo_blk + subtile >= WINDOW_TILES-1) { // ensure tile has been loaded by now.

                warpgroup::mma_fence(att_block[subtile]);
                warpgroup::mm_ABt(att_block[subtile], q_smem[tic], k_smem[(start_block+subtile)%WINDOW_TILES_MEM]);
                warpgroup::mma_commit_group();
            }
            else {
                neg_infty(att_block[subtile]); // initial blocks must be zero
            }
        }
        warpgroup::mma_async_wait();

        // make first block anticausal.
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            auto &attn_subtile = reinterpret_cast<rt_fl_1x1<>&>(att_block[0].tiles[0][j]);
            if (j<warpid) neg_infty(attn_subtile);
            else if (j==warpid) make_anticausal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty());
        }
        // make last block causal
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            auto &attn_subtile = reinterpret_cast<rt_fl_1x1<>&>(att_block[WINDOW_TILES-1].tiles[0][j]);
            if (j>warpid) neg_infty(attn_subtile);
            else if (j==warpid) make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty());
        }

        // now do the softmax. first we subtract max for numerical stability. then exp.
        #pragma unroll
        for(int subtile = 0; subtile < WINDOW_TILES; subtile++) {
            row_max(max_vec, att_block[subtile], max_vec); // accumulate onto the max_vec
        }
        #pragma unroll
        for(int subtile = 0; subtile < WINDOW_TILES; subtile++) {
            sub_row(att_block[subtile], att_block[subtile], max_vec);
            exp(att_block[subtile], att_block[subtile]);
        }
        // now we sum so that we can divide.
        #pragma unroll
        for(int subtile = 0; subtile < WINDOW_TILES; subtile++) {
            row_sum(norm_vec, att_block[subtile], norm_vec);
        }
        #pragma unroll
        for(int subtile = 0; subtile < WINDOW_TILES; subtile++) {
            div_row(att_block[subtile], att_block[subtile], norm_vec);
            copy(att_block_bf[subtile], att_block[subtile]); // cast to bf16 for next matmul
        }
        __syncthreads();
        for(int subtile = 0; subtile < WINDOW_TILES; subtile++) {
            warpgroup::mma_fence(o_reg);
            warpgroup::mma_AB(o_reg, att_block_bf[subtile], v_smem[(start_block+subtile)%WINDOW_TILES_MEM]);
            warpgroup::mma_commit_group();
        }
        warpgroup::mma_async_wait();

        mul(o_reg, o_reg, beta); // beta multiply

        warpgroup::store(o_smem[tic], o_reg);
        __syncthreads();
        
        tma::store_async_wait();
        if (warpid == 0) { // store o
            tma::store_async(tma_o, o_smem[tic], blockIdx.x * qo_blocks + qo_blk); 
            tma::store_commit_group(); 
        }
    }
    tma::store_async_wait();
}


// For testing via C++
#include "harness.impl" // (comment out when using the code below)


// For binding to PyTorch (comment out include for harness.imple when using the code below)
// #include "src/common/pyutils/torch_helpers.cuh"
// #include <iostream>
// void sliding_window_tk(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {
//     std::cout << "Entered Sliding window handler" << std::endl;
//     CHECK_INPUT(q);
//     CHECK_INPUT(k);
//     CHECK_INPUT(v);
//     CHECK_INPUT(o);
    
//     auto batch = q.size(0);
//     auto heads = q.size(1);
//     auto threads = NUM_WORKERS * kittens::WARP_THREADS;
//     auto n     = q.size(2);
//     uint d     = q.size(3);

//     bool k_same = true, v_same = true;
//     for(auto i = 0; i < 2; i++) { 
//         k_same &= q.size(i) == k.size(i);
//         v_same &= q.size(i) == v.size(i);
//     }
//     k_same &= d == k.size(3);
//     v_same &= d == v.size(3);
//     v_same &= v.size(2) == n;
    
//     // This is just a restriction of what we're doing now...
//     TORCH_CHECK(k_same, "X and K_out should be same size");
//     TORCH_CHECK(v_same, "X and V_out should be same size");
//     TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "Q is a Bfloat");
//     TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "K is a Bfloat");
//     TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "V is a Bfloat");
//     TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "O is a Bfloat");
//     TORCH_CHECK(n % (NUM_WORKERS*kittens::TILE_DIM) == 0, "The number of elements should be divisible the number of workers times stored fragments");

//     // convert to bf16
//     c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
//     c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();

//     const bf16* q_bf = reinterpret_cast<const bf16*>(q_ptr);
//     const bf16* k_bf = reinterpret_cast<const bf16*>(k_ptr);
//     const bf16* v_bf = reinterpret_cast<const bf16*>(v_ptr);
//           bf16* o_bf = reinterpret_cast<bf16*>(o_ptr);

//     std::cout << "Checks and casts" << std::endl;
//     unsigned long mem_size = kittens::MAX_SHARED_MEMORY;
//     cudaFuncSetAttribute(
//         sliding_window,
//         cudaFuncAttributeMaxDynamicSharedMemorySize,
//         mem_size
//     );

//     std::cout << "Set dynamic memory" << std::endl;
//     sliding_window<<<batch*heads,threads,mem_size>>>(n, d, q_bf, k_bf, v_bf, o_bf);

//     // TODO: setup to launch with CUDA STREAM
//     // auto stream_wrapper = at::cuda::getCurrentCUDAStream(q.device().index());
//     // cudaStream_t stream = stream_wrapper.stream();
//     // sliding_window<H,T><<<batch*head,threads,0,stream>>>(n, 
//     //                     q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), 
//     //                     o.data_ptr<T>());

//     std::cout << "Launched kernel" << std::endl;
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//     std::cout << "Exiting" << std::endl;
// }

