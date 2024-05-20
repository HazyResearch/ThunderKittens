#include "../../../src/kittens.cuh"

#define MMA_N 128
#define MMA_K 128

#define WG_PER_BLOCK 4

#define MMA_M (64*WG_PER_BLOCK)

using namespace kittens;

// m, n, k must be multiples of 128 here.
__launch_bounds__(WG_PER_BLOCK*kittens::WARPGROUP_THREADS,1)
__global__ void simple_gemm(int m, int n, int k, CUtensorMap* A, CUtensorMap* B, CUtensorMap* C) {
    const int K_tiles = k / MMA_K;
    const int block_row = blockIdx.y*WG_PER_BLOCK;
    const int block_col = blockIdx.x;
    const int wg = kittens::warpid() / 4;
    const int wg_row = block_row + wg;
    const int wg_col = block_col;

    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_bf<4,8,wgmma_swizzle_l> (&a_smem)[2][WG_PER_BLOCK] = al.allocate<st_bf<4,8,wgmma_swizzle_l>,   2, WG_PER_BLOCK >();
    st_bf<8,8,wgmma_swizzle_l> (&b_smem)[2]               = al.allocate<st_bf<8,8,wgmma_swizzle_l>,   2               >();
    st_bf<4,8,wgmma_swizzle_l>       (&c_smem)  [WG_PER_BLOCK] = *reinterpret_cast<st_bf<4,8,wgmma_swizzle_l>(*)    [WG_PER_BLOCK]>(a_smem); // overload that memory for later.

    __shared__ uint64_t ab_smem_barrier, c_smem_barrier;
    int tic = 0, toc = 1;
    if (threadIdx.x == 0) {
        tma::init_barrier<typeof(c_smem[0]), WG_PER_BLOCK>(c_smem_barrier);
        tma::init_barrier(ab_smem_barrier);
        tma::set_bytes(ab_smem_barrier, size_bytes<typeof(a_smem[0][0]), WG_PER_BLOCK> + size_bytes<typeof(b_smem[0])>);

        #pragma unroll
        for(int j = 0; j < WG_PER_BLOCK; j++) {
            tma::load_async(a_smem[tic][j], A, ab_smem_barrier, block_row+j, 0);
        }
        tma::load_async(b_smem[tic], B, ab_smem_barrier, block_col, 0);
    }
    __syncthreads();

    rt_fl_1x8<> ab_reg;
    zero(ab_reg);

    for (int i = 0; i < K_tiles; i++, tic^=1, toc^=1) {
        tma::arrive_and_wait(ab_smem_barrier, tic);
        __syncthreads();

        if(threadIdx.x == 0 && i+1 < K_tiles) { // launch next load
            tma::set_bytes(ab_smem_barrier, size_bytes<typeof(a_smem[0][0]), WG_PER_BLOCK> + size_bytes<typeof(b_smem[0])>);

            #pragma unroll
            for(int j = 0; j < WG_PER_BLOCK; j++) { // represents wg_row_id
                tma::load_async(a_smem[toc][j], A, ab_smem_barrier, block_row+j, i+1);
            }
            tma::load_async(b_smem[toc], B, ab_smem_barrier, block_col, i+1);
        }

        warpgroup::mma_fence(ab_reg);
        warpgroup::mma_ABt(ab_reg, a_smem[tic][wg], b_smem[tic]);
        warpgroup::mma_commit_group();
        warpgroup::mma_async_wait();
    }
    __syncthreads();
    warpgroup::store(c_smem[wg], ab_reg);
    __syncthreads();

    if (kittens::warpid()%4 == 0) {
        tma::store_async(C, c_smem[wg], wg_row, wg_col);
        tma::store_commit_group();
    }
    tma::store_async_wait();
}

#include "harness.impl"
