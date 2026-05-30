#include "kittens.cuh"
using namespace kittens;

#include <cuda_bf16.h>
#include <cuda_runtime.h>

static constexpr int TILE_M = 128;
static constexpr int TILE_N = 64;
static constexpr int TILE_K = 32;
static constexpr int NUM_WARPS = 4;
static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
static constexpr int WARP_M = TILE_M / NUM_WARPS;

static_assert(TILE_M % NUM_WARPS == 0);
static_assert(WARP_M % 16 == 0);
static_assert(TILE_N % 16 == 0);
static_assert(TILE_K % 16 == 0);

struct matmul_globals {
    using a_tile = st_bf<TILE_M, TILE_K>;
    using b_tile = st_bf<TILE_K, TILE_N>;
    using c_tile = st_bf<TILE_M, TILE_N>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using c_gl = gl<bf16, 1, 1, -1, -1, c_tile>;

    a_gl A;
    b_gl B;
    c_gl C;
    int N;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void kernel(const __grid_constant__ matmul_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator<128> al((int *)&__shm[0]);

    matmul_globals::a_tile &a_smem = al.allocate<matmul_globals::a_tile>();
    matmul_globals::b_tile &b_smem = al.allocate<matmul_globals::b_tile>();
    matmul_globals::c_tile &c_smem = al.allocate<matmul_globals::c_tile>();

    const int bid_n = blockIdx.x;
    const int bid_m = blockIdx.y;

    rt_fl<WARP_M, TILE_N> accum;
    warp::zero(accum);

    const int num_k_tiles = g.N / TILE_K;
    for(int tile_k = 0; tile_k < num_k_tiles; tile_k++) {
        warpgroup::load(a_smem, g.A, {bid_m, tile_k});
        warpgroup::load(b_smem, g.B, {tile_k, bid_n});
        __syncthreads();

        warpgroup::mma_AB(accum, a_smem, b_smem);
        __syncthreads();
    }

    rt_bf<WARP_M, TILE_N> c_reg;
    warp::copy(c_reg, accum);

    warpgroup::store(c_smem, c_reg);
    __syncthreads();

    warpgroup::store(g.C, c_smem, {bid_m, bid_n});
}

void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    using a_gl = matmul_globals::a_gl;
    using b_gl = matmul_globals::b_gl;
    using c_gl = matmul_globals::c_gl;

    a_gl a_arg{reinterpret_cast<bf16*>(A), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    b_gl b_arg{reinterpret_cast<bf16*>(B), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    c_gl c_arg{reinterpret_cast<bf16*>(C), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    matmul_globals g{a_arg, b_arg, c_arg, N};

    dim3 blocks(N / TILE_N, N / TILE_M);
    const int smem_size = sizeof(matmul_globals::a_tile) +
                          sizeof(matmul_globals::b_tile) +
                          sizeof(matmul_globals::c_tile) + 1024;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<blocks, NUM_THREADS, smem_size>>>(g);
}

#include "launch.cu"
