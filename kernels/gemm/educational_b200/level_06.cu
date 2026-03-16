#include "kittens.cuh"
using namespace kittens;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;
static constexpr int TILE_K = 64;
static constexpr int NUM_WARPS = 4;
static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

using a_tile = st_bf<TILE_M, TILE_K>;
using b_tile = st_bf<TILE_K, TILE_N>;
using d_tile = st_bf<TILE_M, TILE_N>;

using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
using d_tt_t = tt<float, TILE_M, TILE_N>;

__global__
__launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ b_gl B_layout,
    const __grid_constant__ d_gl D_layout,
    int N
) {
    const int wg_laneid = warpgroup::laneid();

    const int grid_n = N / TILE_N;
    const int bid_m = blockIdx.x / grid_n;
    const int bid_n = blockIdx.x % grid_n;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a_tile (&a_smem) = al.allocate<a_tile>();
    b_tile (&b_smem) = al.allocate<b_tile>();
    d_tile (&d_smem) = al.allocate<d_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore inputs_finished;
    __shared__ semaphore compute_done;

    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(inputs_finished, 1, 0);
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1, 1> tm_alloc{};
    d_tt_t accum;
    if (wg_laneid == 0) {
        accum = tm_alloc.allocate<d_tt_t>(0);
    }
    warpgroup::sync(1);

    const int num_k_iters = N / TILE_K;
    int phase = 0;

    for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
        if (threadIdx.x == 0) {
            wait(inputs_finished, phase ^ 1);
            tma::expect_bytes(inputs_arrived, sizeof(a_tile) + sizeof(b_tile));
            tma::load_async(a_smem, A_layout, {bid_m, iter_k}, inputs_arrived);
            tma::load_async(b_smem, B_layout, {iter_k, bid_n}, inputs_arrived);
        }

        wait(inputs_arrived, phase);
        phase ^= 1;

        if (wg_laneid == 0) {
            if (iter_k == 0) mm_AB (accum, a_smem, b_smem, inputs_finished);
            else             mma_AB(accum, a_smem, b_smem, inputs_finished);
        }
    }

    if (wg_laneid == 0) {
        detail::tcgen05::commit<1>(compute_done);
    }
    wait(compute_done, 0);

    rt_bf<TILE_M / 4, TILE_N> d_reg;
    warpgroup::load_async(d_reg, accum);
    tensor_load_wait();

    warpgroup::sync(1);
    warpgroup::store(d_smem, d_reg);
    warpgroup::sync(1);

    if (wg_laneid == 0) {
        tma::store_async(D_layout, d_smem, {bid_m, bid_n});
        tma::store_async_read_wait();
    }
}

void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    a_gl A_layout{reinterpret_cast<bf16*>(A), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    b_gl B_layout{reinterpret_cast<bf16*>(B), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    d_gl D_layout{reinterpret_cast<bf16*>(C), nullptr, nullptr, (unsigned long)N, (unsigned long)N};

    int grid = (N / TILE_M) * (N / TILE_N);
    int smem_size = MAX_SHARED_MEMORY - 1024;

    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    matmul_kernel<<<grid, NUM_THREADS, smem_size>>>(A_layout, B_layout, D_layout, N);
}

#include "launch.cu"
