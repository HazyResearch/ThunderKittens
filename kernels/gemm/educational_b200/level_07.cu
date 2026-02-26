#include "kittens.cuh"
using namespace kittens;

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;
static constexpr int TILE_K = 64;
static constexpr int PIPE_STAGES = 3;

static constexpr int NUM_WARPS = 4;
static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

using a_tile = st_bf<TILE_M, TILE_K>;
using b_tile = st_bf<TILE_K, TILE_N>;
using d_tile = st_bf<TILE_M, TILE_N>;

using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
using d_tt_t = tt<float, TILE_M, TILE_N>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ b_gl B_layout,
    const __grid_constant__ d_gl D_layout,
    int N
) {
    const int warpid = threadIdx.x / WARP_THREADS;
    const int laneid = threadIdx.x % WARP_THREADS;
    const int wg_laneid = warpgroup::laneid();

    const int grid_n = N / TILE_N;
    const int bid_m = blockIdx.x / grid_n;
    const int bid_n = blockIdx.x % grid_n;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a_tile (&a_smem)[PIPE_STAGES] = al.allocate<a_tile, PIPE_STAGES>();
    b_tile (&b_smem)[PIPE_STAGES] = al.allocate<b_tile, PIPE_STAGES>();
    d_tile (&d_smem) = al.allocate<d_tile>();

    __shared__ semaphore inputs_arrived[PIPE_STAGES];
    __shared__ semaphore inputs_finished[PIPE_STAGES];
    __shared__ semaphore compute_done;

    if (threadIdx.x == 0) {
        for (int i = 0; i < PIPE_STAGES; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 1, 0);
        }
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1, 1> tm_alloc{};
    d_tt_t accum;
    if (wg_laneid == 0) {
        accum = tm_alloc.allocate<d_tt_t>(0);
    }
    __syncthreads();

    const int num_k_iters = N / TILE_K;
    int phase = 0;

    if (warpid == 0 && laneid == 0) {
        for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
            const int stage = iter_k % PIPE_STAGES;

            wait(inputs_finished[stage], phase ^ 1);
            if (stage == PIPE_STAGES - 1) phase ^= 1;

            tma::expect_bytes(inputs_arrived[stage], sizeof(a_tile) + sizeof(b_tile));
            tma::load_async(a_smem[stage], A_layout, {bid_m, iter_k}, inputs_arrived[stage]);
            tma::load_async(b_smem[stage], B_layout, {iter_k, bid_n}, inputs_arrived[stage]);
        }
    }
    else if (warpid == 1 && laneid == 0) {
        for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
            const int stage = iter_k % PIPE_STAGES;

            wait(inputs_arrived[stage], phase);
            if (stage == PIPE_STAGES - 1) phase ^= 1;

            if (iter_k == 0) mm_AB (accum, a_smem[stage], b_smem[stage], inputs_finished[stage]);
            else              mma_AB(accum, a_smem[stage], b_smem[stage], inputs_finished[stage]);
        }
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
