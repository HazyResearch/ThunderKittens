#include "kittens.cuh"
using namespace kittens;

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

static constexpr int TILE_M = 128;
static constexpr int TILE_N = 64;
static constexpr int TILE_K = 32;
static constexpr int PIPE_STAGES = 2;
static constexpr int SUPERGROUP_N = 8;

static constexpr int NUM_CONSUMER_WARPS = 4;
static constexpr int NUM_PRODUCER_WARPS = 1;
static constexpr int NUM_WARPS = NUM_CONSUMER_WARPS + NUM_PRODUCER_WARPS;
static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
static constexpr int PRODUCER_WARP = NUM_CONSUMER_WARPS;
static constexpr int WARP_M = TILE_M / NUM_CONSUMER_WARPS;

static_assert(TILE_M % NUM_CONSUMER_WARPS == 0);
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

struct input_tiles {
    matmul_globals::a_tile a[PIPE_STAGES];
    matmul_globals::b_tile b[PIPE_STAGES];
};

static_assert(sizeof(matmul_globals::c_tile) <= sizeof(input_tiles),
              "C tile must fit in the retired A/B shared-memory buffer.");

union shared_tiles {
    input_tiles inputs;
    matmul_globals::c_tile c;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void kernel(const __grid_constant__ matmul_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);

    shared_tiles &tiles = al.allocate<shared_tiles>();
    auto &a_smem = tiles.inputs.a;
    auto &b_smem = tiles.inputs.b;

    __shared__ semaphore inputs_arrived[PIPE_STAGES];
    __shared__ semaphore inputs_finished[PIPE_STAGES];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < PIPE_STAGES; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 1, 0);
        }
    }
    __syncthreads();

    const int grid_m = g.N / TILE_M;
    const int grid_n = g.N / TILE_N;
    const int2 tile_coord = get_swizzled_2d_idx<SUPERGROUP_N>(
        grid_m, grid_n, blockIdx.x);
    const int bid_m = tile_coord.x;
    const int bid_n = tile_coord.y;
    const int num_k_tiles = g.N / TILE_K;

    uint32_t phasebits = 0xFFFF0000;

    if (warpid() == PRODUCER_WARP) {
        if (laneid() == 0) {
            int input_ring = 0;
            for (int tile_k = 0; tile_k < num_k_tiles; tile_k++) {
                wait(inputs_finished[input_ring], get_phasebit<1>(phasebits, input_ring));

                tma::expect_bytes(inputs_arrived[input_ring],
                                  sizeof(matmul_globals::a_tile) + sizeof(matmul_globals::b_tile));
                tma::load_async(a_smem[input_ring], g.A, {bid_m, tile_k}, inputs_arrived[input_ring]);
                tma::load_async(b_smem[input_ring], g.B, {tile_k, bid_n}, inputs_arrived[input_ring]);

                update_phasebit<1>(phasebits, input_ring);
                input_ring = ring_advance<PIPE_STAGES>(input_ring);
            }
        }
        return;
    }

    if (threadIdx.x < NUM_CONSUMER_WARPS * WARP_THREADS) {
        rt_fl<WARP_M, TILE_N> accum;
        warp::zero(accum);

        int input_ring = 0;
        for (int tile_k = 0; tile_k < num_k_tiles; tile_k++) {
            wait(inputs_arrived[input_ring], get_phasebit<0>(phasebits, input_ring));
            update_phasebit<0>(phasebits, input_ring);

            warpgroup::mma_AB(accum, a_smem[input_ring], b_smem[input_ring]);

            warpgroup::sync(1);
            warpgroup::arrive(inputs_finished[input_ring]);
            input_ring = ring_advance<PIPE_STAGES>(input_ring);
        }

        matmul_globals::c_tile &c_smem = tiles.c;
        rt_bf<WARP_M, TILE_N> c_reg;
        warp::copy(c_reg, accum);

        warpgroup::store(c_smem, c_reg);
        warpgroup::sync(1);

        if (warpgroup::laneid() == 0) {
            tma::store_async(g.C, c_smem, {bid_m, bid_n});
            tma::store_async_read_wait();
        }
    }
}

void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    using a_gl = matmul_globals::a_gl;
    using b_gl = matmul_globals::b_gl;
    using c_gl = matmul_globals::c_gl;

    a_gl a_arg{reinterpret_cast<bf16*>(A), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    b_gl b_arg{reinterpret_cast<bf16*>(B), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    c_gl c_arg{reinterpret_cast<bf16*>(C), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    matmul_globals g{a_arg, b_arg, c_arg, N};

    const int grid_m = N / TILE_M;
    const int grid_n = N / TILE_N;
    const int smem_size = sizeof(shared_tiles) + 1024;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<grid_m * grid_n, NUM_THREADS, smem_size>>>(g);
}

#include "launch.cu"
