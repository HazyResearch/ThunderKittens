#include "kittens.cuh"
using namespace kittens;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

static constexpr int TILE_M = 128;
static constexpr int TILE_N = 256;
static constexpr int TILE_K = 64;
static constexpr int PIPE_STAGES = 4;
static constexpr int CLUSTER_SIZE = 2;

static constexpr int EPI_PIPE_DEPTH = 8;
static constexpr int NUM_D_TILES = 2;

static constexpr int NUM_CONSUMERS = 2;
static constexpr int NUM_WARPS = (NUM_CONSUMERS + 1) * 4;
static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

using a_tile = st_bf<TILE_M, TILE_K>;
using b_tile = st_bf<TILE_K, TILE_N / 2>;
using d_tile = st_bf<TILE_M, TILE_N / EPI_PIPE_DEPTH>;

using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

using d_tt_t = tt<float, TILE_M, TILE_N>;

__global__
__cluster_dims__(CLUSTER_SIZE, 1, 1)
__launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ b_gl B_layout,
    const __grid_constant__ d_gl D_layout,
    int N
) {
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a_tile>();
        B_layout.template prefetch_tma<b_tile>();
        D_layout.template prefetch_tma<d_tile>();
    }

    const int cta_rank = cluster_ctarank();
    const int iters_per_task = N / TILE_K;

    const int cluster_idx = blockIdx.x / CLUSTER_SIZE;
    const int grid_n = N / TILE_N;
    const int2 tile_coord = { cluster_idx / grid_n, cluster_idx % grid_n };

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a_tile (&a_smem)[PIPE_STAGES][NUM_CONSUMERS] = al.allocate<a_tile, PIPE_STAGES, NUM_CONSUMERS>();
    b_tile (&b_smem)[PIPE_STAGES]                = al.allocate<b_tile, PIPE_STAGES>();
    d_tile (&d_smem)[NUM_CONSUMERS][NUM_D_TILES]  = al.allocate<d_tile, NUM_CONSUMERS, NUM_D_TILES>();

    tensor_allocator<1, 2> tm_alloc{};

    __shared__ semaphore inputs_arrived[PIPE_STAGES], inputs_finished[PIPE_STAGES];
    __shared__ semaphore outputs_arrived[NUM_CONSUMERS];
    uint32_t bitfield = 0xFFFF0000;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < PIPE_STAGES; i++) {
            init_semaphore(inputs_arrived[i], 0, NUM_CONSUMERS);
            init_semaphore(inputs_finished[i], 0, NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < NUM_CONSUMERS; i++) {
            init_semaphore(outputs_arrived[i], 0, 1);
        }
    }
    everyone::tma::cluster::sync();

    if (warpgroup::groupid() == NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();

        if (warp::laneid() == 0 && warpgroup::warpid() == 3) {
            int input_ring = 0;
            for (int idx = 0; idx < iters_per_task; idx++) {
                tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                update_phasebit<1>(bitfield, input_ring);

                #pragma unroll
                for (int i = 0; i < NUM_CONSUMERS; i++)
                    tma::cluster::load_async(a_smem[input_ring][i], A_layout,
                        {(tile_coord.x * 2 + cta_rank) * NUM_CONSUMERS + i, idx},
                        inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);

                tma::cluster::load_async(b_smem[input_ring], B_layout,
                    {idx, tile_coord.y * 2 + cta_rank},
                    inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);

                input_ring = ring_advance<PIPE_STAGES>(input_ring);
            }
        }
        else if (cta_rank == 0 && warp::laneid() == 0 && warpgroup::warpid() < NUM_CONSUMERS) {
            d_tt_t accum = tm_alloc.allocate<d_tt_t>(warpgroup::warpid() * TILE_N);

            int input_ring = 0;
            for (int idx = 0; idx < iters_per_task; idx++) {
                tma::cluster::expect_bytes(inputs_arrived[input_ring],
                    (CLUSTER_SIZE * NUM_CONSUMERS * sizeof(a_tile) + 2 * sizeof(b_tile)) / NUM_CONSUMERS);
                tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                update_phasebit<0>(bitfield, input_ring);

                if (idx == 0) mm2_AB (accum, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                else          mma2_AB(accum, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);

                input_ring = ring_advance<PIPE_STAGES>(input_ring);
            }
            detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived[warpgroup::warpid()]);
        }
    }
    else {
        warpgroup::increase_registers<224>();

        d_tt_t accum = tm_alloc.allocate<d_tt_t>(warpgroup::groupid() * TILE_N);

        wait(outputs_arrived[warpgroup::groupid()], 0);

        rt_bf<TILE_M / 4, TILE_N / EPI_PIPE_DEPTH> d_reg[EPI_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < EPI_PIPE_DEPTH; i++)
            warpgroup::load_async(d_reg[i],
                accum.template subtile<tt<float, TILE_M, TILE_N / EPI_PIPE_DEPTH>>(
                    0, (TILE_N / EPI_PIPE_DEPTH) * i));
        tensor_load_wait();

        warpgroup::sync(warpgroup::groupid() + 1);

        #pragma unroll
        for (int i = 0; i < EPI_PIPE_DEPTH; i++) {
            warpgroup::tma::store_async_read_wait<NUM_D_TILES - 1>();
            warpgroup::sync(warpgroup::groupid() + 1);
            warpgroup::store(d_smem[warpgroup::groupid()][i % NUM_D_TILES], d_reg[i]);
            warpgroup::sync(warpgroup::groupid() + 1);
            warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                D_layout, d_smem[warpgroup::groupid()][i % NUM_D_TILES],
                {(2 * tile_coord.x + cta_rank) * NUM_CONSUMERS + warpgroup::groupid(),
                 EPI_PIPE_DEPTH * tile_coord.y + i});
        }
    }
}

void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    a_gl A_layout{reinterpret_cast<bf16*>(A), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    b_gl B_layout{reinterpret_cast<bf16*>(B), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    d_gl D_layout{reinterpret_cast<bf16*>(C), nullptr, nullptr, (unsigned long)N, (unsigned long)N};

    int grid = (N / (CLUSTER_SIZE * NUM_CONSUMERS * TILE_M)) * (N / TILE_N) * CLUSTER_SIZE;
    int smem_size = MAX_SHARED_MEMORY - 1024;

    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    matmul_kernel<<<grid, NUM_THREADS, smem_size>>>(A_layout, B_layout, D_layout, N);
}

#include "launch.cu"
