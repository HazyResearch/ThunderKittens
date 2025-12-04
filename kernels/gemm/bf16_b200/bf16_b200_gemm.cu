#include "kittens.cuh"
#include <iostream>

using namespace kittens;

static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 64;

static constexpr int CLUSTER_M = 2*Mb;
static constexpr int CLUSTER_N = Nb;

static constexpr int CLUSTER_SIZE = 2;
static constexpr int NUM_CONSUMERS = 1;
static constexpr int NUM_PRODUCERS = 1;
static constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
static constexpr int NUM_THREADS = NUM_WORKERS * WARP_THREADS;
static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

static constexpr int SMEM_PIPE_DEPTH = 5;
static constexpr int MMA_PIPE_DEPTH = 2;
static constexpr int TMEM_PIPE_DEPTH = 8;

using a_tile = st_bf<Mb, Kb>;
using b_tile = st_bf<Nb/2, Kb>;
using d_tile = st_bf<Mb, Nb/TMEM_PIPE_DEPTH>;

using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

struct globals {
    a_gl a;
    b_gl b;
    d_gl d;

    __host__ __inline__ dim3 grid() { return dim3(148); }
    __host__ __inline__ dim3 block() { return dim3(NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <int SUPER_M>
__device__ inline int2 get_task_idx(const globals &g, int task_iter) {
    int task_id = task_iter * (gridDim.x/CLUSTER_SIZE) + blockIdx.x/CLUSTER_SIZE;
    int Rblocks = g.d.rows() / CLUSTER_M, Cblocks = g.d.cols() / CLUSTER_N;
    int super_rows = (Rblocks/SUPER_M)*SUPER_M,
        final_rows = Rblocks - super_rows,
        super_repeat = SUPER_M*Cblocks;
    if (task_id < super_rows * Cblocks) {
        return { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
    }
    else if (task_id < Rblocks*Cblocks) {
        int remainder_id = task_id - super_rows*Cblocks;
        return { super_rows + remainder_id%final_rows, remainder_id/final_rows };
    }
    else {
        return { -1, -1 };
    }
}

template <int SUPER_M>
__cluster_dims__(CLUSTER_SIZE, 1, 1) __launch_bounds__(NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ globals g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int cta_rank = cluster_ctarank();
    const int iters_per_task = g.a.cols() / Kb;

    static_assert(sizeof(a_tile) * SMEM_PIPE_DEPTH +
                  sizeof(b_tile) * SMEM_PIPE_DEPTH +
                  sizeof(d_tile) * 2 <= DYNAMIC_SHARED_MEMORY);
    a_tile (&a_smem)[SMEM_PIPE_DEPTH] = al.allocate<a_tile, SMEM_PIPE_DEPTH>();
    b_tile (&b_smem)[SMEM_PIPE_DEPTH] = al.allocate<b_tile, SMEM_PIPE_DEPTH>();
    d_tile (&d_smem)[2]               = al.allocate<d_tile, 2>();

    tensor_allocator<1, 2> tm_alloc{};
    using d_tt_t = tt<float, Mb, Nb>;

    __shared__ semaphore inputs_arrived[SMEM_PIPE_DEPTH], inputs_finished[SMEM_PIPE_DEPTH], outputs_arrived, outputs_finished[MMA_PIPE_DEPTH];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 0) { 
        #pragma unroll
        for (int i = 0; i < SMEM_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_finished[i], 0, CLUSTER_SIZE);
        }
    }
    everyone::tma::cluster::sync();

    if (warpgroup::groupid() == NUM_CONSUMERS) {
        warpgroup::increase_registers<256>();
        if (warp::laneid() == 0 && warpgroup::warpid() == 3) {
            int input_ring = 0;
            for (int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx<SUPER_M>(g, task_iter);
                if (rowcol.x == -1) {
                    for(int idx = 0; idx < (SMEM_PIPE_DEPTH); idx++) {
                        tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                        input_ring=ring_advance<SMEM_PIPE_DEPTH>(input_ring);
                    }
                    arrive(outputs_arrived);
                    break;
                }
                for (int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    update_phasebit<1>(bitfield, input_ring);
                    if(task_iter>0 && idx==SMEM_PIPE_DEPTH-1 && laneid() == 0) arrive(outputs_arrived); // TODO REVIEW 
                    tma::cluster::load_async(a_smem[input_ring], g.a, {rowcol.x*2+cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], g.b, {rowcol.y*2+cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    input_ring=ring_advance<SMEM_PIPE_DEPTH>(input_ring);
                }
            }
        }
        else if (cta_rank == 0 && warp::laneid() == 0 && warpgroup::warpid() == 0) {
            d_tt_t d_tt[MMA_PIPE_DEPTH] = {tm_alloc.allocate<d_tt_t>(0), tm_alloc.allocate<d_tt_t>(256)};
            int input_ring = 0;
            for(int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx<SUPER_M>(g, task_iter);
                if(rowcol.x == -1) break;
                tma::cluster::wait(outputs_finished[task_iter%2], ((task_iter+2)/2)%2);
                tma::cluster::expect_bytes(inputs_arrived[input_ring], 2*sizeof(a_tile) + 2*sizeof(b_tile));
                tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                update_phasebit<0>(bitfield, input_ring);
                mm2_ABt(d_tt[task_iter%2], a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                input_ring=ring_advance<SMEM_PIPE_DEPTH>(input_ring);
                for(int idx = 1; idx < iters_per_task; idx++) {
                    tma::cluster::expect_bytes(inputs_arrived[input_ring], 2*sizeof(a_tile) + 2*sizeof(b_tile));
                    tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    update_phasebit<0>(bitfield, input_ring);
                    mma2_ABt(d_tt[task_iter%2], a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                    input_ring=ring_advance<SMEM_PIPE_DEPTH>(input_ring);
                }
            }
        }
    }
    else {
        warpgroup::increase_registers<256>();
        d_tt_t d_tt[2] = {tm_alloc.allocate<d_tt_t>(0), tm_alloc.allocate<d_tt_t>(256)};
        for(int task_iter = 0; true; task_iter++) {
            int2 rowcol = get_task_idx<SUPER_M>(g, task_iter);
            if(rowcol.x == -1) break;
            wait(outputs_arrived, task_iter%2);
            rt_bf<Mb/4, Nb/TMEM_PIPE_DEPTH> d_reg[TMEM_PIPE_DEPTH];
            #pragma unroll
            for(int i = 0; i < TMEM_PIPE_DEPTH; i++) {
                warpgroup::load_async(d_reg[i], d_tt[task_iter%2].subtile<tt<float, Mb, Nb/TMEM_PIPE_DEPTH>>(0, Nb/TMEM_PIPE_DEPTH*i));
                tensor_load_wait();
                warpgroup::tma::store_async_read_wait<1>();
                warpgroup::sync(1);
                warpgroup::store(d_smem[i%2], d_reg[i]);
                warpgroup::sync(1);
                warpgroup::tma::store_async(g.d, d_smem[i%2], {2*rowcol.x+cta_rank, TMEM_PIPE_DEPTH*rowcol.y+i});
            }
            warpgroup::tma::store_async_read_wait();
            warpgroup::tma::cluster::arrive(outputs_finished[task_iter%2], 0);
        }
    }
}

constexpr bool NCU = false;
constexpr bool CHECK_CORRECTNESS = false;

#include <iostream>
#include <random>
#include <omp.h>

void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[j * K + k];
            }
            c[i * N + j] = sum;
        }
    }
}

template <int SUPER_M>
__host__ double run_benchmark(size_t M, size_t N, size_t K) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << " SUPER_M=" << SUPER_M << "  --------------------\n";
    std::cout << "Block size: " << Mb*2 << "x" << Nb << "x" << Kb << "\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    if (CHECK_CORRECTNESS) {
        cpu_gemm(h_A, h_B, h_C_ref, M, N, K);
        std::cout << "Performed CPU matrix multiplication" << std::endl;
    }

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    CUDACHECK(cudaMalloc(&d_A, M*K*sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_B, K*N*sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_C, M*N*sizeof(__nv_bfloat16)));
    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);
    CUDACHECK(cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice));
    std::cout << "Copied matrices to device" << std::endl;

    // Prepare kernel inputs
    a_gl Ag{d_A, nullptr, nullptr, M, K};
    b_gl Bg{d_B, nullptr, nullptr, N, K};
    d_gl Dg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Dg};

    // Set kernel attributes
    cudaFuncSetAttribute(kernel<SUPER_M>, cudaFuncAttributeMaxDynamicSharedMemorySize, G.dynamic_shared_memory());

    // Warmup
    for(int i = 0; i < (NCU ? 0 : 500); i++)
        kernel<SUPER_M><<<G.grid(), G.block(), G.dynamic_shared_memory()>>>(G);

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(start));
    constexpr int ITERS = (NCU ? 1 : 100);
    for(int i = 0; i < ITERS; i++)
        kernel<SUPER_M><<<G.grid(), G.block(), G.dynamic_shared_memory()>>>(G);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double useconds = milliseconds * 1000.0 / ITERS;
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;
    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    if (CHECK_CORRECTNESS) {
        // Copy result back to host
        CUDACHECK(cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost));
        std::cout << "Copied result back to host" << std::endl;

        // Convert result back to float for comparison
        for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);
        std::cout << "Converted result back to float" << std::endl;

        // Check result
        float max_error = 0.0f;
        float average_error = 0.0f;
        int error_count = 0;
        for (int i = 0; i < M * N; ++i) {
            float error = std::abs(h_C[i] - h_C_ref[i]);
            if(error > .5f) { // large because of bf16 vs fp32 numerics
                if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
                else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
                error_count++;
            }
            max_error = std::max(max_error, error);
            average_error += error;
        }
        average_error /= M*N;

        std::cout << "Max error: " << max_error << std::endl;
        std::cout << "Average error: " << average_error << std::endl;
        std::cout << "Error count: " << error_count << std::endl;
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

__host__ int main() {
    int N;
    N = 1024;
    run_benchmark<4>(N, N, N);
    N = 2048;
    run_benchmark<4>(N, N, N);
    N = 4096;
    run_benchmark<4>(N, N, N);
    N = 8192;
    run_benchmark<8>(N, N, N);
    N = 16384;
    run_benchmark<8>(N, N, N);
    return 0;
}
