#include "kittens.cuh"
using namespace kittens;

using a_tile = st_bf<4,4>;
using b_tile = st_bf<4,16>;
using c_tile = st_bf<4,16>;

template<int _NUM_CONSUMER_WARPGROUPS>
struct producer_consumer_parameters {
    static constexpr int NUM_CONSUMER_WARPGROUPS = _NUM_CONSUMER_WARPGROUPS;
    static_assert(NUM_CONSUMER_WARPGROUPS >= 2 && NUM_CONSUMER_WARPGROUPS <= 6); // The register alloc is only set up for this range.
    static constexpr int NUM_CONSUMER_WARPS      = NUM_CONSUMER_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_WARPS               = NUM_CONSUMER_WARPS + WARPGROUP_WARPS; // producers, too
    static constexpr int NUM_THREADS             = NUM_WARPS * WARP_THREADS;
    static constexpr int NUM_PRODUCER_REG        = NUM_CONSUMER_WARPGROUPS == 2 ? 32 : 24;
    static constexpr int NUM_CONSUMER_REG        = 480/NUM_CONSUMER_WARPGROUPS-8; // valid up to 6 consumer warpgroups
};

struct globals {
    int n_blocks;
    const CUtensorMap* A_tma;
    const CUtensorMap* B_tma;
    CUtensorMap* C_tma;
    __host__ __device__ inline globals(int n_blocks, const CUtensorMap* A_tma, const CUtensorMap* B_tma, CUtensorMap* C_tma) :
        n_blocks(n_blocks), A_tma(A_tma), B_tma(B_tma), C_tma(C_tma) {}
};

struct block { // the chunk of data that the producer and consumer are working on
    a_tile (&a_block)[2];
    b_tile (&b_block);
    __device__ inline block(a_tile (&a_block)[2], b_tile (&b_block)) : a_block(a_block), b_block(b_block) {}
};

struct producer_consumer {
    static constexpr int NUM_CONSUMER_WARPGROUPS = 2;
    using params = producer_consumer_parameters<NUM_CONSUMER_WARPGROUPS>;

    struct producer {
        struct state {
            int row_idx, col_idx;
        }; // persistent registers; none needed for this kernel.
        __device__ static void setup(state &s, globals &g) { // setup and load the first iteration
            warpgroup::decrease_registers<params::NUM_PRODUCER_REG>(); // decrease registers for the producer warpgroup
            s.row_idx = blockIdx.x * 2; // 2 tiles vertical per block
            s.col_idx = blockIdx.y; // 1 tile horizontal per block
        }
        __device__ static void load(state &s, block &b, globals &g, kittens::barrier &bar, int iter) { // barrier for the producer to load into
            if(warpgroup::warpid() == 0) {
                tma::expect_bytes(bar, size_bytes<a_tile>*NUM_CONSUMER_WARPGROUPS + size_bytes<b_tile>);
                #pragma unroll
                for(int i = 0; i < NUM_CONSUMER_WARPGROUPS; i++) {
                    tma::load_async(b.a_block[i], g.A_tma, bar, s.row_idx+i, iter);
                }
                tma::load_async(b.b_block, g.B_tma, bar, iter, s.col_idx);
            }
        }
        __device__ static void finish(state &s, globals &g) {}
    };

    struct consumer {
        struct state {
            rt_fl<1,16> acc;
            c_tile &out_block;
            __host__ __device__ inline state(c_tile &out_block) : out_block(out_block) {}
        }; // persistent registers; none needed for this kernel.
        __device__ static void setup(state &s, globals &g) { // setup locals for before the first iteration
            warpgroup::increase_registers<params::NUM_CONSUMER_REG>();
            zero(s.acc);
        }
        __device__ static void compute(state &s, block &b, globals &g, int iter) {
            warpgroup::mma_fence(s.acc);
            warpgroup::mma_AB(s.acc, b.a_block[warpgroup::groupid()], b.b_block);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
        }
        __device__ static void finish(state &s, globals &g) {
            warpgroup::store(s.out_block, s.acc);
            warpgroup::sync(); // writes to shared memory are now visible
            if(warpgroup::warpid() == 0) { // first warp stores
                tma::store_async(g.C_tma, s.out_block, blockIdx.x * 2 + warpgroup::groupid(), blockIdx.y);
                tma::store_commit_group();
            }
            tma::store_async_read_wait(); // this isn't really necessary, but it illustrates the principle.
            warpgroup::sync();
        }
    };
};

// This is a producer+consumer copy kernel that demonstrates the use of TMA to implement a two-stage pipeline.
__global__ __launch_bounds__(producer_consumer::params::NUM_THREADS, 1)
void gpu_gemm(globals g) {
    using pc = producer_consumer;

    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    a_tile (&a_smem) [2][producer_consumer::params::NUM_CONSUMER_WARPGROUPS] = alloc.allocate<a_tile, 2, producer_consumer::params::NUM_CONSUMER_WARPGROUPS>();
    b_tile (&b_smem) [2]                                                     = alloc.allocate<b_tile, 2>();
    c_tile (&c_smem) [producer_consumer::params::NUM_CONSUMER_WARPGROUPS]    = alloc.allocate<c_tile, producer_consumer::params::NUM_CONSUMER_WARPGROUPS>();
    block blocks[] = {
        block(a_smem[0], b_smem[0]),
        block(a_smem[1], b_smem[1])
    };

    // Initialize barriers. This is constant for all two-stage producer-consumer kernels.
    __shared__ kittens::barrier producer_arrived[2], consumer_arrived[2];
    int tic = 0, toc = 1; // these are used to track the two-stage pipeline.
    if (warpid() == 0) { // a single warp (in fact a single thread) does these.
        init_barrier(producer_arrived[tic], 0, 1); // needs to wait on just one memory transaction, each
        init_barrier(producer_arrived[toc], 0, 1);
        init_barrier(consumer_arrived[tic], pc::params::NUM_CONSUMER_WARPS, 0); // needs to wait on one thread from each consumer warp
        init_barrier(consumer_arrived[toc], pc::params::NUM_CONSUMER_WARPS, 0);
    }

    __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

    if(warpgroup::groupid() == pc::params::NUM_CONSUMER_WARPGROUPS) { // last warpgroup is a producer
        typename pc::producer::state s;
        pc::producer::setup(s, g);
        pc::producer::load(s, blocks[tic], g, producer_arrived[tic], 0); // load initial block
        for (int block_idx = 1; block_idx < g.n_blocks; block_idx++, tic=tic^1, toc=toc^1) {
            pc::producer::load(s, blocks[toc], g, producer_arrived[toc], block_idx);
            wait(consumer_arrived[tic], ((block_idx-1)/2)%2); // phase changes at half the rate of the tic/toc
        }
        pc::producer::finish(s, g);
    }
    else { // other warpgroups are consumers
        typename pc::consumer::state s(c_smem[warpgroup::groupid()]);
        pc::consumer::setup(s, g);
        for (int block_idx = 0; block_idx < g.n_blocks; block_idx++, tic^=1, toc^=1) {
            wait(producer_arrived[tic], (block_idx/2)%2); // wait for memory to arrive
            pc::consumer::compute(s, blocks[tic], g, block_idx);
            if(laneid() == 0) arrive(consumer_arrived[tic]); // work is complete, signal to the producer that it may start the next load.
        }
        pc::consumer::finish(s, g);
    }
}


#include <iostream>
#include <random>
#include <cuda_bf16.h>

#include <omp.h>
void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

int main() {
    const int M = 4096, N = 4096, K = 4096;
    const size_t size_bytes = M * N * sizeof(float);
    const size_t size_bytes_bf16 = M * N * sizeof(__nv_bfloat16);

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);

    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_bytes_bf16);
    cudaMalloc(&d_B, size_bytes_bf16);
    cudaMalloc(&d_C, size_bytes_bf16);

    std::cout << "Allocated device memory" << std::endl;

    std::cout << "a_tile::rows=" << a_tile::rows << " a_tile::cols=" << a_tile::cols << std::endl;
    std::cout << "b_tile::rows=" << b_tile::rows << " b_tile::cols=" << b_tile::cols << std::endl;
    std::cout << "c_tile::rows=" << c_tile::rows << " c_tile::cols=" << c_tile::cols << std::endl;
    CUtensorMap* tma_A_d = tma::allocate_and_create_tensor_map<a_tile>(d_A, M/a_tile::rows, K/a_tile::cols);
    CUtensorMap* tma_B_d = tma::allocate_and_create_tensor_map<b_tile>(d_B, K/b_tile::rows, N/b_tile::cols);
    CUtensorMap* tma_C_d = tma::allocate_and_create_tensor_map<c_tile>(d_C, M/c_tile::rows, N/c_tile::cols);

    std::cout << "Allocated TMA memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);

    cudaMemcpy(d_A, h_A_bf16, size_bytes_bf16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, size_bytes_bf16, cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    unsigned long mem_size = 200000; // need to launch two blocks if possible.
    
    cudaFuncSetAttribute(gpu_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    // Launch kernel
    dim3 grid(M / c_tile::rows, N / c_tile::cols); // rows, cols
    dim3 block(producer_consumer::params::NUM_THREADS);

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < 10; i++) {
        gpu_gemm<<<grid, block, mem_size>>>(globals(K/a_tile::cols, tma_A_d, tma_B_d, tma_C_d));
    }
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double seconds = diff.count();

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K * 10; // 2 FLOPs per multiply-add
    double tflops = (flops / seconds) / 1e12;

    std::cout << "Launched kernel with grid (" << grid.x << ", " << grid.y << ") and block (" << block.x << ")\n";
    std::cout << "Kernel execution time: " << seconds << " seconds\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    // Copy result back to host
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
    cudaMemcpy(h_C_bf16, d_C, size_bytes_bf16, cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if(error > 1.0) { // large because of bf16 vs fp32 numerics
            if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        max_error = std::max(max_error, error);
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

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

    return 0;
}
