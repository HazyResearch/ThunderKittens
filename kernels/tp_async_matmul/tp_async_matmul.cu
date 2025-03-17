/*
    Idea is from this article: https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487
    Implementation is by Stuart Sul
*/

#include "kittens.cuh"
#include "prototype.cuh"

constexpr int NUM_DEVICES = 8;

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                           (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(
                args.state.accum, // dest registers
                args.input.a[warpgroup::groupid()], // A matrix
                reinterpret_cast<wide_tile&>(args.input.b) // B matrix
            );
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::warpid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                             {args.common.coord.x, args.common.coord.y+i});
                tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
            }
            zero(args.state.accum);
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};


// constexpr bool NCU = false;
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

template<typename mmt>
void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block, cudaStream_t stream) {
    using global_layout = typename mmt::layout::global_layout;
    using globals  = typename mmt::layout::globals;
    global_layout Ag{d_A, nullptr, nullptr, M, K};
    global_layout Bg{d_B, nullptr, nullptr, K, N};
    global_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024, stream>>>(G);
}

__global__ void print_bfloat16(__nv_bfloat16* data, int dev_idx) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
        blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("Dev %d: ", dev_idx);
        for (int i = 0; i < 10; ++i) printf("%f ", __bfloat162float(data[i])); 
        printf("\n");
    }
}

template<typename mmt>
int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << mmt::M_BLOCK*64 << "x" << mmt::N_BLOCK*64 << "\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

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
    // NOTE: in order for the check to pass, it should be ITER = 1 without warmups
    bool check = false;
    if(check) cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);

    // Allocate device memory
    int M_sh = M / NUM_DEVICES, N_sh = N / NUM_DEVICES;
    __nv_bfloat16 *d_A[NUM_DEVICES][2], *d_B[NUM_DEVICES], *d_C[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        cudaMalloc(&d_A[dev_idx][0], M_sh*K*sizeof(__nv_bfloat16)); // row-wise sharding; used to alternate comm & comp
        cudaMalloc(&d_A[dev_idx][1], M_sh*K*sizeof(__nv_bfloat16)); // row-wise sharding
        cudaMalloc(&d_B[dev_idx], K*N_sh*sizeof(__nv_bfloat16)); // column-wise sharding
        cudaMalloc(&d_C[dev_idx], M*N_sh*sizeof(__nv_bfloat16)); // A is asynchronously all-gathered, so column-wise sharding only
    }

    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        cudaMemcpy(d_A[dev_idx][0], &h_A_bf16[dev_idx * M_sh * K], M_sh*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        for (int i = 0; i < K; ++i) { // must memcpy per row as it's not contiguous
            cudaMemcpy(d_B[dev_idx] + i * N_sh, &h_B_bf16[i*N + dev_idx*N_sh], N_sh*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        }
    }
    // for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
    //     cudaSetDevice(dev_idx);
    //     print_bfloat16<<<1, 1, 0>>>(d_A[dev_idx][0], dev_idx);
    //     cudaDeviceSynchronize();
    // }
    // for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
    //     cudaSetDevice(dev_idx);
    //     print_bfloat16<<<1, 1, 0>>>(d_B[dev_idx], dev_idx);
    //     cudaDeviceSynchronize();
    // }

    std::cout << "Copied matrices to device" << std::endl;

    // Set up kernel - smem
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    }

    // Set up kernel - streams
    cudaStream_t streams[NUM_DEVICES][2];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        cudaStreamCreate(&streams[dev_idx][0]);
        cudaStreamCreate(&streams[dev_idx][1]);
    }

    // Set up kernel - threadpool
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;
    KittensClub club(device_ids, NUM_DEVICES);

    // Set up kerenl - P2P access (todo: only access to next device needed)
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        for (int peer_dev_idx = 0; peer_dev_idx < NUM_DEVICES; ++peer_dev_idx) {
            if (peer_dev_idx == dev_idx) continue;
            int can_access_peer;
            cudaDeviceCanAccessPeer(&can_access_peer, dev_idx, peer_dev_idx);
            if (can_access_peer) cudaDeviceEnablePeerAccess(peer_dev_idx, 0);
            else {
                std::cerr << "Device " << dev_idx << " cannot access device " << peer_dev_idx << std::endl;
                return -1;
            }
        }
    }

    // Launch kernel
    dim3 grid(mmt::grid(M_sh, N_sh, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    // Skip warmup for now

    // Start timing
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = 10;
    for(int i = 0; i < ITERS; i++) {
        for (int step = 0; step < NUM_DEVICES; ++step) {
            club.execute([&](int dev_idx) {
                int comp_idx = step % 2;
                int comm_idx = (step + 1) % 2;
                int shard_idx = (dev_idx + step) % NUM_DEVICES;
                int peer_idx = (dev_idx + 1) % NUM_DEVICES;
                // if (dev_idx == 7) { // debug code
                //     printf("Step %d\n", step);
                //     print_bfloat16<<<1, 1, 0>>>(d_A[dev_idx][comp_idx], dev_idx);
                //     cudaDeviceSynchronize();
                //     print_bfloat16<<<1, 1, 0>>>(d_A[dev_idx][comm_idx], dev_idx);
                //     cudaDeviceSynchronize();
                //     print_bfloat16<<<1, 1, 0>>>(d_B[dev_idx], dev_idx);
                //     cudaDeviceSynchronize();
                // }
                inner_run<mmt>(  d_A[dev_idx][comp_idx],
                                 d_B[dev_idx],
                                 d_C[dev_idx] + shard_idx * M_sh * N_sh,
                                 M_sh, N_sh, K, grid, block, 
                                 streams[dev_idx][comp_idx]   );
                if (step < NUM_DEVICES - 1)
                    cudaMemcpyAsync( d_A[dev_idx][comm_idx], 
                                     d_A[peer_idx][comp_idx], 
                                     M_sh*K*sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, 
                                     streams[dev_idx][comm_idx]   );
                cudaDeviceSynchronize();
            });
        }
    }

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;

    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    // Copy result back to host
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        for (int i = 0; i < M; ++i) { // must memcpy per row as it's not contiguous
            cudaMemcpy(&h_C_bf16[i * N + dev_idx * N_sh], d_C[dev_idx] + i * N_sh, N_sh*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        }
    }
    // for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
    //     cudaSetDevice(dev_idx);
    //     print_bfloat16<<<1, 1, 0>>>(d_C[dev_idx] + 2000 * N_sh, dev_idx);
    //     cudaDeviceSynchronize();
    // }
    // for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
    //     printf("Dev %d: ", dev_idx);
    //     for (int i = 0; i < 10; ++i) printf("%f ", h_C_ref[2000 * N + dev_idx * N_sh + i]);
    //     printf("\n");
    // }

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    // NOTE: in order for the check to pass, it should be ITER = 1 without warmups
    if (check) {
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
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        cudaFree(d_A[dev_idx][0]);
        cudaFree(d_A[dev_idx][1]);
        cudaFree(d_B[dev_idx]);
        cudaFree(d_C[dev_idx]);
        cudaStreamDestroy(streams[dev_idx][0]);
        cudaStreamDestroy(streams[dev_idx][1]);
    }

    return 0;
}

int main() {
    // int Cblocks = 22, Rblocks = 24;
    // int Cblocks192 = 20, Rblocks192 = 16;
    // run_benchmark<matmul_template<4>>(4096, 4096, 4096, Rblocks, Cblocks, Rblocks192, Cblocks192);
    // run_benchmark<matmul_template<8>>(4096, 4096, 4096, Rblocks, Cblocks, Rblocks192, Cblocks192);
    // run_benchmark<matmul_template<12>>(4096, 4096, 4096, Rblocks, Cblocks, Rblocks192, Cblocks192);
    int N = 8124;
    run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // N = 3072;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<3,3,8>>(N, N, N);
    // N = 4096;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // N = 6144;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<3,3,8>>(N, N, N);
    // N = 8192;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // N = 12288;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<3,3,8>>(N, N, N);
    // N = 16384;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<2,4,12>>(N, N, N);
    // run_benchmark<matmul_template<3,3,12>>(192*12, 192*11, 8192);
    // run_benchmark<matmul_template<2,4,11>>(128*22, 256* 6, 8192);
    // run_benchmark<matmul_template<2,4,1>>(128 * 132, 256, 256);
    // run_benchmark<matmul_template<2,4,1>>(128 * 133, 256, 256);
    // run_benchmark<matmul_template<2,4,1>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<2,4,8>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<2,4,12>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<2,4,128>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<3,3,12>>(192*22, 192*6*2, 8192);
    // run_benchmark<matmul_template<3,3,12>>(192*22, 192*6*2, 16384);
    // run_benchmark<matmul_template<2,4,11>>(128*22*2, 256* 6*2, 8192);
    // run_benchmark<matmul_template<3,3,12>>(192*12*2, 192*11*2, 8192*2);
    // run_benchmark<matmul_template<2,4,11>>(128*22*2, 256* 6*2, 8192*2);
    return 0;
}