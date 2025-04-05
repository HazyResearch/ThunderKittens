#include <iostream>
#include <random>

#include "kittens.cuh"
#include "prototype.cuh"
#include <cuda_bf16.h>

constexpr size_t N = 65536;
constexpr int NUM_WARMUPS = 2; // number of warpups
constexpr int NUM_ITERS = 10; // number of iterations for benchmarking
constexpr int NUM_DEVICES = 8; // number of GPUs

constexpr bool CHECK = NUM_ITERS == 1 && NUM_WARMUPS == 0;

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    using  pglobal_layout = pgl<global_layout, NUM_DEVICES, true, true, base_tile>;
    struct globals        { pglobal_layout A, B, C; int dev_idx; };
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
        int dev_idx = args.globals.dev_idx;
        int Rblocks = args.globals.C[dev_idx].rows() / (M_BLOCK*64), Cblocks = args.globals.C[dev_idx].cols() / (N_BLOCK*64);
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
        args.num_iters = args.globals.A[dev_idx].cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            int dev_idx = args.globals.dev_idx;
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A[dev_idx],
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B[dev_idx],
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
            int dev_idx = args.globals.dev_idx;
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::warpid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                // tma::store_async(args.globals.C[dev_idx], args.finish.c[warpgroup::groupid()][i],
                //                              {args.common.coord.x, args.common.coord.y+i});
                tma::store_add_async(args.globals.C/*PGL*/, args.finish.c[warpgroup::groupid()][i],
                                             {args.common.coord.x, args.common.coord.y+i}, dev_idx);
                tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
            }
            zero(args.state.accum);
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<typename mmt>
void run(size_t M, size_t N, size_t K) {
    
    std::cout << "------------------------- Benchmark -------------------------\n";
    std::cout << "  M = " << M << ", N = " << N << ", K = " << K << "\n";
    std::cout << "  Block size: " << mmt::M_BLOCK * 64 << "x" << mmt::N_BLOCK * 64 << "\n";

    // Host-side matrices
    float *host_A = new float[M * K];
    float *host_B = new float[K * N];
    float *host_C = new float[NUM_DEVICES * M * N];
    float *host_C_ref = new float[M * N];

    // Device ID array
    int device_ids[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;

    // Initialize A & B matrices
    std::mt19937 prng(42);
    std::uniform_real_distribution<> random(-0.5, 0.5);
    std::cout << "\n  Matrix A (M x K): ";
    for (size_t i = 0; i < M * K; ++i) {
        host_A[i] = random(prng);
        if (i < 10)
            std::cout << host_A[i] << " ";
    }
    std::cout << "\n  Matrix B (K x N): ";
    for (size_t i = 0; i < K * N; ++i) {
        host_B[i] = random(prng);
        if (i < 10)
            std::cout << host_B[i] << " ";
    }
    std::cout << "\n";

    // Generate expected output (just do first 10x10 tile)
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += host_A[i * K + k] * host_B[k * N + j];
            }
            host_C_ref[i * N + j] = sum;
        }
    }
    std::cout << "  Expected C (M x N): ";
    for (int i = 0; i < 10; i++) {
        std::cout << host_C_ref[i] << " ";
    }
    std::cout << "\n";

    // Convert to BF16 on host
    __nv_bfloat16 *host_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *host_B_bf16 = new __nv_bfloat16[K * N];
    for (size_t i = 0; i < M * K; ++i) host_A_bf16[i] = __float2bfloat16(host_A[i]);
    for (size_t i = 0; i < K * N; ++i) host_B_bf16[i] = __float2bfloat16(host_B[i]);

    // Allocate device-side matrices
    size_t K_sh = K / NUM_DEVICES;
    __nv_bfloat16 *device_A_sh[NUM_DEVICES], *device_B_sh[NUM_DEVICES], *device_C[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        pglCudaMalloc(NUM_DEVICES, device_ids, dev_idx, &device_A_sh[dev_idx], M * K_sh * sizeof(__nv_bfloat16));
        pglCudaMalloc(NUM_DEVICES, device_ids, dev_idx, &device_B_sh[dev_idx], K_sh * N * sizeof(__nv_bfloat16));
        pglCudaMalloc(NUM_DEVICES, device_ids, dev_idx, &device_C[dev_idx], M * N * sizeof(__nv_bfloat16));
    }

    // Copy to device matrices
    for (size_t dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        for (size_t i = 0; i < M; ++i) {
            CUDACHECK(cudaMemcpy(device_A_sh[dev_idx] + i * K_sh,      // i-th row of device A
                                 host_A_bf16 + i * K + dev_idx * K_sh, // i-th row, dev_idx-th block of host A
                                 K_sh * sizeof(__nv_bfloat16), 
                                 cudaMemcpyHostToDevice));
        }
        // Since B is sharded row-wise, we can do a single cudaMemcpy
        CUDACHECK(cudaMemcpy(device_B_sh[dev_idx], 
                             host_B_bf16 + dev_idx * K_sh * N, 
                             K_sh * N * sizeof(__nv_bfloat16), 
                             cudaMemcpyHostToDevice));
    }
    
    // Set up layouts
    using pglobal_layout = typename mmt::layout::pglobal_layout;
    using globals = typename mmt::layout::globals;

    pglobal_layout pgl_A(device_ids, device_A_sh, nullptr, nullptr, M, K_sh);
    pglobal_layout pgl_B(device_ids, device_B_sh, nullptr, nullptr, K_sh, N);
    pglobal_layout pgl_C(device_ids, device_C, nullptr, nullptr, M, N);

    // Because gl doesn't have a default constructor and I don't want
    // the timing to include constructor calls
    globals* G = static_cast<globals*>(::operator new[](sizeof(globals) * N));
    for (int dev_idx = 0; dev_idx < N; ++dev_idx) new (&G[dev_idx]) globals{pgl_A, pgl_B, pgl_C, dev_idx};

    // Prepare kernel launch
    KittensClub club(device_ids, NUM_DEVICES);
    unsigned long smem_size = kittens::MAX_SHARED_MEMORY - 1024; // MAX_SHARED_MEMORY = 227KB for Hopper
    club.execute([&](int dev_idx) {
        CUDACHECK(cudaFuncSetAttribute(kittens::prototype::lcf::kernel<mmt>, 
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                       smem_size));
    });
    dim3 grid(mmt::grid(M, N, K_sh)); // use sharded K
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    for (int i = 0; i < NUM_WARMUPS; ++i) { // warmup
        club.execute([&](int dev_idx) { // warmup
            kittens::prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY - 1024>>>(G[dev_idx]);
            CUDACHECK(cudaDeviceSynchronize());
        });
    }

    // Start timing
    std::cout << "\n  Launching kernels with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ") on all devices\n";
    auto start = std::chrono::high_resolution_clock::now();

    // Launch!
    for (int i = 0; i < NUM_ITERS; ++i) {
        club.execute([&](int dev_idx) {
            kittens::prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY - 1024>>>(G[dev_idx]);
            CUDACHECK(cudaDeviceSynchronize());
        });
    }

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double avg_seconds = elapsed.count() / NUM_ITERS;
    std::cout << "    Execution time: " << (avg_seconds * 1e3) << " ms\n";

    // Copy & convert back to host
    __nv_bfloat16 *host_C_bf16 = new __nv_bfloat16[M * N];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        if (CHECK) {
            CUDACHECK(cudaMemcpy(host_C_bf16, device_C[dev_idx], M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
            for (int i = 0; i < M * N; ++i) host_C[dev_idx * M * N + i] += __bfloat162float(host_C_bf16[i]);
        } else {
            CUDACHECK(cudaMemcpy(host_C_bf16, device_C[dev_idx], 10 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
            for (int i = 0; i < 10; ++i) host_C[dev_idx * M * N + i] += __bfloat162float(host_C_bf16[i]);
        }

        std::cout << "  Matrix C (Host " << dev_idx << "): ";
        for (int i = 0; i < 10; ++i) std::cout << host_C[dev_idx * M * N + i] << " ";
        std::cout << "\n";
    }
    std::cout << "  Matrix C (M x N): ";
    for (int i = 0; i < 10; i++) {
        std::cout << host_C[i] << " ";
    }
    std::cout << "\n";

    // Verify result (just do first 10x10 tile)
    if (CHECK) {
        float max_error = 0.f;
        int n_error = 0;
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
            for (int i = 0; i < 10; ++i) {
                for (int j = 0; j < 10; j++) {
                    float error = std::abs(host_C[dev_idx * M * N + i * N + j] - host_C_ref[i * N + j]);
                    if (error > 3e-1) { // large due to bf16 <-> fp32 conversion
                        ++n_error;
                    }
                    max_error = std::max(max_error, error);
                }
            }
        }
        std::cout << "    Maximum error: " << max_error << "\n";
        std::cout << "    Error count (out of 10x10): " << n_error << "\n";
    }
    std::cout << "-------------------------------------------------------------\n";

    // Clean up
    delete[] host_A;
    delete[] host_A_bf16;
    delete[] host_B;
    delete[] host_B_bf16;
    delete[] host_C;
    delete[] host_C_bf16;
    delete[] host_C_ref;
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        pglCudaFree(dev_idx, device_A_sh[dev_idx], M * K_sh * sizeof(__nv_bfloat16));
        pglCudaFree(dev_idx, device_B_sh[dev_idx], K_sh * N * sizeof(__nv_bfloat16));
        pglCudaFree(dev_idx, device_C[dev_idx], M * N * sizeof(__nv_bfloat16));
        G[dev_idx].~globals();
    }
    ::operator delete[](G);
}

int main() {
    run<matmul_template<2, 4, 8>>(N, N, N);
    return 0;
}
