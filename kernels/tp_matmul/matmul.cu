/*

    Nothing useful. Just so I can understand TK better

    Original output, run from TK directory:
    --------------------  M=4096 N=4096 K=4096  --------------------
    Block size: 128x256
    Allocated host memory
    Initialized matrices
    Performed CPU matrix multiplication
    Allocated device memory
    Copied matrices to device
    Launching warmup kernel with grid (132, 1), block (384)
    Launching kernel with grid (132, 1), block (384)
    Avg Kernel execution time: 176.418 us
    Achieved performance: 779.055 TFLOPs
    Copied result back to host
    Converted result back to float
    Max error: 0.0982647
    Error count: 0

    My output (previous 32 program):
    ------------------------- Benchmark -------------------------
    M = 4096, N = 4096, K = 4096
    Block size: 128x256

    Matrix A (M x K): 0.296543 -0.316565 0.279691 0.0968502 -0.0541672 -0.400025 -0.0407511 -0.166291 -0.357133 0.150888 
    Matrix B (K x N): 0.0904346 0.178184 0.267407 0.0267467 -0.389856 0.35117 -0.260957 -0.339091 -0.398018 0.262312 
    Expected C (M x N): -1.39006 1.66512 2.44395 -8.22833 2.22921 6.12578 -7.62248 -1.25781 3.26331 3.74322 

    Launching kernel with grid (132, 1), block (384)
        Execution time: 0.176627 ms
        Performance: 778.131 TFLOPs
    Matrix C (M x N): -1.38281 1.67188 2.45312 -8.25 2.23438 6.125 -7.625 -1.26562 3.28125 3.76562 
        Maximum error: 0.0982647
        Error count: 0
    -------------------------------------------------------------

    My output (TK kernel + all_reduce):
    ------------------------- Benchmark -------------------------
    M = 4096, N = 4096, K = 4096
    Block size: 128x256

    Matrix A (M x K): 0.296543 -0.316565 0.279691 0.0968502 -0.0541672 -0.400025 -0.0407511 -0.166291 -0.357133 0.150888 
    Matrix B (K x N): 0.0904346 0.178184 0.267407 0.0267467 -0.389856 0.35117 -0.260957 -0.339091 -0.398018 0.262312 
    Expected C (M x N): -1.39006 1.66512 2.44395 -8.22833 2.22921 6.12578 -7.62248 -1.25781 3.26331 3.74322 

    Launching kernels with grid (132, 1), block (384) on all devices
        Execution time: 0.191944 ms
    Matrix C (M x N): -1.39062 1.65625 2.4375 -8.1875 2.21875 6.125 -7.625 -1.24219 3.25 3.78125 
        Maximum error: 0.272608
        Error count: 0
    -------------------------------------------------------------

*/

#include <iostream>
#include <iostream>
#include <random>

// #include "multi-gpu.cuh"

#include "kittens.cuh"
#include "prototype.cuh"
#include <cuda_bf16.h>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

using base_tile = st_bf<64, 64>;
using g_layout = gl<bf16, 1, 1, -1, -1, base_tile>;
using pglobal_layout = pgl<gl<bf16, 1, 1, -1, -1, base_tile>, true>;

constexpr int N = 1024;
constexpr int NUM_ITERS = 10;
constexpr int NUM_DEVICES = 8;
constexpr int WARPSIZE = 32;
constexpr int STRIDE = 4;

template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { 
        global_layout A, B;
        PglObj<global_layout> C_pgl; 
        int dev_idx;
        SyncSpace sync_space;
    };
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
    using gang = kittens::gang<0,1,2,3,4,5,6,7>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C_pgl.gl.rows() / (M_BLOCK*64), Cblocks = args.globals.C_pgl.gl.cols() / (N_BLOCK*64);
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
                tma::store_async(args.globals.C_pgl.gl, args.finish.c[warpgroup::groupid()][i],
                                             {args.common.coord.x, args.common.coord.y+i});
                tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
            }
            zero(args.state.accum);
            if(laneid() == 0) arrive(args.finish_finished);
            
            // gang::sync(args.globals.sync_space);
            // kittens::all_reduce_add(args.globals.C_pgl);

            // TODO: Need to figure out best way to specify tile size to perform reduction op
            // This is only done for a 64 x 64 tile, only want to reduce on that bit 
            // gang::reduce_add(args.globals.C_pgl);
        }
    };
};

__global__ void finish(PglObj<g_layout> C_pgl, SyncSpace s_m) {
    using gang = kittens::gang<0,1,2,3,4,5,6,7>;
    gang::sync(s_m); 
    kittens::all_reduce_add(C_pgl);
}

__global__ void all_reduce_bf16(kittens::bf16 *device_mat, const int N);

template<typename mmt, typename GL>
void inner_run(kittens::bf16 *device_A, kittens::bf16 *device_B, PglObj<GL> C_pgl,
    size_t M, size_t N, size_t K, dim3 grid, dim3 block, SyncSpace s_m, int dev_idx) {
        
    using global_layout = typename mmt::layout::global_layout;
    using globals = typename mmt::layout::globals;

    global_layout A_global{device_A, nullptr, nullptr, M, K};
    global_layout B_global{device_B, nullptr, nullptr, K, N};
    
    globals G{A_global, B_global, C_pgl, dev_idx, s_m};

    kittens::prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY - 1024>>>(G);
    cudaDeviceSynchronize();
    int nelem = M * N;
    int nelem_per_dev = nelem / NUM_DEVICES;
    int offset = nelem_per_dev * s_m.dev_id;
    finish<<<(nelem_per_dev + 2048 * STRIDE - 1) / (2048 * STRIDE), 256>>>(C_pgl, s_m);
   
}

// CUDA driver API
#define CUCHECK(cmd) do {                                     \
    CUresult err = cmd;                                       \
    if (err != CUDA_SUCCESS) {                                \
        const char *errStr;                                   \
        cuGetErrorString(err, &errStr);                       \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, errStr);                      \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// CUDA runtime API
#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

void init_bf16_mat(__nv_bfloat16* matrix, int size, std::mt19937& prng, 
                    std::uniform_real_distribution<>& dist) {
    for (int i = 0; i < size; ++i) {
        // Convert to BF16 immediately during initialization
        matrix[i] = __float2bfloat16(dist(prng));
        
        // Print first 10 values (convert back to float for display)
        if (i < 10)
            std::cout << __bfloat162float(matrix[i]) << " ";
    }
    std::cout << "\n";
}

template<typename mmt>
void run(size_t M, size_t N, size_t K) {
    
    std::cout << "------------------------- Benchmark -------------------------\n";
    std::cout << "  M = " << M << ", N = " << N << ", K = " << K << "\n";
    std::cout << "  Block size: " << mmt::M_BLOCK * 64 << "x" << mmt::N_BLOCK * 64 << "\n";

    // Host-side matrices
    __nv_bfloat16* host_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16* host_B_bf16 = new __nv_bfloat16[K * N];

    // Initialize A & B matrices
    std::mt19937 prng(42);
    std::uniform_real_distribution<> random(-0.5, 0.5);
    
    std::cout << "Matrix A (M x K): ";
    init_bf16_mat(host_A_bf16, M * K, prng, random);
    
    std::cout << "Matrix B (K x N): ";
    init_bf16_mat(host_B_bf16, K * N, prng, random);
    
    float *host_C_ref = new float[M * N];
    // Generate expected output (just do first 10x10 tile)
    std::cout << "  Expected C (M x N): ";
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += float(host_A_bf16[i * K + k]) * float(host_B_bf16[k * N + j]);
            }
            host_C_ref[i * N + j] = sum;
        }
    }
    for (int i = 0; i < 10; i++) {
        std::cout << host_C_ref[i] << " ";
    }
    std::cout << "\n";

    // Allocate device-side matrices
    int K_sh = K / NUM_DEVICES;
    __nv_bfloat16 *device_A_uc[NUM_DEVICES], *device_B_uc[NUM_DEVICES]; // *device_C[NUM_DEVICES]
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaMalloc(&device_A_uc[dev_idx], M * K_sh * sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&device_B_uc[dev_idx], K_sh * N * sizeof(__nv_bfloat16)));
    }

    // Copy to device matrices
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        for (int i = 0; i < M; ++i) { // TODO: do a single cudaMemcpy
            CUDACHECK(cudaMemcpy(device_A_uc[dev_idx] + i * K_sh,      // i-th row of device A
                                 host_A_bf16 + i * K + dev_idx * K_sh, // i-th row, dev_idx-th block of host A
                                 K_sh * sizeof(__nv_bfloat16), 
                                 cudaMemcpyHostToDevice));
        }
        // Since B is sharded row-wise, we can do a single cudaMemcpy
        CUDACHECK(cudaMemcpy(device_B_uc[dev_idx], 
                             host_B_bf16 + dev_idx * K_sh * N, 
                             K_sh * N * sizeof(__nv_bfloat16), 
                             cudaMemcpyHostToDevice));
    }

    /*
        Setup multimem stuff
    */
    assert(NUM_DEVICES > 1);

    // Get device_ids
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;

    __nv_bfloat16 **device_C_ptrs = new __nv_bfloat16*[NUM_DEVICES];
    CUmemGenericAllocationHandle *device_C_handles = new CUmemGenericAllocationHandle[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        pglCudaMalloc(dev_idx, &device_C_ptrs[dev_idx], &device_C_handles[dev_idx], M * N * sizeof(__nv_bfloat16));
    }
    pglobal_layout C_pgl(device_ids, NUM_DEVICES, device_C_ptrs, nullptr, nullptr, M, N);


    // Initialize parallel global layout
    KittensClub club(device_ids, NUM_DEVICES); // threadpool
    unsigned long smem_size = kittens::MAX_SHARED_MEMORY - 1024; // MAX_SHARED_MEMORY = 227KB for Hopper
    club.execute([smem_size](int dev_idx) {
        CUDACHECK(cudaFuncSetAttribute(kittens::prototype::lcf::kernel<mmt>, 
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                       smem_size));
    });

    dim3 grid(mmt::grid(M, N, K_sh)); // use sharded K
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);

    SyncManager sync_m(NUM_DEVICES, device_ids);

    for (int i = 0; i < 2; ++i) { // warmup
        club.execute([&device_A_uc, &device_B_uc, &C_pgl, &M, &N, &K_sh, &grid, &block, &sync_m](int dev_idx) { // warmup
            inner_run<mmt>(device_A_uc[dev_idx], device_B_uc[dev_idx], C_pgl.get_pgl_obj(dev_idx), M, N, K_sh, grid, block, sync_m.get_sync_space(dev_idx), dev_idx);
            CUDACHECK(cudaDeviceSynchronize());
        });
    }
    
    // Start timing
    std::cout << "\n  Launching kernels with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ") on all devices\n";
    auto start = std::chrono::high_resolution_clock::now();

    // Launch!
    for (int i = 0; i < NUM_ITERS; ++i) {
        club.execute([&device_A_uc, &device_B_uc, &C_pgl, &M, &N, &K_sh, &grid, &block, &sync_m](int dev_idx) {
            inner_run<mmt>(device_A_uc[dev_idx], device_B_uc[dev_idx], C_pgl.get_pgl_obj(dev_idx), M, N, K_sh, grid, block, sync_m.get_sync_space(dev_idx), dev_idx);
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
    float *host_C = new float[M * N];
    int random_dev_idx = 3;
    CUDACHECK(cudaSetDevice(random_dev_idx));
    CUDACHECK(cudaMemcpy(host_C_bf16, (void *)C_pgl[random_dev_idx].raw_ptr, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; ++i) host_C[i] = __bfloat162float(host_C_bf16[i]);

    std::cout << "  Matrix C (M x N): ";
    for (int i = 0; i < 10; i++) {
        std::cout << host_C[i] << " ";
    }
    std::cout << "\n";

    // Verify result (just do first 10x10 tile)
    float max_error = 0.f;
    int n_error = 0;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; j++) {
            float error = std::abs(host_C[i * N + j] - host_C_ref[i * N + j]);
            if (error > 1.0) // large due to bf16 <-> fp32 conversion
                ++n_error;
            max_error = std::max(max_error, error);
        }
    }
    std::cout << "    Maximum error: " << max_error << "\n";
    std::cout << "    Error count (out of 10x10): " << n_error << "\n";
    std::cout << "-------------------------------------------------------------\n";

    // Clean up
    delete[] host_A_bf16;
    delete[] host_B_bf16;
    delete[] host_C;
    delete[] host_C_bf16;
    delete[] host_C_ref;
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaFree(device_A_uc[dev_idx]));
        CUDACHECK(cudaFree(device_B_uc[dev_idx]));
    }

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        pglCudaFree(dev_idx, device_C_ptrs[dev_idx], device_C_handles[dev_idx], M * N * sizeof(__nv_bfloat16));
    }
}

int main() {
    run<matmul_template<2, 4, 8>>(N, N, N);
    return 0;
}
