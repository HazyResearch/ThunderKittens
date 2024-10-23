#include "strassen_prep.cu"

enum MAT_HALF {
    NORMAL = 0,
    FIRST_HALF = 1,
    SECOND_HALF = 2
};
template<int BLOCK_M, int BLOCK_N>
struct matmul_layout {
    using a_tile = st_hf<64, 64>;
    using a_tile_tma = st_hf<BLOCK_M, 64>; // we can get the whole of it with a single load
    using b_tile = st_hf<64, BLOCK_N>;
    using c_tile = st_hf<64, BLOCK_N>;
    using a_layout = gl<half, 1, 1, -1, -1, a_tile_tma>;
    using b_layout = gl<half, 1, 1, -1, -1, b_tile>;
    using c_layout = gl<half, 1, 1, -1, -1, c_tile>;
    struct globals { // global layout (here with TMA descriptors)
        a_layout a;
        b_layout b;
        c_layout c;
    };
    struct input_block {
        a_tile a[BLOCK_M/a_tile::rows];
        b_tile b;
    };
    struct consumer_state { rt_hf<16, BLOCK_N> accumulator;   };
    struct finish_block   { c_tile c[BLOCK_M/a_tile::rows]; };
};
template<int BLOCK_M, int BLOCK_N, int SUPER_M=8>
struct matmul_template {
    using layout = matmul_layout<BLOCK_M, BLOCK_N>;
    template<int C_00, int C_01, int C_10, int C_11, MAT_HALF a_offset, MAT_HALF b_offset>
    struct ker {
        static_assert(C_00 >= -1 && C_00 <= 1 && C_01 >= -1 && C_01 <= 1 && C_10 >= -1 && C_10 <= 1 && C_11 >= -1 && C_11 <= 1, "C_00, C_01, C_10, C_11 must be between -1 and 1");
        using layout = layout;
        static constexpr int NUM_CONSUMER_WARPS = BLOCK_M/16, NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS / 4, ONE_ITER=false;
        // Helper functions
        __host__ static inline dim3 grid(int M, int N, int K) {
            return dim3(M*N/(BLOCK_M*BLOCK_N)/4);
        }
        __device__ static inline bool task_coord(kittens::coord &coords, const typename layout::globals &g, int iter) {
            // internal block orderings
            int Rblocks = g.c.rows / (BLOCK_M*2), Cblocks = g.c.cols / (BLOCK_N*2);
            int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
            if (iter < super_rows * Cblocks)
                coords = { SUPER_M*(iter/super_repeat) + iter%SUPER_M,
                            (iter%super_repeat)/SUPER_M };
            else if(iter < Rblocks*Cblocks) {
                int remainder_id = iter - super_rows*Cblocks;
                coords = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
            }
            else return false;
            if(warpid() >= NUM_CONSUMER_WARPS) {
                coords = { coords.r*BLOCK_M/layout::a_tile_tma::rows, coords.c }; // producer
                int a_row_offset = (a_offset == MAT_HALF::SECOND_HALF) ? (g.a.rows/2 / layout::a_tile_tma::rows) : 0;
                int a_col_offset = (a_offset == MAT_HALF::SECOND_HALF) ? (g.a.cols/2 / layout::a_tile_tma::cols) : 0;
                int b_row_offset = (b_offset == MAT_HALF::SECOND_HALF) ? (g.b.rows/2 / layout::b_tile::rows) : 0;
                int b_col_offset = (b_offset == MAT_HALF::SECOND_HALF) ? (g.b.cols/2 / layout::b_tile::cols) : 0;
                coords = { b_row_offset, a_col_offset, coords.r + a_row_offset, coords.c + b_col_offset };
            }
            else coords = { coords.r*BLOCK_M/layout::c_tile::rows + warpgroup::groupid(), coords.c };
            return true;
        }
        __device__ static inline int iters(const typename layout::globals &g, const kittens::coord &tc) { 
            int iters = min(g.a.cols, g.b.rows) / 64;
            return iters;
        } // incorporates that sometimes we use the input matrix.
        struct producer {
            __device__ static void setup(producer_setup_args<layout> args) { // setup and load the first iteration
                warpgroup::producer_registers(); // decrease registers for the producer warpgroup
            }
            __device__ static void load(producer_load_args<layout> args) { // semaphore for the producer to load into
                if(warpgroup::warpid() == 0) {
                    tma::expect(args.inputs_arrived, args.input);
                    tma::load_async(reinterpret_cast<layout::a_tile_tma&>(args.input.a), args.globals.a, {args.task_coord.r, args.iter + args.task_coord.d}, args.inputs_arrived);
                    tma::load_async(args.input.b, args.globals.b, {args.iter + args.task_coord.b, args.task_coord.c }, args.inputs_arrived);
                    arrive(args.inputs_arrived, 3);
                }
            }
        };
        struct consumer {
            __device__ static void setup(consumer_setup_args<layout> args) { // setup locals for before the first iteration
                warpgroup::consumer_registers<NUM_CONSUMER_WARPGROUPS>();
                zero(args.state.accumulator);
            }
            __device__ static void work(consumer_work_args<layout> args) {
                warpgroup::mma_AB(args.state.accumulator, args.input.a[warpgroup::groupid()], args.input.b);
                warpgroup::mma_async_wait();
                arrive(args.inputs_finished);
            }
            __device__ static void finish(consumer_finish_args<layout> args) {
                int row_offset_blocks = (args.globals.c.rows / layout::c_tile::rows) / 2, col_offset_blocks = (args.globals.c.cols / layout::c_tile::cols) / 2;
                warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accumulator);
                warpgroup::sync();
                if(warpgroup::warpid() == 0) {
                    if constexpr (C_00 == 1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {args.task_coord.r, args.task_coord.c});
                    }
                    if constexpr (C_01 == 1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {args.task_coord.r, args.task_coord.c + col_offset_blocks});
                    }
                    if constexpr (C_10 == 1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {args.task_coord.r + row_offset_blocks, args.task_coord.c});
                    }
                    if constexpr (C_11 == 1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {args.task_coord.r + row_offset_blocks, args.task_coord.c + col_offset_blocks});
                    }
                }
                tma::store_async_read_wait(); // make sure the store is finished before we start writing back
                warpgroup::sync();
                if constexpr (C_00 >= 0 && C_01 >= 0 && C_10 >= 0 && C_11 >= 0) {
                    arrive(args.finish_finished);
                    return;
                }
                warpgroup::mul(args.finish.c[warpgroup::groupid()], args.finish.c[warpgroup::groupid()], -1.f);
                warpgroup::sync();
                if(warpgroup::warpid() == 0) {
                    if constexpr (C_00 == -1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {args.task_coord.r, args.task_coord.c});
                    }
                    if constexpr (C_01 == -1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {args.task_coord.r, args.task_coord.c + col_offset_blocks});
                    }
                    if constexpr (C_10 == -1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {args.task_coord.r + row_offset_blocks, args.task_coord.c});
                    }
                    if constexpr (C_11 == -1) {
                        tma::store_add_async(args.globals.c, args.finish.c[warpgroup::groupid()],
                        {args.task_coord.r + row_offset_blocks, args.task_coord.c + col_offset_blocks});
                    }
                }
                tma::store_async_read_wait(); // make sure the store is finished before we start writing back
                warpgroup::sync();
                arrive(args.finish_finished);
            }
        };
    };
};

#include <assert.h>

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

template<int C00, int C01, int C10, int C11>
void cpu_strassen(float *a_prep, float *b_prep, float* c_mat,
                  int A_rows, int A_cols, int B_rows, int B_cols, int C_rows, int C_cols,
                  MAT_HALF a_offset, MAT_HALF b_offset) {
    int a_row_offset = (a_offset == MAT_HALF::SECOND_HALF) ? A_rows/2 : 0;
    int a_col_offset = (a_offset == MAT_HALF::SECOND_HALF) ? A_cols/2 : 0;
    int b_row_offset = (b_offset == MAT_HALF::SECOND_HALF) ? B_rows/2 : 0;
    int b_col_offset = (b_offset == MAT_HALF::SECOND_HALF) ? B_cols/2 : 0;
    int c_row_offset = C_rows/2, c_col_offset = C_cols/2;
    int K_loop = (a_offset != MAT_HALF::NORMAL) ? A_cols/2 : A_cols; // if not normal, we need a half-sized loop.
    if(a_offset != MAT_HALF::NORMAL && b_offset != MAT_HALF::NORMAL) { // Neither normal, which is to say all are inputs. This never happens in Strassen.
        assert(A_rows == C_rows);
        assert(A_cols == B_rows);
        assert(B_cols == C_cols);
        assert(false); // should not get here, in any case.
    }
    else if(a_offset != MAT_HALF::NORMAL) { // B normal
        assert(A_rows == C_rows);
        assert(A_cols == B_rows);
        assert(B_cols*2 == C_cols);
    }
    else if(b_offset != MAT_HALF::NORMAL) { // A normal
        assert(A_rows*2 == C_rows);
        assert(A_cols == B_rows);
        assert(B_cols == C_cols);
    }
    else {
        assert(A_rows*2 == C_rows);
        assert(A_cols == B_rows);
        assert(B_cols*2 == C_cols);
    }
    static_assert(C00 >= -1 && C00 <= 1 && C01 >= -1 && C01 <= 1 && C10 >= -1 && C10 <= 1 && C11 >= -1 && C11 <= 1, "C_00, C_01, C_10, C_11 must be between -1 and 1");
    for(int i = 0; i < C_rows/2; i++) {
        for(int j = 0; j < C_cols/2; j++) {
            for(int k = 0; k < K_loop; k++) {
                float a = a_prep[(i + a_row_offset) * A_cols + (k + a_col_offset)];
                float b = b_prep[(k + b_row_offset) * B_cols + (j + b_col_offset)];
                float mini = a*b;
                if constexpr (C00 == 1) c_mat[i * C_cols + j] += mini;
                if constexpr (C01 == 1) c_mat[i * C_cols + j + c_col_offset] += mini;
                if constexpr (C10 == 1) c_mat[(i + c_row_offset) * C_cols + j] += mini;
                if constexpr (C11 == 1) c_mat[(i + c_row_offset) * C_cols + j + c_col_offset] += mini;
                if constexpr (C00 == -1) c_mat[i * C_cols + j] -= mini;
                if constexpr (C01 == -1) c_mat[i * C_cols + j + c_col_offset] -= mini;
                if constexpr (C10 == -1) c_mat[(i + c_row_offset) * C_cols + j] -= mini;
                if constexpr (C11 == -1) c_mat[(i + c_row_offset) * C_cols + j + c_col_offset] -= mini;
            }
        }
    }
}

template<typename mmt, typename T=half>
int verify_strassen(int M, int N, int K) {

    std::cout << " ---------------------------------------------------------------------------------" << std::endl;
    std::cout << " Running Strassen with M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << " ---------------------------------------------------------------------------------" << std::endl;

    // Allocate host memory
    float *h_A_cpu = new float[M * K];
    float *h_B_cpu = new float[K * N];
    float *h_A_prep_cpu = new float[5 * (M/2) * (K/2)];
    float *h_B_prep_cpu = new float[5 * (K/2) * (N/2)];
    float *h_C_cpu = new float[M * N];
    float *h_C_ref = new float[M * N];

    // Initialize matrix with random values
    std::random_device rd;
    std::mt19937 gen(42);
    float range = 1.f / K;
    range = sqrt(sqrt(range));
    std::uniform_real_distribution<float> dis(-range, range);

    static_assert(std::is_same_v<T, half>);
    for (int i = 0; i < M * K; ++i) {
        h_A_cpu[i] = dis(gen);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B_cpu[i] = dis(gen);
    }

    std::cout << "Initialized matrices" << std::endl;

    // strassen_prep_cpu(h_A_cpu, h_A_prep_cpu, M, K);
    // strassen_prep_cpu(h_B_cpu, h_B_prep_cpu, K, N);

    int a_slice_offset = (M/2)*(K/2);
    int b_slice_offset = (K/2)*(N/2);
    
    // do normal gemm on h_C_ref
    // cpu_gemm(h_A_cpu, h_B_cpu, h_C_ref, M, N, K);

    // zero out h_C_ref
    for(int i = 0; i < M*N; i++) h_C_cpu[i] = 0.0f;

    // Allocate device memory for input and output matrices
    half *d_A, *d_B, *d_A_prep, *d_B_prep, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_A_prep, 5 * (M/2) * (K/2) * sizeof(half));
    cudaMalloc(&d_B_prep, 5 * (K/2) * (N/2) * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));

    // Convert float arrays to half arrays
    half *h_A_half = new half[M * K];
    half *h_B_half = new half[K * N];
    half *h_A_prep_half = new half[5 * (M/2) * (K/2)];
    half *h_B_prep_half = new half[5 * (K/2) * (N/2)];

    for (int i = 0; i < M * K; ++i) h_A_half[i] = __float2half(h_A_cpu[i]);
    for (int i = 0; i < K * N; ++i) h_B_half[i] = __float2half(h_B_cpu[i]);
    for (int i = 0; i < 5 * (M/2) * (K/2); ++i) h_A_prep_half[i] = __float2half(h_A_prep_cpu[i]);
    for (int i = 0; i < 5 * (K/2) * (N/2); ++i) h_B_prep_half[i] = __float2half(h_B_prep_cpu[i]);

    std::cout << "Converted matrices to half precision" << std::endl;

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A_half, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_half, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_prep, h_A_prep_half, 5 * (M/2) * (K/2) * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_prep, h_B_prep_half, 5 * (K/2) * (N/2) * sizeof(half), cudaMemcpyHostToDevice);

    // Free temporary host memory
    delete[] h_A_half;
    delete[] h_B_half;
    delete[] h_A_prep_half;
    delete[] h_B_prep_half;

    // Initialize d_C to zero (optional, depending on your CUDA kernel implementation)
    cudaMemset(d_C, 0, M * N * sizeof(half));

    std::cout << "Initialized d_C to zero" << std::endl;

    // Call the GPU kernel (assuming it's implemented elsewhere)
    using pk = prep_ker<2048>;
    pk::layout::x_layout A_prep_X(d_A, nullptr, nullptr, M, K);
    pk::layout::y_layout A_prep_Y(d_A_prep, nullptr, nullptr, M/2, K/2);
    pk::layout::x_layout B_prep_X(d_B, nullptr, nullptr, K, N);
    pk::layout::y_layout B_prep_Y(d_B_prep, nullptr, nullptr, K/2, N/2);
    pk::layout::globals A_prep_G{A_prep_X, A_prep_Y};
    pk::layout::globals B_prep_G{B_prep_X, B_prep_Y};
    unsigned long prep_mem_size = 226000; // Adjust if needed
    cudaFuncSetAttribute(prototype::pc<pk>, cudaFuncAttributeMaxDynamicSharedMemorySize, prep_mem_size);
    dim3 prep_grid(132); // Adjust if needed
    dim3 prep_block(256);

    prototype::pc<pk><<<prep_grid, prep_block, prep_mem_size>>>(A_prep_G);
    prototype::pc<pk><<<prep_grid, prep_block, prep_mem_size>>>(B_prep_G);

    // Check for CUDA errors after prep kernels
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Prep kernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Clean up and return or exit
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_A_prep);
        cudaFree(d_B_prep);
        cudaFree(d_C);
        return -1; // or exit(1);
    }

    // Lots of using's
    using layout = typename mmt::layout;
    using globals = typename layout::globals;
    using a_layout = typename layout::a_layout;
    using b_layout = typename layout::b_layout;
    using c_layout = typename layout::c_layout;

    using ker1 = mmt::template ker< 1, 0, 0,  1, MAT_HALF::NORMAL,      MAT_HALF::NORMAL>;
    using ker2 = mmt::template ker< 0, 0, 1, -1, MAT_HALF::NORMAL,      MAT_HALF::FIRST_HALF>;
    using ker3 = mmt::template ker< 0, 1, 0,  1, MAT_HALF::FIRST_HALF,  MAT_HALF::NORMAL>;
    using ker4 = mmt::template ker< 1, 0, 1,  0, MAT_HALF::SECOND_HALF, MAT_HALF::NORMAL>;
    using ker5 = mmt::template ker<-1, 1, 0,  0, MAT_HALF::NORMAL,      MAT_HALF::SECOND_HALF>;
    using ker6 = mmt::template ker< 0, 0, 0,  1, MAT_HALF::NORMAL,      MAT_HALF::NORMAL>;
    using ker7 = mmt::template ker< 1, 0, 0,  0, MAT_HALF::NORMAL,      MAT_HALF::NORMAL>;

    a_layout A_big{d_A, nullptr, nullptr, M, K};
    b_layout B_big{d_B, nullptr, nullptr, K, N};
    c_layout C_big{d_C, nullptr, nullptr, M, N};
    
    a_layout A_P0{d_A_prep, nullptr, nullptr, M/2, K/2};
    a_layout A_P1{d_A_prep + 1*a_slice_offset, nullptr, nullptr, M/2, K/2};
    a_layout A_P2{d_A_prep + 2*a_slice_offset, nullptr, nullptr, M/2, K/2};
    a_layout A_P3{d_A_prep + 3*a_slice_offset, nullptr, nullptr, M/2, K/2};
    a_layout A_P4{d_A_prep + 4*a_slice_offset, nullptr, nullptr, M/2, K/2};

    b_layout B_P0{d_B_prep, nullptr, nullptr, K/2, N/2};
    b_layout B_P1{d_B_prep + 1*b_slice_offset, nullptr, nullptr, K/2, N/2};
    b_layout B_P2{d_B_prep + 2*b_slice_offset, nullptr, nullptr, K/2, N/2};
    b_layout B_P3{d_B_prep + 3*b_slice_offset, nullptr, nullptr, K/2, N/2};
    b_layout B_P4{d_B_prep + 4*b_slice_offset, nullptr, nullptr, K/2, N/2};

    unsigned long mem_size = 113000*2;
    cudaFuncSetAttribute(prototype::pc<ker1>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    cudaFuncSetAttribute(prototype::pc<ker2>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    cudaFuncSetAttribute(prototype::pc<ker3>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    cudaFuncSetAttribute(prototype::pc<ker4>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    cudaFuncSetAttribute(prototype::pc<ker5>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    cudaFuncSetAttribute(prototype::pc<ker6>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    cudaFuncSetAttribute(prototype::pc<ker7>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    globals G1{A_P0,  B_P0,  C_big};
    globals G2{A_P1,  B_big, C_big};
    globals G3{A_big, B_P4,  C_big};
    globals G4{A_big, B_P3,  C_big};
    globals G5{A_P2,  B_big, C_big};
    globals G6{A_P3,  B_P2,  C_big};
    globals G7{A_P4,  B_P1,  C_big};

    dim3 grid = ker1::grid(M, N, K); // rows, cols
    dim3 block = dim3(prototype::num_threads<ker1>);

    std::cout << "Grid dimensions: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    std::cout << "Block dimensions: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;

    cudaDeviceSynchronize(); std::cout << "starting ker1" << std::endl;
    prototype::pc<ker1><<<grid, block, mem_size>>>(G1);
    cudaDeviceSynchronize(); std::cout << "starting ker2" << std::endl;
    prototype::pc<ker2><<<grid, block, mem_size>>>(G2);
    cudaDeviceSynchronize(); std::cout << "starting ker3" << std::endl;
    prototype::pc<ker3><<<grid, block, mem_size>>>(G3);
    cudaDeviceSynchronize(); std::cout << "starting ker4" << std::endl;
    prototype::pc<ker4><<<grid, block, mem_size>>>(G4);
    cudaDeviceSynchronize(); std::cout << "starting ker5" << std::endl;
    prototype::pc<ker5><<<grid, block, mem_size>>>(G5);
    cudaDeviceSynchronize(); std::cout << "starting ker6" << std::endl;
    prototype::pc<ker6><<<grid, block, mem_size>>>(G6);
    cudaDeviceSynchronize(); std::cout << "starting ker7" << std::endl;
    prototype::pc<ker7><<<grid, block, mem_size>>>(G7);
    cudaDeviceSynchronize();

    // Copy GPU output back to CPU
    half *h_C_gpu = new half[M * N];
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    // Convert half precision to float for comparison
    float *h_C_gpu_float = new float[M * N];
    for (int i = 0; i < M * N; ++i) {
        h_C_gpu_float[i] = __half2float(h_C_gpu[i]);
    }

    // Update the comparison loop to use h_C_gpu_float instead of h_C_cpu
    double max_diff = 0.0;
    double avg_diff = 0.0;
    int wrong_count = 0;
    for (int i = 0; i < M * N; ++i) {
        double diff = std::abs(h_C_gpu_float[i] - h_C_ref[i]);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
        if (diff > 0.1) {
            if (wrong_count < 10) {
                int row = i / N;
                int col = i % N;
                std::cout << "GPU Mismatch at (" << row << ", " << col << "): "
                          << "Strassen=" << h_C_gpu_float[i] << ", Ref=" << h_C_ref[i] 
                          << ", diff=" << diff << std::endl;
            }
            wrong_count++;
        }
    }
    avg_diff /= M*N;
    std::cout << "Max difference: " << max_diff << std::endl;
    std::cout << "Average difference: " << avg_diff << std::endl;
    std::cout << "Number of elements exceeding tolerance: " << wrong_count << std::endl;

    // Time the mf
    cudaDeviceSynchronize();
    std::cout << "Timing the kernels:" << std::endl;
    constexpr int ITERS = 10;
    double total_prep_us = 0, total_ker_us = 0;
    for(int i = 0; i < ITERS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        prototype::pc<pk><<<prep_grid, prep_block, mem_size>>>(A_prep_G);
        prototype::pc<pk><<<prep_grid, prep_block, mem_size>>>(B_prep_G);
        cudaDeviceSynchronize();
        auto mid = std::chrono::high_resolution_clock::now();
        prototype::pc<ker1><<<grid, block, mem_size>>>(G1);
        prototype::pc<ker2><<<grid, block, mem_size>>>(G2);
        prototype::pc<ker3><<<grid, block, mem_size>>>(G3);
        prototype::pc<ker4><<<grid, block, mem_size>>>(G4);
        prototype::pc<ker5><<<grid, block, mem_size>>>(G5);
        prototype::pc<ker6><<<grid, block, mem_size>>>(G6);
        prototype::pc<ker7><<<grid, block, mem_size>>>(G7);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> prep_diff = mid - start;
        double prep_us = prep_diff.count() * 1e6;
        std::chrono::duration<double> ker_diff = end - mid;
        double ker_us = ker_diff.count() * 1e6;
        total_prep_us += prep_us;
        total_ker_us += ker_us;
    }
    double avg_prep_us = total_prep_us / ITERS;
    double avg_ker_us = total_ker_us / ITERS;
    std::cout << "Average prep time: " << avg_prep_us << " us" << std::endl;
    std::cout << "Average ker time: " << avg_ker_us << " us" << std::endl;
    std::cout << "Total time: " << avg_prep_us + avg_ker_us << " us" << std::endl;
    std::cout << "Effective TFLOPs: " << 2.0 * M * N * K / (avg_prep_us + avg_ker_us) / 1e6 << " TFLOPs" << std::endl;
    std::cout << "Actual TFLOPs: " << 1.75 * M * N * K / (avg_prep_us + avg_ker_us) / 1e6 << " TFLOPs" << std::endl;
    std::cout << "Effective TFLOPs without prep: " << 2.0 * M * N * K / avg_ker_us / 1e6 << " TFLOPs" << std::endl;
    std::cout << "Actual TFLOPs without prep: " << 1.75 * M * N * K / avg_ker_us / 1e6 << " TFLOPs" << std::endl;

    // Update the print statement to use h_C_gpu_float
    std::cout << "C_gpu: ";
    for(int i = 0; i < 20; i++) std::cout << h_C_gpu_float[i] << " ";
    std::cout << std::endl;
    std::cout << "C_ref: ";
    for(int i = 0; i < 20; i++) std::cout << h_C_ref[i] << " ";
    std::cout << std::endl;

    // // Compare results
    // max_diff = 0.0;
    // wrong_count = 0;
    // avg_diff = 0.0;
    // for (int i = 0; i < M * N; ++i) {
    //     double diff = std::abs(h_C_cpu[i] - h_C_ref[i]);
    //     max_diff = std::max(max_diff, diff);
    //     avg_diff += diff;
    //     if (diff > 0.1) {
    //         if (wrong_count < 10) {
    //             int row = i / N;
    //             int col = i % N;
    //             std::cout << "CPU Mismatch at (" << row << ", " << col << "): "
    //                       << "Strassen=" << h_C_cpu[i] << ", Ref=" << h_C_ref[i] 
    //                       << ", diff=" << diff << std::endl;
    //         }
    //         wrong_count++;
    //     }
    // }
    // avg_diff /= M*N;
    // std::cout << "Max difference: " << max_diff << std::endl;
    // std::cout << "Average difference: " << avg_diff << std::endl;
    // std::cout << "Number of elements exceeding tolerance: " << wrong_count << std::endl;

    // // print the first 20 elements of each output matrix
    // std::cout << "C_cpu: ";
    // for(int i = 0; i < 20; i++) std::cout << h_C_cpu[i] << " ";
    // std::cout << std::endl;
    // std::cout << "C_ref: ";
    // for(int i = 0; i < 20; i++) std::cout << h_C_ref[i] << " ";
    // std::cout << std::endl;

    // Clean up
    delete[] h_A_cpu;
    delete[] h_B_cpu;
    delete[] h_A_prep_cpu;
    delete[] h_B_prep_cpu;
    delete[] h_C_cpu;
    delete[] h_C_ref;

    return wrong_count;
}

int main() {
    // int m = 18432, n = 16896, k = 16896;
    int m = 16384, n = 16384, k = 16384;
    verify_strassen<matmul_template<256, 256, 8>>(m, n, k);
    verify_strassen<matmul_template<256, 256, 12>>(m, n, k);
}



// int main() {
//     const size_t BIG_M = 256 * 12 * 2, BIG_N = 256 * 11 * 2, BIG_K = 8192 * 2 * 2;
//     const size_t M = BIG_M / 2, N = BIG_N / 2, K = BIG_K / 2; using mmt = matmul_template<256, 256, 8>; // 813.5 TFLOPs

//     using a_tile   = typename std::remove_reference<decltype(std::declval<typename mmt::layout::input_block>().a[0])>::type;
//     using b_tile   = typename std::remove_reference<decltype(std::declval<typename mmt::layout::input_block>().b)>::type;
//     using c_tile   = typename std::remove_reference<decltype(std::declval<typename mmt::layout::finish_block>().c[0])>::type;
//     using a_global = typename std::remove_reference<decltype(std::declval<typename mmt::layout::globals>().a)>::type;
//     using b_global = typename std::remove_reference<decltype(std::declval<typename mmt::layout::globals>().b)>::type;
//     using c_global = typename std::remove_reference<decltype(std::declval<typename mmt::layout::globals>().c)>::type;
//     using globals  = typename mmt::layout::globals;

//     std::cout << "BIG_M=" << BIG_M << " BIG_N=" << BIG_N << " BIG_K=" << BIG_K << std::endl;
//     std::cout << "breaking down into M=" << M << " N=" << N << " K=" << K << std::endl;

//     std::cout << "Has store: "  << (bool)kittens::prototype::detail::has_store<mmt> << '\n';
//     std::cout << "Has finish: " << (bool)kittens::prototype::detail::has_finish<mmt> << '\n';

//     // Allocate host memory
//     float *h_A = new float[M * K];
//     float *h_B = new float[K * N];
//     float *h_C = new float[M * N];
//     float *h_C_ref = new float[M * N];

//     std::cout << "Allocated host memory" << std::endl;

//     // Initialize random number generator
//     std::random_device rd;
//     std::mt19937 gen(42);
//     std::uniform_real_distribution<> dis(-1.0, 1.0);

//     // Initialize matrices with random values
//     for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
//     for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);

//     std::cout << "Initialized matrices" << std::endl;

//     // Perform CPU matrix multiplication for reference
//     // cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

//     std::cout << "Performed CPU matrix multiplication" << std::endl;

//     half *d_prep_X, *d_prep_Y;
//     cudaMalloc(&d_prep_X, BIG_M*BIG_K*2);
//     cudaMalloc(&d_prep_Y, M*K*10);

//     std::cout << "Allocated prep memory" << std::endl;
//     std::cout << "Timing prep kernel:" << std::endl;
//     using prep_ker = prep_ker;
//     prep_ker::layout::x_layout Xg(d_prep_X, nullptr, nullptr, BIG_M, BIG_K);
//     prep_ker::layout::y_layout Yg(d_prep_Y, nullptr, nullptr, M, K);
//     prep_ker::layout::globals prep_G{Xg, Yg};

//     unsigned long prep_mem_size = 113000*2; // need to launch two blocks if possible.
//     cudaFuncSetAttribute(prototype::pc<prep_ker>, cudaFuncAttributeMaxDynamicSharedMemorySize, prep_mem_size);
//     dim3 prep_grid(132); // rows, cols
//     dim3 prep_block(256);

//     for(int i = 0; i < 2; i++) {
//         prototype::pc<prep_ker><<<prep_grid, prep_block, prep_mem_size>>>(prep_G);
//     }
//     cudaDeviceSynchronize();

//     std::cout << "Finished warmup!" << std::endl;
//     const int PREP_ITERS = 1;
//     auto prep_start = std::chrono::high_resolution_clock::now();
//     for(int i = 0; i < PREP_ITERS; i++) {
//         prototype::pc<prep_ker><<<prep_grid, prep_block, prep_mem_size>>>(prep_G);
//     }
//     cudaDeviceSynchronize();
//     auto prep_end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> prep_diff = prep_end - prep_start;
//     double prep_us = prep_diff.count()*1e6 / PREP_ITERS;
//     std::cout << "Prep kernel execution time (per iter): " << prep_us << " us\n";
//     // Check for CUDA errors
//     cudaError_t cudaStatus = cudaGetLastError();
//     if (cudaStatus != cudaSuccess) {
//         std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
//         // Optionally, you might want to exit the program or handle the error in some way
//         return -1;
//     }

//     cudaFree(d_prep_X);
//     cudaFree(d_prep_Y);

//     // // Allocate device memory
//     half *d_A, *d_B, *d_C;
//     cudaMalloc(&d_A, M*K*2);
//     cudaMalloc(&d_B, K*N*2);
//     cudaMalloc(&d_C, BIG_M*BIG_N*2);

//     std::cout << "Allocated device memory" << std::endl;

//     std::cout << "a_tile::rows=" << a_tile::rows << " a_tile::cols=" << a_tile::cols << std::endl;
//     std::cout << "b_tile::rows=" << b_tile::rows << " b_tile::cols=" << b_tile::cols << std::endl;
//     std::cout << "c_tile::rows=" << c_tile::rows << " c_tile::cols=" << c_tile::cols << std::endl;
//     a_global Ag{d_A, nullptr, nullptr, M, K};
//     b_global Bg{d_B, nullptr, nullptr, K, N};
//     c_global Cg{d_C, nullptr, nullptr, BIG_M, BIG_N};
//     globals G{Ag, Bg, Cg};

//     std::cout << "Allocated memory" << std::endl;

//     // Convert to 16-bit and copy to device
//     half *h_A_half = new half[M * K];
//     half *h_B_half = new half[K * N];
//     for (int i = 0; i < M * K; ++i) h_A_half[i] = __float2half(h_A[i]);
//     for (int i = 0; i < K * N; ++i) h_B_half[i] = __float2half(h_B[i]);

//     cudaMemcpy(d_A, h_A_half, M*K*2, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B_half, K*N*2, cudaMemcpyHostToDevice);

//     std::cout << "Copied matrices to device" << std::endl;

//     unsigned long mem_size = MAX_SHARED_MEMORY; // need to launch two blocks if possible.

//     using ker = mmt::template ker<1, 0, 0, 0>;
    
//     cudaFuncSetAttribute(prototype::pc<ker>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
//     // Launch kernel
//     dim3 grid = ker::grid(M, N, K); // rows, cols
//     dim3 block(prototype::num_threads<ker>);

//     // Start timing
//     cudaDeviceSynchronize();
//     std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << "), and " << K/a_tile::cols << " reduction block dimension\n";
//     std::cout << "Kernel has " << kittens::prototype::input_pipe_stages<mmt> << " input pipeline stages and " << kittens::prototype::output_pipe_stages<mmt> << " output pipeline stages\n";

//     constexpr int ITERS = 1;
//     for(int i = 0; i < 5; i++) {
//         prototype::pc<ker><<<grid, block, mem_size>>>(G);
//     }
//     cudaDeviceSynchronize();

//     std::cout << "Finished warmup!" << std::endl;
//     auto start = std::chrono::high_resolution_clock::now();
//     for(int i = 0; i < ITERS*7; i++) {
//         prototype::pc<ker><<<grid, block, mem_size>>>(G);
//     }
//     cudaDeviceSynchronize();
//     // End timing
//     auto end = std::chrono::high_resolution_clock::now();

//     // Calculate duration
//     std::chrono::duration<double> diff = end - start;
//     double us = diff.count()*1e6 / ITERS;

//     // Calculate TFLOPs
//     double flops = double(2.0) * BIG_M * BIG_N * BIG_K * ITERS; // 2 FLOPs per multiply-add
//     double expected_tax = (double(BIG_M+BIG_N)*BIG_K*2 + 5*2*((M+N)*K)) / 3.3e12 * 1e6;
//     double tflops = (flops / us) / 1e6;
//     double expected_taxed_tflops = (flops / (expected_tax+us)) / 1e6;
//     double taxed_tflops = (flops / (prep_us*2+us)) / 1e6;

//     std::cout << "Kernel execution time: " << us << " us\n";
//     std::cout << "Best possible pre-cost: " << expected_tax << " us\n";
//     std::cout << "Actual pre-cost: " << prep_us*2 << " us\n";
//     std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
//     std::cout << "Best taxed performance: " << expected_taxed_tflops << " TFLOPs\n";
//     std::cout << "Actual taxed performance: " << taxed_tflops << " TFLOPs\n";
//     std::cout << "Naive performance: " << tflops*7/8 << " TFLOPs\n";
//     // Check for CUDA errors
//     cudaStatus = cudaGetLastError();
//     if (cudaStatus != cudaSuccess) {
//         std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
//         // Optionally, you might want to exit the program or handle the error in some way
//         return -1;
//     }

//     // Copy result back to host
//     half *h_C_half = new half[M * N];
//     cudaMemcpy(h_C_half, d_C, M*N*2, cudaMemcpyDeviceToHost);

//     std::cout << "Copied result back to host" << std::endl;

//     // Convert result back to float for comparison
//     for (int i = 0; i < M * N; ++i) h_C[i] = __half2float(h_C_half[i]);

//     std::cout << "Converted result back to float" << std::endl;

//     // // Check result
//     // float max_error = 0.0f;
//     // int error_count = 0;
//     // for (int i = 0; i < M * N; ++i) {
//     //     float error = std::abs(h_C[i] - h_C_ref[i]);
//     //     if(error > 1.0) { // large because of half vs fp32 numerics
//     //         if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
//     //         else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
//     //         error_count++;
//     //     }
//     //     max_error = std::max(max_error, error);
//     // }

//     // std::cout << "Max error: " << max_error << std::endl;
//     // std::cout << "Error count: " << error_count << std::endl;

//     // Clean up
//     delete[] h_A;
//     delete[] h_B;
//     delete[] h_C;
//     delete[] h_C_ref;
//     delete[] h_A_half;
//     delete[] h_B_half;
//     delete[] h_C_half;
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     return 0;
// }


    // cpu_strassen<1, 0, 0, 1>( // M1
    //     h_A_prep_cpu + 0*a_slice_offset, 
    //     h_B_prep_cpu + 0*b_slice_offset, 
    //     h_C_cpu,
    //     M/2, K/2,
    //     K/2, N/2,
    //     M, N,
    //     MAT_HALF::NORMAL, MAT_HALF::NORMAL
    // );
    // cpu_strassen<0, 0, 1, -1>( // M2
    //     h_A_prep_cpu + 1*a_slice_offset, 
    //     h_B_cpu, 
    //     h_C_cpu,
    //     M/2, K/2,
    //     K, N,
    //     M, N,
    //     MAT_HALF::NORMAL, MAT_HALF::FIRST_HALF
    // );
    // cpu_strassen<0, 1, 0, 1>( // M3
    //     h_A_cpu, 
    //     h_B_prep_cpu + 4*b_slice_offset, 
    //     h_C_cpu,
    //     M, K,
    //     K/2, N/2,
    //     M, N,
    //     MAT_HALF::FIRST_HALF, MAT_HALF::NORMAL
    // );
    // cpu_strassen<1, 0, 1, 0>( // M4
    //     h_A_cpu, 
    //     h_B_prep_cpu + 3*b_slice_offset, 
    //     h_C_cpu,
    //     M, K,
    //     K/2, N/2,
    //     M, N,
    //     MAT_HALF::SECOND_HALF, MAT_HALF::NORMAL
    // );
    // cpu_strassen<-1, 1, 0, 0>( // M5
    //     h_A_prep_cpu + 2*a_slice_offset, 
    //     h_B_cpu,
    //     h_C_cpu,
    //     M/2, K/2,
    //     K, N,
    //     M, N,
    //     MAT_HALF::NORMAL, MAT_HALF::SECOND_HALF
    // );
    // cpu_strassen<0, 0, 0, 1>( // M6
    //     h_A_prep_cpu + 3*a_slice_offset, 
    //     h_B_prep_cpu + 2*b_slice_offset, 
    //     h_C_cpu,
    //     M/2, K/2,
    //     K/2, N/2,
    //     M, N,
    //     MAT_HALF::NORMAL, MAT_HALF::NORMAL
    // );
    // cpu_strassen<1, 0, 0, 0>( // M7
    //     h_A_prep_cpu + 4*a_slice_offset, 
    //     h_B_prep_cpu + 1*b_slice_offset, 
    //     h_C_cpu,
    //     M/2, K/2,
    //     K/2, N/2,
    //     M, N,
    //     MAT_HALF::NORMAL, MAT_HALF::NORMAL
    // );