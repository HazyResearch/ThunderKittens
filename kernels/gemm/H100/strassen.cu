


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
    struct globals { // global layout (here with TMA descriptors)
        gl<half, 1, 1, -1, -1, a_tile_tma> a;
        gl<half, 1, 1, -1, -1, b_tile>     b;
        gl<half, 1, 1, -1, -1, c_tile>     c;
        MAT_HALF a_offset, b_offset; // 1 if we are pulling from the second half of a twice-as-large tile
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
    template<int C_00, int C_01, int C_10, int C_11>
    struct ker {
        static_assert(C_00 >= -1 && C_00 <= 1 && C_01 >= -1 && C_01 <= 1 && C_10 >= -1 && C_10 <= 1 && C_11 >= -1 && C_11 <= 1, "C_00, C_01, C_10, C_11 must be between -1 and 1");
        using layout = layout;
        static constexpr int NUM_CONSUMER_WARPS = BLOCK_M/16, NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS / 4;
        // Helper functions
        __host__ static inline dim3 grid(int M, int N, int K) {
            return dim3(M*N/(BLOCK_M*BLOCK_N));
        }
        __device__ static inline bool task_coord(kittens::coord &coords, const typename layout::globals &g, int iter) {
            // internal block orderings
            int Rblocks = g.c.rows / (BLOCK_M*2), Cblocks = g.c.cols / (BLOCK_N*2);
            int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
            if (blockIdx.x < super_rows * Cblocks)
                coords = { SUPER_M*(blockIdx.x/super_repeat) + blockIdx.x%SUPER_M,
                            (blockIdx.x%super_repeat)/SUPER_M };
            else {
                int remainder_id = blockIdx.x - super_rows*Cblocks;
                coords = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
            }
            if(warpid() >= NUM_CONSUMER_WARPS) {
                coords = { coords.r*BLOCK_M/layout::a_tile_tma::rows, coords.c }; // producer
                int a_row_offset = (g.a_offset == MAT_HALF::SECOND_HALF) ? (g.c.rows / layout::a_tile_tma::rows) : 0;
                int a_col_offset = (g.a_offset == MAT_HALF::SECOND_HALF) ? (g.c.cols / layout::a_tile_tma::cols) : 0;
                int b_row_offset = (g.b_offset == MAT_HALF::SECOND_HALF) ? (g.c.rows / layout::b_tile::rows) : 0;
                int b_col_offset = (g.b_offset == MAT_HALF::SECOND_HALF) ? (g.c.cols / layout::b_tile::cols) : 0;
                coords = { b_row_offset, a_col_offset, coords.r + a_row_offset, coords.c + b_col_offset };
            }
            else                               coords = { coords.r*BLOCK_M/layout::c_tile::rows + warpgroup::groupid(), coords.c };
            return iter < gridDim.x*gridDim.y*gridDim.z;
        }
        __device__ static inline int iters(const typename layout::globals &g, const kittens::coord &tc) { return min(g.a.cols, g.b.rows) / 64; } // incorporates that sometimes we use the input matrix.
        struct producer {
            __device__ static void setup(producer_setup_args<layout> args) { // setup and load the first iteration
                warpgroup::producer_registers(); // decrease registers for the producer warpgroup
            }
            __device__ static void load(producer_load_args<layout> args) { // barrier for the producer to load into
                if(warpgroup::warpid() == args.iter%4) {
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
                if(C_00 >= 0 && C_01 >= 0 && C_10 >= 0 && C_11 >= 0) return;
                mul(args.state.accumulator, args.state.accumulator, -1.f);
                tma::store_async_read_wait(); // make sure the store is finished before we start writing back
                warpgroup::sync();
                warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accumulator);
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
            }
        };
    };
};


// void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
//     #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             float sum = 0.0f;
//             for (int k = 0; k < K; k++) {
//                 sum += a[i * K + k] * b[k * N + j];
//             }
//             c[i * N + j] = sum;
//         }
//     }
// }

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
