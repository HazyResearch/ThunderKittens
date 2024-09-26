#include "mini_strassen_kernel.cu"

#include <random>
#include <chrono>
#include <vector>
#include <array>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include <cuda_fp16.h>

struct Ar2D {
    int R, C;
    std::vector<float> data;
    Ar2D(int M, int N) : R(M), C(N), data(M*N) {}
    float& operator[](int2 i) { return data[(i.x*C + i.y)]; }
};
struct Ar3D {
    int R, C;
    std::vector<half> data;
    Ar3D(int _R, int _C): R(_R), C(_C), data(7*R*C) {}
    half& operator[](int3 i) { return data[(i.x*R + i.y)*C + i.z]; }
};

void init_mat(Ar2D &mat, int salt) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < mat.R; i++) {
        for(int j = 0; j < mat.C; j++) {
            // Hash the coordinates to generate a pseudo-random float
            unsigned int hash = (((i+salt) * 1640531513) ^ ((j+16*salt) * 2654435789)) * 2246822519u;
            hash ^= hash >> 13;
            hash *= 3266489917u;
            hash ^= hash >> 16;
            float random_float = (double)(hash & 0xFFFFFF) / (double)(0xFFFFFF) * 1.0f - 0.5f;
            
            if constexpr (std::is_same_v<float, float>) {
                mat[int2{i, j}] = random_float;
            }
            else {
                mat[int2{i, j}] = __float2half(random_float);
            }
        }
    }
}

void cpu_matmul(Ar2D &A, Ar2D &B, Ar2D &C) {
    int M = A.R, N = B.C, K = A.C;
    const int BLOCK_SIZE = 32; // Adjust this based on your CPU's cache size

    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[{i, k}] * B[{k, j}];
            }
            C[{i, j}] = sum;
        }
    }
}

template<bool A_mat>
void cpu_strassen_generate(Ar2D &a, Ar3D &int_a) {
    const int R = a.R, C = a.C;
    const int half_R = int_a.R, half_C = int_a.C;
    std::cout << "R: " << R << " C: " << C << " half_R: " << half_R << " half_C: " << half_C << std::endl;
    assert(R == 2*half_R && C == 2*half_C);
    for(int i = 0; i < half_R; i++) {
        for(int j = 0; j < half_C; j++) {
            int iB = i + half_R, jB = j + half_C;
            if constexpr (A_mat) {
                int_a[{0, i, j}] = a[{i, j}] + a[{iB, jB}];
                int_a[{1, i, j}] = a[{iB, j}] + a[{iB, jB}];
                int_a[{2, i, j}] = a[{i, j}];
                int_a[{3, i, j}] = a[{iB, jB}];
                int_a[{4, i, j}] = a[{i, j}] + a[{i, jB}];
                int_a[{5, i, j}] = a[{iB, j}] - a[{i, j}];
                int_a[{6, i, j}] = a[{i, jB}] - a[{iB, jB}];
            }
            else {
                int_a[{0, i, j}] = a[{i, j}] + a[{iB, jB}];
                int_a[{1, i, j}] = a[{i, j}];
                int_a[{2, i, j}] = a[{i, jB}] - a[{iB, jB}];
                int_a[{3, i, j}] = a[{iB, j}] - a[{i, j}];
                int_a[{4, i, j}] = a[{iB, jB}];
                int_a[{5, i, j}] = a[{i, j}] + a[{i, jB}];
                int_a[{6, i, j}] = a[{iB, j}] + a[{iB, jB}];
            }
        }
    }
}
template<bool A_mat>
void gpu_strassen_generate_internal(Ar2D &a, Ar3D &int_a) {
    const int R = a.R, C = a.C;
    const int half_R = int_a.R, half_C = int_a.C;
    std::cout << "R: " << R << " C: " << C << " half_R: " << half_R << " half_C: " << half_C << std::endl;
    assert(R == 2*half_R && C == 2*half_C);
    for(int i_block = 0; i_block < R/128; i_block++) {
        for(int j_block = 0; j_block < C/128; j_block++) {
            int r_off1 = i_block*128, c_off1 = j_block*128;
            int r_off2 = i_block*64,  c_off2 = j_block*64;
            for(int k = 0; k < 64; k++) {
                for(int l = 0; l < 64; l++) {
                    int i = r_off1 + k, j = c_off1 + l;
                    int iB = i + 64, jB = j + 64;
                    if constexpr (A_mat) {
                        int_a[{0, r_off2+k, c_off2+l}] = __float2half(a[{i, j}] + a[{iB, jB}]);
                        int_a[{1, r_off2+k, c_off2+l}] = __float2half(a[{iB, j}] + a[{iB, jB}]);
                        int_a[{2, r_off2+k, c_off2+l}] = __float2half(a[{i, j}]);
                        int_a[{3, r_off2+k, c_off2+l}] = __float2half(a[{iB, jB}]);
                        int_a[{4, r_off2+k, c_off2+l}] = __float2half(a[{i, j}] + a[{i, jB}]);
                        int_a[{5, r_off2+k, c_off2+l}] = __float2half(a[{iB, j}] - a[{i, j}]);
                        int_a[{6, r_off2+k, c_off2+l}] = __float2half(a[{i, jB}] - a[{iB, jB}]);
                    }
                    else {
                        int_a[{0, r_off2+k, c_off2+l}] = __float2half(a[{i, j}] + a[{iB, jB}]);
                        int_a[{1, r_off2+k, c_off2+l}] = __float2half(a[{i, j}]);
                        int_a[{2, r_off2+k, c_off2+l}] = __float2half(a[{i, jB}] - a[{iB, jB}]);
                        int_a[{3, r_off2+k, c_off2+l}] = __float2half(a[{iB, j}] - a[{i, j}]);
                        int_a[{4, r_off2+k, c_off2+l}] = __float2half(a[{iB, jB}]);
                        int_a[{5, r_off2+k, c_off2+l}] = __float2half(a[{i, j}] + a[{i, jB}]);
                        int_a[{6, r_off2+k, c_off2+l}] = __float2half(a[{iB, j}] + a[{iB, jB}]);
                    }
                }
            }
        }
    }
}
void gpu_strassen_generate(Ar2D &a, Ar2D &b, half **a_d, half **b_d) {
    Ar3D A_strassen(a.R/2, a.C/2), B_strassen(b.R/2, b.C/2);
    gpu_strassen_generate_internal<true>(a, A_strassen);
    std::cout << "First 20 elements of A_strassen:" << std::endl;
    for (int i = 0; i < 20; ++i) {
        float value = __half2float(A_strassen.data[i]);
        std::cout << value << " ";
        if ((i + 1) % 5 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
    int a_bytes = sizeof(half)*A_strassen.data.size(), b_bytes = sizeof(half)*B_strassen.data.size();
    gpu_strassen_generate_internal<false>(b, B_strassen);
    std::cout << "First 20 elements of B_strassen:" << std::endl;
    for (int i = 0; i < 20; ++i) {
        float value = __half2float(B_strassen.data[i]);
        std::cout << value << " ";
        if ((i + 1) % 5 == 0) std::cout << std::endl;
    }
    std::cout << "GPU strassen matrices are of size " << a_bytes << " and " << b_bytes << " bytes" << std::endl;
    cudaMalloc(a_d, a_bytes);
    std::cout << "Allocated " << a_bytes << " bytes for a_d" << std::endl;
    cudaMalloc(b_d, b_bytes);
    std::cout << "Allocated " << b_bytes << " bytes for b_d" << std::endl;
    cudaMemcpy(*a_d, A_strassen.data.data(), a_bytes, cudaMemcpyHostToDevice);
    std::cout << "Copied " << a_bytes << " bytes to a_d" << std::endl;
    cudaMemcpy(*b_d, B_strassen.data.data(), b_bytes, cudaMemcpyHostToDevice);
    std::cout << "Copied " << b_bytes << " bytes to b_d" << std::endl;
}

void cpu_strassen_matmul_internal(Ar3D &a, Ar3D &b, Ar2D &c) {
    int R = c.R/2, C = c.C/2, K = a.C;
    for(int i = 0; i < 2*R; i++) for(int j = 0; j < 2*C; j++) c[{i, j}] = 0; // zero c
    for(int i = 0; i < R; i++) {
        for(int j = 0; j < C; j++) {
            for(int k = 0; k < K; k++) {
                int iB = i + R, jB = j + C;
                float f;
                f = __half2float(a[{0, i, k}] * b[{0, k, j}]);   // M1
                c[{i, j}] += f; c[{iB, jB}] += f;  // M1
                f = __half2float(a[{1, i, k}] * b[{1, k, j}]);   // M2
                c[{iB, j}] += f; c[{iB, jB}] -= f; // M2
                f = __half2float(a[{2, i, k}] * b[{2, k, j}]);   // M3
                c[{i, jB}] += f; c[{iB, jB}] += f; // M3
                f = __half2float(a[{3, i, k}] * b[{3, k, j}]);   // M4
                c[{i, j}] += f; c[{iB, j}] += f;   // M4
                f = __half2float(a[{4, i, k}] * b[{4, k, j}]);   // M5
                c[{i, j}] -= f; c[{i, jB}] += f;   // M5
                f = __half2float(a[{5, i, k}] * b[{5, k, j}]);   // M6
                c[{iB, jB}] += f;                // M6
                f = __half2float(a[{6, i, k}] * b[{6, k, j}]);   // M7
                c[{i, j}] += f;                  // M7
            }
        }
    }
}
void gpu_strassen_matmul_internal(int M, int N, int K, half **a_d, half **b_d, half **c_d) {
    cudaMalloc(c_d, sizeof(half*)*M*N);
    using mmt = matmul_template<2, 1>;
    dim3 grid = mmt::grid(M, N, K);
    std::cout << "Grid size: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    // Allocate host memory for the first 20 elements
    half* h_a_sample = new half[20];
    // Copy the first 20 elements from device to host
    cudaMemcpy(h_a_sample, *a_d, 20 * sizeof(half), cudaMemcpyDeviceToHost);
    std::cout << "[from GPU] First 20 elements of a_d:" << std::endl;
    for (int i = 0; i < 20; ++i) {
        float value = __half2float(h_a_sample[i]);
        std::cout << value << " ";
        if ((i + 1) % 5 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
    // Allocate host memory for the first 20 elements of b_d
    half* h_b_sample = new half[20];
    // Copy the first 20 elements from device to host
    cudaMemcpy(h_b_sample, *b_d, 20 * sizeof(half), cudaMemcpyDeviceToHost);
    std::cout << "[from GPU] First 20 elements of b_d:" << std::endl;
    for (int i = 0; i < 20; ++i) {
        float value = __half2float(h_b_sample[i]);
        std::cout << value << " ";
        if ((i + 1) % 5 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
    // Free the host memory
    delete[] h_a_sample;
    // Free the host memory
    delete[] h_b_sample;
    
    dim3 block(prototype::num_threads<mmt>);
    unsigned long mem_size = MAX_SHARED_MEMORY; // need to launch two blocks if possible.
    cudaFuncSetAttribute(prototype::pc<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    mmt::layout::input_layout Ag(*a_d, nullptr, nullptr, M/2, K/2);
    mmt::layout::input_layout Bg(*b_d, nullptr, nullptr, K/2, N/2);
    mmt::layout::output_layout Cg(*c_d, nullptr, nullptr, M, N);
    mmt::layout::globals G{Ag, Bg, Cg};

    prototype::pc<mmt><<<grid, block, mem_size>>>(G); // warmup
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    prototype::pc<mmt><<<grid, block, mem_size>>>(G);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;
    double elapsed_us = elapsed.count();
    std::cout << "Kernel execution time: " << elapsed_us << " microseconds" << std::endl;

    // Calculate TFLOPS
    // For Strassen's algorithm, the number of FLOPs is approximately 2(FMA) * 7*(M/2)*(N/2)*(K/2)
    double num_flops = 7.0 * 2.0 * (M/2.0) * (N/2.0) * (K/2.0);
    double tflops = (num_flops / elapsed_us) / 1e6;  // Convert microseconds to seconds and FLOPS to TFLOPS
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
    std::cout << "Relative to naive: " << tflops*8/7 << " TFLOPS" << std::endl;
}

struct gpu_runner {
    static void strassen(Ar2D &a, Ar2D &b, Ar2D &c) {
        int M = a.R, N = b.C, K = a.C;
        half *a_d, *b_d, *c_d;
        std::cout << "Starting GPU Strassen" << std::endl;
        gpu_strassen_generate(a, b, &a_d, &b_d);
        std::cout << "Ran Strassen generate" << std::endl;
        gpu_strassen_matmul_internal(M, N, K, &a_d, &b_d, &c_d);
        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
            // Optionally, you might want to exit the program or handle the error in some way
            exit(-1);
        }
        std::cout << "Ran Strassen matmul internal" << std::endl;
        half *h_c = new half[M*N];
        cudaMemcpy(h_c, c_d, sizeof(half)*M*N, cudaMemcpyDeviceToHost);
        std::cout << "Copied results back to host" << std::endl;
        for(int i = 0; i < M*N; i++) {
            c.data[i] = __half2float(h_c[i]);
        }
        cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);
        std::cout << "Freed memory" << std::endl;
    }
};
struct cpu_runner {
    static void strassen(Ar2D &a, Ar2D &b, Ar2D &c) {
        int M = a.R, N = b.C, K = a.C;
        Ar3D A_strassen(M/2, K/2), B_strassen(K/2, N/2);
        cpu_strassen_generate<true>(a, A_strassen);
        cpu_strassen_generate<false>(b, B_strassen);
        cpu_strassen_matmul_internal(A_strassen, B_strassen, c);
    }
};

template<typename runner>
void test_strassen(int M, int N, int K) {
    Ar2D A(M, K);
    Ar2D B(K, N);
    Ar2D C_ref(M, N), C_test(M, N);
    std::cout << "Initializing matrices" << std::endl;
    init_mat(A, 0);
    std::cout << "Initialized A" << std::endl;
    init_mat(B, 1);
    std::cout << "Initialized B" << std::endl;
    cpu_matmul(A, B, C_ref);
    std::cout << "Computed reference matrix" << std::endl;
    runner::strassen(A, B, C_test);
    std::cout << "Computed C_test" << std::endl;
    int wrong_count = 0;
    double total_error = 0.0;
    double total_ref = 0.0;
    int total_elements = M * N;

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float ref = __half2float(C_ref[{i, j}]), test = __half2float(C_test[{i, j}]);
            double error = std::abs(test - ref);
            total_error += error;
            total_ref += std::abs(ref);
            if(error > 0.01f && wrong_count < 10) {
                std::cout << "Mismatch at (" << i << ", " << j << "): " << test << " != " << ref << " (ref)" << std::endl;
                wrong_count++;
            }
        }
    }

    double average_error = total_error / total_elements;
    double average_ref = total_ref / total_elements;
    if(wrong_count == 0) {
        std::cout << "No mismatches found" << std::endl;
    } else {
        std::cout << "Found >=" << wrong_count << " mismatches" << std::endl;
    }

    std::cout << "Average error: " << average_error << std::endl;
    std::cout << "Average ref: " << average_ref << std::endl;
    std::cout << "Average error / average ref: " << average_error / average_ref << std::endl;
}

int main() {
    test_strassen<cpu_runner>(256, 256, 256);
    test_strassen<gpu_runner>(256, 256, 256);
    test_strassen<gpu_runner>(2816, 1536, 4096);
}
