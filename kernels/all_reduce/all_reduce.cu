#include <random>
#include <kittens.cuh>

using namespace kittens;

constexpr int NUM_DEVICES = 8;
constexpr int N = 4096;
constexpr int TILE_N = 32;

// Individual, single-GPU global memory layout
using GL = gl<float, 1, 1, N, N>;

// PGL represents a same-shape global memory layout across multiple GPUs
using PGL = pgl<gl<float, 1, 1, N, N>, NUM_DEVICES>;

// Example kernel using PGL
// Performs all-reduce operation on d_in, stores the result to d_out
// Since d_in is a PGL object, we can utilize the NVSwitch accelerator to perform
// optimized collective operations
// Since d_out is a PGL object, any memory write to it is broadcasted to all 
// participating devices
__global__ void all_reduce_kernel(PGL d_in, PGL d_out, int dev_idx) {
    const int row = blockIdx.y + dev_idx * (N / TILE_N / NUM_DEVICES);
    const int col = blockIdx.x;

    // Declare intermediate registers to store the results of all-reduce
    rt_fl<TILE_N, TILE_N> intermediate;

    // Perform all-reduce!
    all_reduce_add(intermediate, d_in, dev_idx, {row, col});

    // Multicast-store the results to all participating devices
    broadcast(d_out, intermediate, dev_idx, {row, col});
}

// Main function. Sets up memory and launches the example kernel 
int main() {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Allocate and initialize host memory
    printf("Preparing input data...\n");
    float **h_in = new float*[NUM_DEVICES];
    float **h_out = new float*[NUM_DEVICES];
    float *expected = new float[N * N]; // reference output

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        h_in[dev_idx] =  new float[N * N];
        h_out[dev_idx] =  new float[N * N];
        for (int i = 0; i < N * N; ++i)
            h_in[dev_idx][i] = dis(gen);
    }
    for (int i = 0; i < N * N; ++i) {
        expected[i] = 0.0f;
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx)
            expected[i] += h_in[dev_idx][i];
    }

    // Allocate and initialize device memory on each device
    float **d_in = new float*[NUM_DEVICES];
    float **d_out = new float*[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        cudaMalloc(&d_in[dev_idx], N * N * sizeof(float));
        cudaMalloc(&d_out[dev_idx], N * N * sizeof(float));
        cudaMemcpy(d_in[dev_idx], h_in[dev_idx], N * N * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Initialize PGLs
    // PGLs require an array of device indices and an array of device memory pointers
    // All of the pointers should point to identically-sized device memory locations
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) 
        device_ids[i] = i;
    PGL d_in_pgl{device_ids, d_in, nullptr, nullptr, nullptr, nullptr}; // no need for runtime dimensions
    PGL d_out_pgl{device_ids, d_out, nullptr, nullptr, nullptr, nullptr};

    // Initialize KittensClub, a host-side threadpool
    // We can run multi-GPU kernels without this, but that incurs context switching overhead
    KittensClub club(device_ids, NUM_DEVICES);

    // Launch the kernel
    // Each GPU device will perform all-reduce on `1 / NUM_DEVICES` portion of the entire layout.
    // It's important that we divide the work, otherwise things will become quickly bottlenecked by bandwidth.
    // Also, we will make each block perform all-reduce on 32x32 sub-tile
    dim3 grid(N / TILE_N, N / NUM_DEVICES / TILE_N); // assume it's divisible
    dim3 block(32); // single warp per block; each will handle 32x32 subtile
    printf("Launching kernel...\n");
    club.execute([&](int dev_idx) {
        all_reduce_kernel<<<grid, block, 0, 0>>>(d_in_pgl, d_out_pgl, dev_idx);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    });

    // Bring back data
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        cudaMemcpy(h_out[dev_idx], d_out[dev_idx], N * N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Verify the results
    printf("Verifying output...\n");
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        float max_abs_error = 0.0;
        for (int i = 0; i < N * N; ++i) {
            float abs_error = fabs(expected[i] - h_out[dev_idx][i]);
            max_abs_error = fmax(max_abs_error, abs_error);
        }
        printf("Device %d: max absolute error: %f\n", dev_idx, max_abs_error);
    }

    // Clean up
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        delete[] h_in[dev_idx];
        delete[] h_out[dev_idx];
        cudaFree(d_in[dev_idx]);
        cudaFree(d_out[dev_idx]);
    }
    delete[] h_in;
    delete[] h_out;
    delete[] expected;
    delete[] d_in;
    delete[] d_out;

    std::cout << "Done!" << std::endl;
    return 0;
}
