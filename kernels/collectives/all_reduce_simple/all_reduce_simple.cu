// Simple, pure-C++ all-reduce kernel for educational purposes

#include <random>
#include <vector>

#include "kittens.cuh"

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
    warp::all_reduce_add(intermediate, d_in, {row, col});

    // Store the results to all participating devices
    // For performance, it is recommended to use TMA, which allows you to broadcast
    for (int i = 0; i < NUM_DEVICES; i++)
        warp::store(d_out[i], intermediate, {row, col});
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
    size_t size = N * N * sizeof(float);
    size_t allocated_size;
    float **d_in = new float*[NUM_DEVICES];
    float **d_out = new float*[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        detail::vmm::vm_alloc_map_set_access((void **)&d_in[dev_idx], &allocated_size, size, dev_idx, NUM_DEVICES);
        detail::vmm::vm_alloc_map_set_access((void **)&d_out[dev_idx], &allocated_size, size, dev_idx, NUM_DEVICES);
        CUDACHECK(cudaMemcpy(d_in[dev_idx], h_in[dev_idx], N * N * sizeof(float), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(d_out[dev_idx], 0, N * N * sizeof(float))); // for sanity check
    }

    // Initialize multicast
    // When binding to PyTorch, this can be done easily with TKParallelTensor
    float *d_in_mc;
    float *d_out_mc;
    detail::vmm::handle d_in_mc_handle;
    detail::vmm::handle d_out_mc_handle;
    size_t mc_allocated_size;
    detail::vmm::multicast_create_handle(&d_in_mc_handle, &mc_allocated_size, allocated_size, NUM_DEVICES);
    detail::vmm::multicast_create_handle(&d_out_mc_handle, &mc_allocated_size, allocated_size, NUM_DEVICES);
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        // No need to set device, as these are low-level driver API calls
        detail::vmm::multicast_check(dev_idx);
        detail::vmm::multicast_bind_device(d_in_mc_handle, dev_idx);
        detail::vmm::multicast_bind_device(d_out_mc_handle, dev_idx);
    }
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        // Must be done after all devices are bound
        detail::vmm::multicast_bind_address(d_in_mc_handle, d_in[dev_idx], allocated_size);
        detail::vmm::multicast_bind_address(d_out_mc_handle, d_out[dev_idx], allocated_size);
    }
    detail::vmm::vm_map((void **)&d_in_mc, d_in_mc_handle, mc_allocated_size);
    detail::vmm::vm_map((void **)&d_out_mc, d_out_mc_handle, mc_allocated_size);
    detail::vmm::vm_set_access((void *)d_in_mc, mc_allocated_size, NUM_DEVICES);
    detail::vmm::vm_set_access((void *)d_out_mc, mc_allocated_size, NUM_DEVICES);

    // Handles can be released immediately after address mapping
    detail::vmm::vm_free(d_in_mc_handle);
    detail::vmm::vm_free(d_out_mc_handle);

    // Initialize PGLs
    PGL d_in_pgl {d_in_mc, d_in, nullptr, nullptr, nullptr, nullptr};
    PGL d_out_pgl {d_out_mc, d_out, nullptr, nullptr, nullptr, nullptr};

    // Initialize KittensClub, a host-side threadpool
    // A better way is to use TKParallelTensor with multiprocessing, but this is good
    // enough for now
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; i++) device_ids[i] = i;
    KittensClub club(device_ids, NUM_DEVICES);

    // Launch the kernel
    // Each GPU device will perform all-reduce on `1 / NUM_DEVICES` portion of the entire layout.
    // It's important that we divide the work, otherwise things will become quickly bottlenecked by bandwidth.
    // Also, we will make each block perform all-reduce on 32x32 sub-tile
    dim3 grid(N / TILE_N, N / TILE_N / NUM_DEVICES); // assume it's divisible
    dim3 block(32); // single warp per block; each will handle 32x32 subtile
    printf("Launching kernel...\n");
    club.execute([&](int dev_idx, cudaStream_t stream) {
        all_reduce_kernel<<<grid, block, 0, stream>>>(d_in_pgl, d_out_pgl, dev_idx);
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

    // Clean up multicast
    // When using TKParallelTensor, these are automatically handled in the destructor
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        detail::vmm::multicast_unbind_device(d_in_mc_handle, mc_allocated_size, dev_idx);
        detail::vmm::multicast_unbind_device(d_out_mc_handle, mc_allocated_size, dev_idx);
    }
    // Must be unmapped after unbinding device, since multicast object will be freed immediately
    detail::vmm::vm_unmap(d_in_mc, mc_allocated_size);
    detail::vmm::vm_unmap(d_out_mc, mc_allocated_size);

    // Clean up device-side memory
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        detail::vmm::vm_unmap(d_in[dev_idx], allocated_size);
        detail::vmm::vm_unmap(d_out[dev_idx], allocated_size);
    }

    // Clean up host-side memory
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        delete[] h_in[dev_idx];
        delete[] h_out[dev_idx];
    }
    delete[] h_in;
    delete[] h_out;
    delete[] expected;
    delete[] d_in;
    delete[] d_out;

    std::cout << "Done!" << std::endl;
    return 0;
}
