// Simple, pure-C++ all-reduce kernel for educational purposes

#include <random>
#include <vector>

#include "kittens.cuh"
#include "pyutils/club.cuh"

using namespace kittens;

constexpr int N = 4096;

constexpr int NUM_DEVICES = 8;
constexpr int NUM_THREADS = 256;

// Individual, single-GPU global memory layout
using GL = gl<float, 1, 1, N, N>;

// PGL represents a same-shape global memory layout across multiple GPUs
using PGL = pgl<gl<float, 1, 1, N, N>, NUM_DEVICES>;

// Example kernel using PGL
// Performs in-place all-reduce operation on `d_data`
// Since `d_data` is a PGL object, we can utilize the NVSwitch accelerator to perform
// optimized collective operations
__global__ void all_reduce_kernel(PGL d_data, int dev_idx) {
    const size_t N_total = d_data.numel();
    const size_t N_per_dev = N_total / NUM_DEVICES;

    // Since we are using float2, each thread handles 2 elements
    const size_t idx = N_per_dev * dev_idx + 2 * NUM_THREADS * blockIdx.x + 2 * threadIdx.x;

    // Perform all-reduce!
    float2 tmp;
    multimem<float2>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<float2 *>(&d_data.mc_ptr[idx]));
    multimem<float2>::st(reinterpret_cast<float2 *>(&d_data.mc_ptr[idx]), tmp);
}

// Main function. Sets up multicast memory and launches the example kernel 
int main() {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Allocate and initialize host memory
    printf("Preparing input data...\n");
    float **h_data = new float*[NUM_DEVICES];
    float *expected = new float[N * N]; // reference output

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        h_data[dev_idx] =  new float[N * N];
        for (int i = 0; i < N * N; ++i)
            h_data[dev_idx][i] = dis(gen);
    }
    for (int i = 0; i < N * N; ++i) {
        expected[i] = 0.0f;
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx)
            expected[i] += h_data[dev_idx][i];
    }

    // Allocate and initialize device memory on each device
    size_t size = N * N * sizeof(float);
    size_t allocated_size;
    float **d_data = new float*[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        detail::vmm::vm_alloc_map_set_access((void **)&d_data[dev_idx], &allocated_size, size, dev_idx, NUM_DEVICES);
        CUDACHECK(cudaMemcpy(d_data[dev_idx], h_data[dev_idx], N * N * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Initialize multicast
    // When binding to PyTorch, this can be done easily with TKParallelTensor
    float *d_data_mc;
    detail::vmm::handle d_data_mc_handle;
    size_t mc_allocated_size;
    detail::vmm::multicast_create_handle(&d_data_mc_handle, &mc_allocated_size, allocated_size, NUM_DEVICES);
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        // No need to set device, as these are low-level driver API calls
        detail::vmm::multicast_check(dev_idx);
        detail::vmm::multicast_bind_device(d_data_mc_handle, dev_idx);
    }
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        // Must be done after all devices are bound
        detail::vmm::multicast_bind_address(d_data_mc_handle, d_data[dev_idx], allocated_size);
    }
    detail::vmm::vm_map((void **)&d_data_mc, d_data_mc_handle, mc_allocated_size);
    detail::vmm::vm_set_access((void *)d_data_mc, mc_allocated_size, NUM_DEVICES);

    // Handles can be released immediately after address mapping
    detail::vmm::vm_free(d_data_mc_handle);

    // Initialize PGLs
    PGL d_data_pgl {d_data_mc, d_data, nullptr, nullptr, nullptr, nullptr};

    // Initialize KittensClub, a host-side threadpool
    // A better way is to use TKParallelTensor with multiprocessing, but this is good
    // enough for now
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; i++) device_ids[i] = i;
    KittensClub club(device_ids, NUM_DEVICES);

    // Launch the kernel
    // Each GPU device will perform all-reduce on `1 / NUM_DEVICES` portion of the entire layout.
    // It's important that we divide the work, otherwise things will become quickly bottlenecked by bandwidth.
    dim3 grid(N * N / (2 * NUM_THREADS * NUM_DEVICES)); // assume it's divisible
    dim3 block(NUM_THREADS);
    printf("Launching kernel...\n");
    club.execute([&](int dev_idx, cudaStream_t stream) {
        all_reduce_kernel<<<grid, block, 0, stream>>>(d_data_pgl, dev_idx);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    });

    // Bring back data
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        cudaMemcpy(h_data[dev_idx], d_data[dev_idx], N * N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Verify the results
    printf("Verifying output...\n");
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        float max_abs_error = 0.0;
        for (int i = 0; i < N * N; ++i) {
            float abs_error = fabs(expected[i] - h_data[dev_idx][i]);
            max_abs_error = fmax(max_abs_error, abs_error);
        }
        printf("Device %d: max absolute error: %f\n", dev_idx, max_abs_error);
    }

    // Clean up multicast
    // When using TKParallelTensor, these are automatically handled in the destructor
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        detail::vmm::multicast_unbind_device(d_data_mc_handle, mc_allocated_size, dev_idx);
    }
    // Must be unmapped after unbinding device, since multicast object will be freed immediately
    detail::vmm::vm_unmap(d_data_mc, mc_allocated_size);

    // Clean up device-side memory
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx)
        detail::vmm::vm_unmap(d_data[dev_idx], allocated_size);

    // Clean up host-side memory
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx)
        delete[] h_data[dev_idx];
    delete[] h_data;
    delete[] d_data;
    delete[] expected;

    std::cout << "Done!" << std::endl;
    return 0;
}
