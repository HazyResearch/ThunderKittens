#include "kittens.cuh"

#include <random>

constexpr int NUM_DEVICES = 8;
constexpr size_t N = 4096;

constexpr int ITER_PER_THREAD = 32;
constexpr int MAX_VEC_SIZE = 16;

using namespace kittens;

using base_tile       =  st_fl<64, 64>;
using global_layout   =  gl<float, 1, 1, -1, -1, base_tile>;
using pglobal_layout  =  pgl<gl<float, 1, 1, -1, -1, base_tile>, true>;
using kittens_pgl = kittens::PglObj<global_layout>;

__global__ void all_reduce_float(kittens_pgl p_o) {
    kittens::all_reduce_add(p_o);
}

int main() {
    std::random_device rd;
    std::mt19937 gen(32);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Setup
    int nelem = N * N;
    size_t size = nelem * sizeof(float);

    // Allocate and initialize host memory
    float **host_mats = new float*[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        host_mats[dev_idx] = new float[nelem];
        for (int i = 0; i < nelem; ++i) host_mats[dev_idx][i] = dis(gen);
    }
    
    float *expected = new float[nelem];
    for (int i = 0; i < nelem; ++i) {
        expected[i] = 0.0f;
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx)
            expected[i] += host_mats[dev_idx][i];
    }

    // Print data
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        std::cout << "Device " << dev_idx << ": ";
        for (int i = 0; i < std::min(nelem, 10); ++i) {
            std::cout << host_mats[dev_idx][i] << " ";
        }
        std::cout << "... (" << nelem << " elements)" << std::endl;
    }
    std::cout << "Expected: ";
    for (int i = 0; i < std::min(nelem, 10); ++i) {
        std::cout << expected[i] << " ";
    }
    std::cout << "... (" << nelem << " elements)" << std::endl;

    // Allocate and copy data to device
    float **dev_mats = new float*[NUM_DEVICES];
    CUmemGenericAllocationHandle *dev_handles = new CUmemGenericAllocationHandle[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        pglCudaMalloc(dev_idx, &dev_mats[dev_idx], &dev_handles[dev_idx], size);
        cudaMemcpy(dev_mats[dev_idx], host_mats[dev_idx], size, cudaMemcpyHostToDevice);
    }

    // Initialize parallel global layout
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;
    pglobal_layout dev_mat_pgl{device_ids, NUM_DEVICES, dev_mats, nullptr, nullptr, N, N};

    // Perform the reduction
    KittensClub club(device_ids, NUM_DEVICES);

    int nelem_per_dev = nelem / NUM_DEVICES;
    constexpr int nelem_per_block = 256 * ITER_PER_THREAD * (MAX_VEC_SIZE / sizeof(float));

    dim3 grid((nelem_per_dev + nelem_per_block - 1) / nelem_per_block);
    dim3 block(64, 4);
    club.execute([&](int worker_id) {
        all_reduce_float<<<grid, block>>>(dev_mat_pgl.get_pgl_obj(worker_id));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    });

    // Bring back data
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        cudaMemcpy(host_mats[dev_idx], dev_mats[dev_idx], size, cudaMemcpyDeviceToHost);
    }

    // Print results
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        std::cout << "Device " << dev_idx << ": ";
        for (int i = 0; i < std::min(nelem, 10); ++i) {
            std::cout << host_mats[dev_idx][i] << " ";
        }
        std::cout << "... (" << nelem << " elements)" << std::endl;
    }

    // Verify the results
    float TOL = 1e-5; // Can use tighter tolerance since we're using full precision floats
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        for (int i = 0; i < nelem; ++i) {
            if (fabs(expected[i] - host_mats[dev_idx][i]) > TOL) {
                std::cerr << "Mismatch at device " << dev_idx << 
                             ", index " << i << 
                             ": expected " << expected[i] << 
                             ", got " << host_mats[dev_idx][i] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    // Cleanup and exit
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        delete[] host_mats[dev_idx];
        pglCudaFree(dev_idx, dev_mats[dev_idx], dev_handles[dev_idx], size);
    }
    delete[] host_mats;
    delete[] expected;
    delete[] dev_mats;
    delete[] dev_handles;

    std::cout << "Done!" << std::endl;
    return 0;
}