#include "kittens.cuh"

#include <random>

constexpr int NUM_DEVICES = 8;
constexpr size_t N = 4096;

constexpr int WARPSIZE = 32;
constexpr int ITER_PER_THREAD = 32;
constexpr int MAX_VEC_SIZE = 16;

using namespace kittens;

using  base_tile       =  st_bf<64, 64>;
using  pglobal_layout  =  pgl<gl<bf16, 1, 1, -1, -1, base_tile>, true>;

__global__ void all_reduce_bf16(bf16 *data, const int N);

int main() {

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Setup
    int nelem = N * N;
    size_t size = nelem * sizeof(bf16);

    // Allocate and initialize host memory
    float **host_mats = new float*[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        host_mats[dev_idx] =  new float[nelem];
        for (int i = 0; i < nelem; ++i) host_mats[dev_idx][i] = dis(gen);
    }
    bf16 **host_mats_bf16 = new bf16*[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        host_mats_bf16[dev_idx] = new bf16[nelem];
        for (int i = 0; i < nelem; ++i)
            host_mats_bf16[dev_idx][i] = __float2bfloat16(host_mats[dev_idx][i]);
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
    bf16 **dev_mats = new bf16*[NUM_DEVICES];
    CUmemGenericAllocationHandle *dev_handles = new CUmemGenericAllocationHandle[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        pglCudaMalloc(dev_idx, &dev_mats[dev_idx], &dev_handles[dev_idx], size);
        cudaMemcpy(dev_mats[dev_idx], host_mats_bf16[dev_idx], size, cudaMemcpyHostToDevice);
    }

    // Initialize parallel global layout
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;
    pglobal_layout dev_mat_pgl{device_ids, NUM_DEVICES, dev_mats, nullptr, nullptr, N, N};

    // Perform the reduction
    KittensClub club(device_ids, NUM_DEVICES);

    int nelem_per_dev = nelem / NUM_DEVICES;
    constexpr int nelem_per_block = 256 * ITER_PER_THREAD * (MAX_VEC_SIZE / sizeof(__nv_bfloat16));

    dim3 grid((nelem_per_dev + nelem_per_block - 1) / nelem_per_block);
    dim3 block(256);

    club.execute([&](int worker_id) {
        all_reduce_bf16<<<grid, block>>>(dev_mat_pgl.raw_multi_ptr[worker_id] + nelem_per_dev * worker_id, nelem_per_dev);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    });

    // Bring back data
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        cudaMemcpy(host_mats_bf16[dev_idx], dev_mats[dev_idx], size, cudaMemcpyDeviceToHost);
    }

    // Convert back to float
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        for (int i = 0; i < nelem; ++i) 
            host_mats[dev_idx][i] = __bfloat162float(host_mats_bf16[dev_idx][i]);
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
    float TOL = 1e-1; // large due to fp16 <-> bf16 conversion
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
        delete[] host_mats_bf16[dev_idx];
        pglCudaFree(dev_idx, dev_mats[dev_idx], dev_handles[dev_idx], size);
    }
    delete[] host_mats;
    delete[] host_mats_bf16;
    delete[] expected;
    delete[] dev_mats;
    delete[] dev_handles;

    std::cout << "Done!" << std::endl;
    return 0;
}

__global__ void all_reduce_bf16(bf16 *data, const int N) {

    if (blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) {
        printf("Error: Only 1D grids and blocks should be passed in\n");
        return;
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARPSIZE;
    int lane_id = threadIdx.x % WARPSIZE;

    constexpr int N_per_iter = MAX_VEC_SIZE / sizeof(__nv_bfloat16);
    constexpr int N_per_warp_per_iter = N_per_iter * WARPSIZE;
    constexpr int N_per_warp = ITER_PER_THREAD * N_per_warp_per_iter;
    int start_idx = N_per_warp * warp_id;

    for (int i = 0; i < ITER_PER_THREAD; ++i) {
        int idx = start_idx + i * N_per_warp_per_iter + lane_id * N_per_iter;
        if (idx < N) {
            volatile float x, y, z, w; // hacking type to hold 2 bfloat16s
            __nv_bfloat16 *ptr = data + idx;
            asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(ptr) : "memory");
            asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
        }
        __syncthreads();
    }
}
