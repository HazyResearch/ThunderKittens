#include "kittens.cuh"
#include <random>
#include <cuda_bf16.h>  // Include bfloat16 support

constexpr int NUM_DEVICES = 2;
constexpr size_t N = 64;

using namespace kittens;

// Change the layout to use __nv_bfloat16 instead of float
using global_layout   =  gl<__nv_bfloat16, 1, 1, -1, -1>;
using pgl_m  =  pgl_manager<gl<__nv_bfloat16, 1, 1, -1, -1>, true>;
using kittens_pgl = kittens::pgl<global_layout>;

// kittens::atomic_add(p_o);
__global__ void all_reduce_int(kittens_pgl p_o) {
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0 ||
        blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) {
        return;
    }
    bf16 *value = p_o.mc_ptr;
    unsigned int packed = (__bfloat16_as_ushort(value[0]) << 16) | 
                          __bfloat16_as_ushort(value[1]);
    asm volatile(
        "multimem.red.relaxed.sys.global.add.bf16x2 [%0], %1;"
        :
        : "l"(p_o.mc_ptr), "r"(packed)
        : "memory"
    );
    // bf16 *value = p_o.mc_ptr;
    // asm volatile(
    //     "multimem.red.relaxed.sys.global.add.bf16 [%0], %1;"
    //     :
    //     : "l"(p_o.mc_ptr), "h"(__bfloat16_as_ushort(value[0]))
    //     : "memory"
    // );
    
    // bf16 *value = p_o.mc_ptr;
    // unsigned int packed1 = (__bfloat16_as_ushort(value[0]) << 16) | 
    //                             __bfloat16_as_ushort(value[1]);
    // unsigned int packed2 = (__bfloat16_as_ushort(value[2]) << 16) | 
    //                         __bfloat16_as_ushort(value[3]);
    // unsigned int packed3 = (__bfloat16_as_ushort(value[4]) << 16) |
    //                         __bfloat16_as_ushort(value[5]);
    // unsigned int packed4 = (__bfloat16_as_ushort(value[6]) << 16) |
    //                         __bfloat16_as_ushort(value[7]);
    // asm volatile(
    //     "multimem.red.relaxed.sys.global.add.v4.bf16x2 [%0], {%1, %2, %3, %4};"
    //     :
    //     : "l"(p_o.mc_ptr), "r"(packed1), "r"(packed2), "r"(packed3), "r"(packed4)
    //     : "memory"
    // );
}

int main() {
    // Setup
    int nelem = N * N;
    size_t size = nelem * sizeof(__nv_bfloat16);  // Use bfloat16 size

    // Create and initialize host arrays with bfloat16 values
    __nv_bfloat16 *host_mat_1 = new __nv_bfloat16[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_1[i] = __float2bfloat16(1.5f);

    __nv_bfloat16 *host_mat_2 = new __nv_bfloat16[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_2[i] = __float2bfloat16(static_cast<float>(i));

    // Print data - convert bfloat16 to float for printing
    printf("Device 1: ");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", __bfloat162float(host_mat_1[i]));
    }
    printf("... (%d elements)\n", nelem);

    printf("Device 2: ");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", __bfloat162float(host_mat_2[i]));
    }
    printf("... (%d elements)\n", nelem);
    
    // Allocate and copy data to device
    __nv_bfloat16 **dev_mats = new __nv_bfloat16*[NUM_DEVICES];
    CUmemGenericAllocationHandle *dev_handles = new CUmemGenericAllocationHandle[NUM_DEVICES];

    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;
    
    cudaSetDevice(0);
    pglCudaMalloc(NUM_DEVICES, device_ids, 0, &dev_mats[0], &dev_handles[0], size);
    cudaMemcpy(dev_mats[0], host_mat_1, size, cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    pglCudaMalloc(NUM_DEVICES, device_ids, 1, &dev_mats[1], &dev_handles[1], size);
    cudaMemcpy(dev_mats[1], host_mat_2, size, cudaMemcpyHostToDevice);

    // Initialize parallel global layout
    pgl_m dev_mat_pgl{device_ids, NUM_DEVICES, dev_mats, nullptr, nullptr, N, N};

    // Perform the reduction
    KittensClub club(device_ids, NUM_DEVICES);
    
    dim3 grid(1);
    dim3 block(32);
    cudaSetDevice(0);
    all_reduce_int<<<grid, block>>>(dev_mat_pgl.get_pgl_obj(0));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Bring back data
    cudaMemcpy(host_mat_1, dev_mats[0], size, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(host_mat_2, dev_mats[1], size, cudaMemcpyDeviceToHost);
    
    // Print results - convert bfloat16 to float for printing
    printf("Device 1: ");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", __bfloat162float(host_mat_1[i]));
    }
    printf("... (%d elements)\n", nelem);

    printf("Device 2: ");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", __bfloat162float(host_mat_2[i]));
    }
    printf("... (%d elements)\n", nelem);

    // Check correctness, for Device 1, all elements should be 1.5 + 1.5 = 3.0
    for (int i = 0; i < nelem; ++i) {
        if (host_mat_1[i] != __float2bfloat16(3.0f)) {
            // printf("%d ", i);
            std::cerr << "Error: Device 1, index " << i << " expected " << 3.0f << " but got " << __bfloat162float(host_mat_1[i]) << std::endl;
            return -1;
        }
    }

    // Cleanup and exit
    delete[] dev_mats;
    delete[] dev_handles;
    delete[] host_mat_1;
    delete[] host_mat_2;

    std::cout << "Done!" << std::endl;
    return 0;
}