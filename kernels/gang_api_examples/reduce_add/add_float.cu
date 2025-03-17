#include "kittens.cuh"

#include <random>

constexpr int NUM_DEVICES = 2;
constexpr size_t N = 64;

using namespace kittens;

using global_layout   =  gl<float, 1, 1, -1, -1>;
using pglobal_layout  =  pgl<gl<float, 1, 1, -1, -1>, true>;
using kittens_pgl = kittens::PglObj<global_layout>;

// need to use same datatype otherwise doesn't add anything
__global__ void all_reduce_int(kittens_pgl p_o) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
        blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        asm volatile(
            "multimem.red.relaxed.sys.global.add.v4.f32 [%0], {4.0,5.0,6.0,7.0};"
            :
            : "l"(p_o.mc_ptr)
            : "memory"
        );
    }
}

int main() {
    // Setup
    int nelem = N * N;
    size_t size = nelem * sizeof(int);

    float *host_mat_1 = new float[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_1[i] = 1.0f;

    float *host_mat_2 = new float[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_2[i] = i;

    // Print data
    printf("Device 1: ");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", host_mat_1[i]);
    }
    printf("... (%d elements)\n", nelem);

    printf("Device 2: ");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", host_mat_2[i]);
    }
    printf("... (%d elements)\n", nelem);
    
    
    // Allocate and copy data to device
    float **dev_mats = new float*[NUM_DEVICES];
    CUmemGenericAllocationHandle *dev_handles = new CUmemGenericAllocationHandle[NUM_DEVICES];

    cudaSetDevice(0);
    pglCudaMalloc(0, &dev_mats[0], &dev_handles[0], size);
    cudaMemcpy(dev_mats[0], host_mat_1, size, cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    pglCudaMalloc(1, &dev_mats[1], &dev_handles[1], size);
    cudaMemcpy(dev_mats[1], host_mat_2, size, cudaMemcpyHostToDevice);

    // Initialize parallel global layout
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;
    pglobal_layout dev_mat_pgl{device_ids, NUM_DEVICES, dev_mats, nullptr, nullptr, N, N};

    // Perform the reduction
    KittensClub club(device_ids, NUM_DEVICES);

    
    dim3 grid(2);
    dim3 block(64);
    cudaSetDevice(0);
    all_reduce_int<<<grid, block>>>(dev_mat_pgl.get_pgl_obj(0));
    printf("Device 0 mc_ptr: %p\n", dev_mat_pgl.get_pgl_obj(0).mc_ptr);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Bring back data
    cudaMemcpy(host_mat_1, dev_mats[0], size, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(host_mat_2, dev_mats[1], size, cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Device 1: ");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", host_mat_1[i]);
    }
    printf("... (%d elements)\n", nelem);

    printf("Device 2: ");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", host_mat_2[i]);
    }
    printf("... (%d elements)\n", nelem);

    // Cleanup and exit
    delete[] dev_mats;
    delete[] dev_handles;
    delete[] host_mat_1;
    delete[] host_mat_2;

    std::cout << "Done!" << std::endl;
    return 0;
}