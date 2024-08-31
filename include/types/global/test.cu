#define KITTENS_HOPPER

#include "global.cuh"
#include <iostream>

using namespace kittens;

using shared_tile = st_bf_4x4;
using a_global_layout = gt<bf16, 4, 4>::l<-1, 64, 1, 1>;
// using a_global_layout = gt_st<shared_tile>::l<-1, 64, 1, 1>;

__global__ void test_kernel(a_global_layout data) {
    data.data[0] = base_types::constants<bf16>::pos_infty();
}

int main() {
    
    std::cout << "Hello World" << std::endl;
    
    bf16 *data;
    // Allocate memory on the GPU for data
    cudaError_t err = cudaMalloc(&data, sizeof(bf16) * shared_tile::num_elements); // Allocating space for 16 bf16 elements
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate memory on GPU: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    a_global_layout a_global(data, 95, nullptr, nullptr, nullptr);

    test_kernel<<<1, 1>>>(a_global);
    cudaDeviceSynchronize();

    bf16 data_host;
    cudaMemcpy(&data_host, a_global.data, sizeof(bf16), cudaMemcpyDeviceToHost);

    std::cout << "a_global.data: (" << a_global.data << ", " << __bfloat162float(data_host) << "); m.batch=" << a_global.batch << " ; m.depth=" << a_global.depth << " ; m.rows=" << a_global.rows << " ; m.cols=" << a_global.cols << std::endl;

    return 0;
}