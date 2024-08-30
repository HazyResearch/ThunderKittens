#define KITTENS_HOPPER

#include "gt.cuh"
#include <iostream>

using namespace kittens;

int main() {
    
    std::cout << "Hello World" << std::endl;
    
    using s = st_bf_4x4;
    bf16 *data;
    // Allocate memory on the GPU for data
    cudaError_t err = cudaMalloc(&data, sizeof(bf16) * 16); // Allocating space for 16 bf16 elements
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate memory on GPU: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    gt<s, true>::make<> m(data, 1, 1, 1, 1);

    std::cout << "m: " << m.data << " ; m.batch=" << m.batch.v << " ; m.depth=" << m.depth.v << " ; m.rows=" << m.rows.v << " ; m.cols=" << m.cols.v << std::endl;

    return 0;
}