#include "kittens.cuh"

using namespace kittens;
#define NUM_THREADS (kittens::WARP_THREADS)

struct micro_globals {
    using x_gl  = gl<float, -1, -1, -1, -1, st_fl<16, 16>>;
    x_gl x, o;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_fl<16, 16> (&x_s) = al.allocate<st_fl<16, 16>>();
    st_fl8<16, 16> (&x_fp8_s) = al.allocate<st_fl8<16, 16>>();
    st_hf<16, 16> (&x_hf_s) = al.allocate<st_hf<16, 16>>();
    
    // register
    rt_fl<16, 16> x_reg;
    rt_fl8<16, 16> x_fp8_reg;
    
    __syncthreads();
    load( x_s, g.x, {0, 0, 0, 0});
    load( x_reg, x_s );
    add(x_reg, x_reg, 1.0f);
    __syncthreads();
    copy(x_fp8_reg, x_reg);
    copy(x_reg, x_fp8_reg);
    // copy(x_fp8_s, x_s);
    printf("3");
    __syncthreads();
    store(g.o, x_s, {0, 0, 0, 0});
    __syncthreads();
}

void dispatch_micro( float *d_x, float *d_o ) {
    using x_gl = gl<float, -1, -1, -1, -1, st_fl<16, 16>>;
    using globals = micro_globals;
    x_gl  x_arg{d_x, 1, 1, 16, 16};
    x_gl  o_arg{d_o, 1, 1, 16, 16};
    globals g{x_arg, o_arg};
    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(
        micro_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    micro_tk<<<1,32,mem_size>>>(g);
    cudaDeviceSynchronize();
}
#include "harness_fp8.impl"

// #include "cuda_fp8.h"
// #include <cstdio>
// __global__ void temp(float *a, float *b) {
//     float t = *a;
//     __nv_fp8_e4m3 fp8(t);
//     *b = float(fp8);
// }
// int main() {
//     float *a, *b;
//     cudaMalloc(&a, sizeof(float));
//     cudaMalloc(&b, sizeof(float));
//     temp<<<1, 1>>>(a, b);
//     cudaDeviceSynchronize();
//     // check for error
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         printf("CUDA error: %s\n", cudaGetErrorString(error));
//     }
//     else {
//         printf("Success\n");
//     }
//     return 0;
// }

