#include "kittens.cuh"
#include "cuda_fp8.h"

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
    zero(x_reg);
    add(x_reg, x_reg, 2.0f);

    copy(x_fp8_reg, x_reg); // Convert float to FP8
    // one(x_fp8_reg);
    //tests
    if (threadIdx.x == 0) {
        __nv_fp8_e4m3 *values = reinterpret_cast<__nv_fp8_e4m3*>(&(x_fp8_reg.tiles[0][0].data[16]));
        printf("Individual values: %f %f %f %f\n", 
               float(values[0]), 
               float(values[1]), 
               float(values[2]), 
               float(values[3]));
    }
    // __syncthreads();
    // copy(x_reg, x_fp8_reg); // Convert FP8 to float

    // // tests
    // if (threadIdx.x == 0) {
    //     printf("After conversion: %f\n", x_reg.tiles[0][0].data[0]);
    //     __nv_fp8_e4m3 one_fp8(1.0f);
    //     printf("Reference one: %f\n", float(one_fp8));
    //     uint8_t bits;
    //     memcpy(&bits, &one_fp8, sizeof(uint8_t));
    //     printf("Correct bit pattern for 1.0: 0x%02X\n", bits);
    // }

    __syncthreads();
    store(x_s, x_reg);
    if (threadIdx.x == 0) { printf("End\n"); } 
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

//     float4 f4 = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
//     __nv_fp8x4_e4m3 fp8_4(f4);
//     printf("Raw storage: %u\n", fp8_4.__x);
//     __nv_fp8_e4m3 *values = reinterpret_cast<__nv_fp8_e4m3*>(&fp8_4);
//     printf("Individual values: %f %f %f %f\n", 
//            float(values[0]), 
//            float(values[1]), 
//            float(values[2]), 
//            float(values[3]));
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

