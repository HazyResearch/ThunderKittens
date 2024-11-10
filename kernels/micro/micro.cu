#include "kittens.cuh"
#include "cuda_fp8.h"

using namespace kittens;
#define NUM_THREADS (kittens::WARP_THREADS)

#define _row 16
#define _col 32

struct micro_globals {
    using x_gl  = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;
    using o_gl  = gl<float, -1, -1, -1, -1, st_fl<_row, _row>>;
    x_gl x;
    o_gl o;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_fl<_row, _col> (&x_s) = al.allocate<st_fl<_row, _col>>();
    st_fl<_row, _row> (&o_s) = al.allocate<st_fl<_row, _row>>();
    st_fl8<_row, _col> (&x_fp8_s) = al.allocate<st_fl8<_row, _col>>();

    // Warp-level MMA
    rt_fl<_row, _col> x_reg_fl;
    rt_fl<_row, _col, ducks::rt_layout::col> x_reg_fl_col;
    rt_fl8<_row, _col> x_fp8_reg;  // fp8x4 ( 4 floats ) per element 
    rt_fl8<_row, _col, ducks::rt_layout::col> x_fp8_reg_col;  // fp8x4 ( 4 floats ) per element
    rt_fl <_row, _row> att_block;  // float2 (2 floats ) per element


    // zero(att_block);
    load(x_s, g.x, {0, 0, 0, 0});
    __syncthreads();
    load(x_reg_fl, x_s);
    // swap_layout(x_reg_fl_col, x_reg_fl);
    __syncthreads();

    // // now do the matmul in fp8
    copy(x_fp8_reg, x_reg_fl);
    // copy(x_fp8_reg_col, x_reg_fl_col);
    __syncthreads();
    if (threadIdx.x == 0) { 
        printf("In mma_ABt\n");
        printf("D: %d %d\n", att_block.rows, att_block.cols);
        printf("A: %d %d\n", x_fp8_reg.rows, x_fp8_reg.cols);
        printf("B: %d %d\n", x_fp8_reg.rows, x_fp8_reg.cols);
        printf("C: %d %d\n", att_block.rows, att_block.cols);
    }
    mma_ABt(att_block, x_fp8_reg, x_fp8_reg, att_block); // o = torch.matmul(x, x.transpose(1, 2))
    __syncthreads();
    store(o_s, att_block);

    if (threadIdx.x == 0) { printf("End\n"); } 
    __syncthreads();
    store(g.o, o_s, {0, 0, 0, 0});
    __syncthreads();
}

void dispatch_micro( float *d_x, float *d_o ) {
    using x_gl = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;
    using o_gl = gl<float, -1, -1, -1, -1, st_fl<_row, _row>>;
    using globals = micro_globals;
    x_gl  x_arg{d_x, 1, 1, _row, _col};
    o_gl  o_arg{d_o, 1, 1, _row, _row};
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

