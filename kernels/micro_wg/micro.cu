#include "kittens.cuh"
#include "cuda_fp8.h"

using namespace kittens;
#define NUM_WORKERS (1)
#define NUM_THREADS ( NUM_WORKERS * kittens::WARP_THREADS)

#define _row 64
#define _wg_row 64
#define _col 32 

struct micro_globals {
    using x_gl = gl<fp8e4m3, -1, -1, -1, -1, st_fl8<_row, _col>>;
    using o_gl = gl<float, -1, -1, -1, -1, st_fl<_row, _row>>;
    x_gl x;
    o_gl o;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // st_fl< _row, _col> (&x_s)     = al.allocate<st_fl< _row, _col>>();
    // st_bf< _row, _col> (&x_s_bf)  = al.allocate<st_bf< _row, _col>>();
    st_fl8<_row, _col> (&x_s_fp8) = al.allocate<st_fl8<_row, _col>>();
    st_fl< _row, _row> (&o_s)     = al.allocate<st_fl< _row, _row>>();
    
    // registers
    rt_fl8<_wg_row, _col> x_fp8_reg;  // fp8x4 ( 4 floats ) per element 
    rt_fl <_wg_row, _row> att_block;  // float2 (2 floats ) per element

    // loads directly from global memory to fp8 tile
    load(x_s_fp8, g.x, {0, 0, 0, 0});


    // loads
    zero(att_block);
    load(x_fp8_reg, x_s_fp8);
    // matmul from registers
    // warpgroup::load(x_fp8_reg, x_s_fp8);
    // warpgroup::mma_ABt(att_block, x_fp8_reg, x_s_fp8); // o = torch.matmul(x, x.transpose(1, 2))
    // mma_ABt(att_block, x_fp8_reg, x_fp8_reg, att_block); // o = torch.matmul(x, x.transpose(1, 2))

    // matmul from shared
    // warpgroup::mma_ABt(att_block, x_s_fp8, x_s_fp8); // o = torch.matmul(x, x.transpose(1, 2))

    // copy(o_s, x_s_fp8);
    __syncthreads();
    warpgroup::store(o_s, att_block);
    __syncthreads();

    // store
    store(g.o, o_s, {0, 0, 0, 0});
    __syncthreads();
}

void dispatch_micro( fp8e4m3 *d_x, float *d_o ) {
    using x_gl = gl<fp8e4m3, -1, -1, -1, -1, st_fl8<_row, _col>>;
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
    micro_tk<<<1,NUM_THREADS,mem_size>>>(g);
    cudaDeviceSynchronize();
}
#include "harness_fp8.impl"



