#define TORCH_COMPILE 

#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda/barrier>

#define NUM_WORKERS (1) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

const int N_CHUNK  = 16;
const int head_dim =  64;                   
const float rope_embd_fraction = 1.0f;

const int rope_dim = rope_embd_fraction * head_dim;
const int half_rope_dim = ( rope_dim / 2 );
const int excess_dim = head_dim - rope_dim; 

const int seq_tiles = N_CHUNK / kittens::TILE_DIM;
const int rope_tiles = rope_dim / kittens::TILE_DIM;
const int half_rope_tiles = half_rope_dim / kittens::TILE_DIM;
const int excess_rope_tiles = excess_dim / kittens::TILE_DIM;

using namespace kittens;

#define tile_1xFULL_ROPE_D st<bf16, seq_tiles, rope_tiles>
#define tile_1xHALF_ROPE_D st<bf16, seq_tiles, half_rope_tiles>
#define tile_1xEXCESS_ROPE_D st<bf16, seq_tiles, excess_rope_tiles>

#define reg_tile_1xFULL_ROPE_D rt_bf<seq_tiles, rope_tiles>
#define reg_tile_1xHALF_ROPE_D rt_bf<seq_tiles, half_rope_tiles>
#define reg_tile_1xEXCESS_ROPE_D rt_bf<seq_tiles, excess_rope_tiles>

__global__ __launch_bounds__(NUM_THREADS, 1)
void _fused_rotary( 
    int n, const bf16* __x, const bf16* __cos_in, const bf16* __sin_in, bf16* __o 
) {
    auto warpid = kittens::warpid();
    auto lane = kittens::laneid();

    static_assert(rope_embd_fraction == 1.0f); // smaller rope_embd_fraction currently unsupported

    // shared memory setup to load from hbm
    const bf16 *x_g   = reinterpret_cast<const bf16*>(__x) + blockIdx.x*(n*head_dim);
    const bf16 *cos_g = reinterpret_cast<const bf16*>(__cos_in); // doesn't change with batch, head
    const bf16 *sin_g = reinterpret_cast<const bf16*>(__sin_in);
          bf16 *o_g   = reinterpret_cast<bf16*>(__o) + blockIdx.x*(n*head_dim);

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    tile_1xFULL_ROPE_D (&x_s)       = al.allocate<tile_1xFULL_ROPE_D>();    
    tile_1xHALF_ROPE_D (&cos_s)     = al.allocate<tile_1xHALF_ROPE_D>(); 
    tile_1xHALF_ROPE_D (&sin_s)     = al.allocate<tile_1xHALF_ROPE_D>(); 

    int tic = 0, toc = 1;    
    const int total_elements = N_CHUNK * head_dim;
    int n_blocks = n / (NUM_WORKERS*kittens::TILE_DIM);
    for (int block = 0; block < n_blocks; block ++, tic ^=1, toc ^=1) {

        // smem loads
        load(x_s,  x_g    + (block * total_elements),      head_dim);
        load(cos_s, cos_g + (block * cos_s.num_elements),  half_rope_dim);
        load(sin_s, sin_g + (block * sin_s.num_elements),  half_rope_dim);

        // register loads
        reg_tile_1xHALF_ROPE_D cos, sin, x1, x2, temp1, temp2;
        reg_tile_1xFULL_ROPE_D x;
        load(x, x_s);
        load(cos, cos_s);
        load(sin, sin_s);

        const int x_width = x.width;
        const int x1_width = x1.width;
        for (int i = 0; i < x1_width; i ++ ) { 
            x1.tiles[0][i] = x.tiles[0][i]; // first half dims
            x2.tiles[0][i] = x.tiles[0][i+x1_width]; // second half dims
        }
        
        // a = torch.cat((x1, x2), dim=-1) * repeat(cos_in, "n d -> 1 n (2 d)" ) 
        mul(temp1, x1, cos);
        mul(temp2, x2, cos);

        // b = torch.cat((-x2, x1), dim=-1)  * repeat(sin_in, "n d -> 1 n (2 d)" )
        mul(x2, x2, __float2bfloat16(-1.0f));
        mul(x2, x2, sin);
        mul(x1, x1, sin);

        // sum ( a + b )
        add(temp1, temp1, x2);
        add(temp2, temp2, x1);

        // assemble the result
        zero(x);
        for (int i = 0; i < x1_width; i ++ ) { 
            x.tiles[0][i] = temp1.tiles[0][i];
            x.tiles[0][i+x1_width] = temp2.tiles[0][i];
        }
        
        // store out
        store(o_g + ( block*x.num_elements ), x, rope_dim);
    }
}


#ifdef TORCH_COMPILE
#include "common/pyutils/torch_helpers.cuh"
#include <iostream>
void 
fused_rotary(
    const torch::Tensor x,  
    const torch::Tensor cos_in,  
    const torch::Tensor sin_in,  
    torch::Tensor out
) {
    CHECK_INPUT(x);
    CHECK_INPUT(cos_in);
    CHECK_INPUT(sin_in);
    CHECK_INPUT(out);

    int batch = x.size(0);
    auto n    = x.size(2);
    auto n_h  = x.size(1);

    TORCH_CHECK(batch == out.size(0),            "Differing batch sizes?");
    TORCH_CHECK(x.size(3) == head_dim,           "x is head_dim?");
    TORCH_CHECK(cos_in.size(1) == half_rope_dim, "cos_in is half_rope_dim?");
    TORCH_CHECK(sin_in.size(1) == half_rope_dim, "sin_in is half_rope_dim?");
    TORCH_CHECK(out.size(3) == head_dim,         "out is head_dim?");

    TORCH_CHECK(x.size(2) % kittens::TILE_DIM == 0,      "sequence length is divisible by 16?");
    TORCH_CHECK(cos_in.size(0) % kittens::TILE_DIM == 0, "sequence length is divisible by 16?");
    TORCH_CHECK(sin_in.size(0) % kittens::TILE_DIM == 0, "sequence length is divisible by 16?");
    TORCH_CHECK(out.size(2) % kittens::TILE_DIM == 0,    "sequence length is divisible by 16?");

    // convert to bf16
    c10::BFloat16 *x_ptr           = x.data_ptr<c10::BFloat16>();
    c10::BFloat16 *cos_in_ptr      = cos_in.data_ptr<c10::BFloat16>();
    c10::BFloat16 *sin_in_ptr      = sin_in.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr           = out.data_ptr<c10::BFloat16>();
    const bf16* x_bf           = reinterpret_cast<const bf16*>(x_ptr);
    const bf16* cos_in_bf      = reinterpret_cast<const bf16*>(cos_in_ptr);
    const bf16* sin_in_bf      = reinterpret_cast<const bf16*>(sin_in_ptr);
          bf16* o_bf           = reinterpret_cast<bf16*>(o_ptr);

    // launch variables
    auto threads = NUM_WORKERS * kittens::WARP_THREADS;
    unsigned long mem_size = N_CHUNK*head_dim*2 + N_CHUNK*half_rope_dim*2*2 + 4800;
    cudaFuncSetAttribute(
        _fused_rotary,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    _fused_rotary<<<batch*n_h,threads,mem_size>>>( 
        n, x_bf, 
        cos_in_bf, sin_in_bf,
        o_bf 
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
#else
#include "harness.impl"
#endif

