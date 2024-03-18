# ThunderKittens
### Tile primitives for speedy kernels

<div align="center" >
    <img src="thunderkittens.png" height=350 alt="ThunderKittens logo" style="margin-bottom:px"/> 
</div>

<br>
<br>

ThunderKittens is a framework to make it easy to write fast deep learning kernels in CUDA (and, soon, ROCm and others, too!)

ThunderKittens is built around three key principles:
1. Speed. Kernels written in ThunderKittens should be at least as fast as those written from scratch -- especially because ThunderKittens can do things the “right” way under the hood.
2. Simplicity. ThunderKittens is stupidly simple to write.
3. Extensibility. ThunderKittens embeds itself natively, so that if you need more than ThunderKittens can offer, it won’t get in your way of building it yourself.

ThunderKittens is built from the hardware up -- we do what the silicon tells us. And modern GPUs tell us that they want to work with fairly small tiles of data. A GPU is not really a 1000x1000 matrix multiply machine (even if it is often used as such); it’s a manycore processor where each core can efficiently run ~16x16 matrix multiplies. Consequently, ThunderKittens is built around manipulating tiles of data no smaller than 16x16 values.

ThunderKittens solves a few key problems that enable high utilization on modern hardware.
1. Tensor cores. ThunderKittens can call fast tensor core functions, including asynchronous WGMMA calls on H100 GPUs.
2. Shared Memory. I got ninety-nine problems but a bank conflict ain’t one.
3. Loads and stores. Hide latencies with asynchronous copies and address generation with TMA.
4. Distributed Shared Memory. L2 is _so_ last year.

*Example: A Simple Atention Kernel*

Here’s an example of what flash attention for an RTX 4090 looks like written in ThunderKittens.

```#include "kittens.cuh"

#define NUM_WORKERS 16
using namespace kittens;
__global__ void attend_ker(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__) {

    auto warpid        = threadIdx.x / 32;
    auto block_start   = blockIdx.x*(n*d);
    const bf16 *_q = __q__ + block_start, *_k = __k__ + block_start, *_v = __v__ + block_start;
          bf16 *_o = __o__ + block_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&k_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&v_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();

    rt_bf_1x4<> q_reg, k_reg, v_reg;
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;
    rt_fl_1x4<> o_prev;
    rt_fl_1x1<>::col_vec max_vec_last, max_vec;
    rt_fl_1x1<>::col_vec norm_vec_last, norm_vec;
    
    int qo_blocks = n / (q_reg.rows*NUM_WORKERS), kv_blocks = n / (q_reg.rows*NUM_WORKERS);
    
    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

        load(q_reg, _q + (q_blk*NUM_WORKERS + warpid)*q_reg.rows*d, d);
        mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment

        neg_infty(max_vec); // zero registers for the Q chunk
        zero(norm_vec);
        zero(o_prev);

        for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

            load(v_smem[warpid], _v + (kv_idx*NUM_WORKERS + warpid)*q_reg.rows*d, d);
            load(k_smem[warpid], _k + (kv_idx*NUM_WORKERS + warpid)*q_reg.rows*d, d);
            __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

            for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {

                load(k_reg, k_smem[subtile]);

                zero(att_block);
                dot(att_block, q_reg, k_reg, att_block);

                copy(norm_vec_last, norm_vec);
                copy(max_vec_last,  max_vec);

                row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
                sub_row(att_block, att_block, max_vec);
                exp(att_block, att_block);

                sub(max_vec_last, max_vec_last, max_vec);
                exp(max_vec_last, max_vec_last);
                mul(norm_vec, norm_vec, max_vec_last);

                row_sum(norm_vec, att_block, norm_vec); // accumulate onto the norm_vec
                div_row(att_block, att_block, norm_vec);

                mul(norm_vec_last, norm_vec_last, max_vec_last);
                div(norm_vec_last, norm_vec_last, norm_vec);

                copy(att_block_mma, att_block); // convert to bf16 for mma

                load(v_reg, v_smem[subtile]);
                rt_bf_1x4<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

                mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma'ing onto it
                mma(o_prev, att_block_mma, v_reg_col, o_prev);
            }
            __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
        }

        store(_o + (q_blk*NUM_WORKERS + warpid)*q_reg.rows*d, o_prev, d); // write out o
    }
}
```

Altogether, this is 58 lines of code (not counting whitespace), and achieves about 122 TFLOPs on an RTX 4090. (74% of theoretical max.) We’ll go through some of these primitives more carefully in the next section, the ThunderKittens manual.

## ThunderKittens Manual

ThunderKittens is actually a pretty small library, in terms of what it gives you.

 - Data types: (Register + shared) * (tiles + vectors), all parameterized by layout, type, and size.
 - Operations for manipulating these objects.

Despite its simplicity, there are still a few sharp edges that you might encounter if you don’t know what’s going on under the hood. So, we do recommend giving this manual a good read before sitting down to write a kernel -- it’s not too long, we promise!

### NVIDIA’s Programming Model

To understand ThunderKittens, it will help to begin by understanding a bit of how NVIDIA’s programming model works, as NVIDIA provides a few different “scopes” to think about when writing parallel code.
1. Thread -- this is the level of doing work on an individual bit of data, like a floating point multiplication. A thread has up to 256 32-bit registers it can access every cycle.
2. Warp -- 32 threads make a warp. This is the level at which instructions are issued by the hardware. It’s also the base (and default) scope from which ThunderKittens operates; most ThunderKittens programming happens here.
3. Warpgroup -- 4 warps make a warpgroup. This is the level from which asynchronous warpgroup matrix multiply-accumulate instructions are issued. (We really wish we could ignore this level, but you unfortunately need it for the H100.) Correspondingly, many matrix multiply and memory operations are supported at the warpgroup level.
4. Block -- N warps make a block, which is the level that shares “shared memory” in the CUDA programming model. In ThunderKittens, N is usually 8. ThunderKittens also supports some shared memory operations at the block level.
5. Grid -- M blocks make a grid, where M should be equal to (or slightly less) than a multiple of the number of SMs on the GPU to avoid tail effects.

“Register” objects exist at the level of warps -- their contents is split amongst the threads of the warp. Register objects include:
- Register tiles, declared as the `kittens::rt` struct in `src/register_tile/rt.cuh`. Kittens provides a few useful wrappers -- for example, a 32x16 row-layout bfloat16 register tile can be declared as `kittens::rt_bf_2x1;` -- row-layout is implicit by default.
- Register vectors, which are associated with register tiles. They come in two flavors: column vectors and row vectors. Column vectors are used to reduce or map across tile rows, and row vectors reduce and map across tile columns. For example, to hold the sum of the rows of the tile declared above, we would create a `kittens::rt_bf_2x1<>::col_vec;`
In contrast, “Shared” objects exist at the level of the block, and sit only in shared memory.

All ThunderKittens functions follow a common signature. Much like an assembly language (ThunderKittens is in essence an abstract tile-oriented RISC instruction set), the destination of every function is the first operand, and the source operands are passed sequentially afterwards.

For example, if we have three 32x64 floating point register tiles: `kittens::rt_fl_2x4 a, b, c;`, we can element-wise multiply `a` and `b` and store the result in `c` with the following call: `kittens::mul(c, a, b);`.

Similarly, if we want to then store the result into a shared tile `__shared__ kittens:st_bf_2x4 s;`,we write the function analogously: `kittens::store(s, c);`.

### Typing

ThunderKittens tries hard to protect you from yourself. In particular, ThunderKittens wants to know layouts of objects at compile-time and will make sure they’re compatible before letting you do operations. This is important because there are subtleties to the allowable layouts for certain operations, and without static checks it is very easy to get painful silent failures. For example, a normal matrix multiply requires the B operand to be in a column layout, whereas an outer dot product requires the B operand to be in a row layout.

If you are being told an operation that you think exists doesn't exist, double-check your layouts -- this is the most common error. Only then report a bug :)

### Scopes

By default, ThunderKittens operations exist at the warp-level. In other words, each function expects to be called by only a single warp, and that single warp will do all of the work of the function. If multiple warps are assigned to the same work, undefined behavior will result. (And if the operation involves memory movement, it is likely to be completely catastrophic.) In general, you should expect your programming pattern to involve calculating a `warpid` at the beginning of the kernel, and assigning tasks to data based on that id.

However, not all ThunderKittens functions operate at the warp level. Many important operations, like TMA loads and stores, distributed memory access, and WGMMA instructions operate at different levels. Fortunately, each scope other than warp scope has its own namespace, so that it is difficult to accidentally use the wrong scope. To make it easier, we also replicate many functions at these other scopes, too, for your convenience.

### Other Restrictions

Most operations in ThunderKittens are pure functional. However, some operations _do_ have special restrictions; ThunderKittens tries to warn you by giving them names that stand out. For example, a register tile transpose needs separable arguments: if it is given the same underlying registers as both source and destination, it will silently fail. Consequently, it is named `transpose_sep`.