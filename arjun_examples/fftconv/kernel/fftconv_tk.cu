//#define TORCH_COMPILE

#include "src/kittens.cuh"

using namespace kittens;

// Number of batches per SM
#define B_TILE 16
// Number of heads per SM
#define H_TILE 16

// Maximum number of warps than can run in parallel on a block
// Maybe increase so we can schedule other warps during a warp's IO time
#define NUM_WORKERS 4


__global__ void fftconv_tk(bf16 *u, bf16 *kf, bf16 *f, bf16 *finv, bf16 *t, bf16 *tinv, bf16 *o, int h, int n){
    int warps = (blockDim.x + kittens::WARP_THREADS - 1) / kittens::WARP_THREADS;
    auto warpid = kittens::warpid();
    int warp_rows = warps * kittens::TILE_DIM;
    // Number of row tiles each warp handles
    int row_tiles = (n1 + warp_rows - 1) / warp_rows;
    // For each row, there's also an equiv number of cols
    int col_tiles = (n1 + kittens::TILE_DIM - 1) / kittens::TILE_DIM;
    int row_idx = warpid * (row_tiles * kittens::TILE_DIM);

    int h_stride = n;
    int b_stride = h * n;
    int h_start = blockIdx.y * H_TILE;
    int b_start = blockIdx.x * B_TILE;

    // Number of tiles for each row/col MMA
    int N_tiles = n1 / kittens::TILE_DIM;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    bf16 *f_smem, *finv_smem, *t_smem, *tinv_smem, *kf_smem, *x_smem;
 
    // Creating smem array based on num warps makes it easier to load in smem
    kittens::st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&f_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();
    kittens::st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&finv_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();
    kittens::st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&t_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();
    kittens::st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&tinv_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();
    kittens::st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&kf_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();
    // Organize by warp into sets of row tiles (easier to load into smem this way)
    kittens::st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&x_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();

    kittens::rt_bf_1x1 a_reg, accum1, accum2, accum3;
    kittens::rt_bf<1, 1, kittens::ducks::rt_layout::col> b_reg;

    kittens::load(f_smem[warpid], f + (row_idx * n1), n1);
    kittens::load(finv_smem[warpid], finv + (row_idx * n1), n1);
    // t and tinv need to be reshaped to n1xn1
    kittens::load(t_smem[warpid], t + (row_idx * n1), n1);
    kittens::load(tinv_smem[warpid], tinv + (row_idx * n1), n1);
    __syncthreads();
    for (int h = 0; h < H_TILE; h++) {
        int h_offset = (h_start + h) * h_stride;
        // kf needs to be reshaped to n1xn1 then transposed in global
        kittens::load(kf_smem[warpid], kf + h_offset + (row_idx * n1), n1);
        for (int b = 0; b < B_TILE; b++) {
            int b_offset = (b_start + b) * b_stride;
            kittens::load(x_smem[warpid], x + h_offset + b_offset + (row_idx * n1), n1);
            __syncthreads(); // Wait for all x to be loaded before we proceed
            // Rows of X that warp handles
            for (int r = 0; r < row_tiles; r++) {
                int row_tile = r*kittens::TILE_DIM;
                int row_offset = row_idx + row_tile;
                accum3::zero();
                // X <- XFinv
                // Col tiles of X, row tiles of Finv
                for (int c1 = 0; c1 < col_tiles; c1++) {
                    accum2::zero();
                    // X <- XF
                    // Col tiles of X, row tiles of F
                    for (int c2 = 0; c2 < col_tiles; c2++) {
                        int col_offset = c2 * kittens::TILE_DIM;
                        
                        // X <-- F^T X
                        accum1::zero();
                        // Col tile of F^T and row tile of X
                        for (int c3 = 0; c3 < col_tiles; c3++) {
                            
                            int tile_i = c3 * kittens::TILE_DIM;
                            auto f_st = kittens::subtile_inplace<1, 1>(f_smem[warpid], row_tile, tile_i);
                            bf16 *_x = x_smem + col_offset + (c3 * kittens::TILE_DIM * n1);
                            int row_warp = tile_i / row_tiles;
                            auto x_st = kittens::subtile_inplace<1, 1>(x_smem[row_warp], (tile_i % row_tiles)*kittens::TILE_DIM, col_offset);
                            
                            // F is symmetric so don't need to worry about transposing F
                            kittens::load(a_reg, f_st);
                            // Transpose x from row-layout in SRAM to col layout in register
                            kittens::load(b_reg, x_st);
                            kittens::mma_AB(accum1, a_reg, b_reg, accum1);
                        }
                        // X = X * t
                        kittens::load(a_reg, t_smem);
                        kittens::mul(accum1, accum1, a_reg);
                        // X = XF
                        //bf16 *_f = f_smem + (row_offset * n1) + (i * kittens::TILE_DIM);
                        // Need to access col of data - calculate which row section of smem array its in
                        int row_warp = row_offset / row_tiles;
                        auto f_st = kittens::subtile_inplace<1, 1>(f_smem[row_warp], row_offset % row_tiles, c1 * kittens::TILE_DIM)
                        kittens::load(b_reg, f_st);
                        kittens::mma(accum2, accum1, b_reg);
                    }
                    // X = X * k_f^T
                    kittens::load(b_reg, kf);
                    kittens::mul(accum2, accum2, b_reg);
                    bf16 *_finv = finv_smem + (row_offset * n1) + (i * kittens::TILE_DIM);
                    int row_warp = row_offset / row_tiles;
                    // TODO check
                    auto finv_st = kittens::subtile_inplace<1, 1>(finv_smem[row_warp], row_offset % row_tiles, c1 * kittens::TILE_DIM)
                    kittens::load(b_reg, _finv);
                    kittens::mma(accum3, accum2, b_reg);
                }
                bf16 *_o = o + h_offset + b_offset + row_offset + col_offset;
                __syncthreads();
                kittens::store(x_smem, accum3);
            }
        }
    }
}

void launch_fftconv_tk(bf16 *u, bf16 *kf, bf16 *f, bf16 *finv, bf16 *t, bf16 *tinv, int h, int n, long mem_size) {
    int warps = NUM_WORKERS;
    // Can't fill the SM w/ warps, and each warp only has a single tile
    if (n1 / kittens::TILE_DIM < NUM_WORKERS) {
        warps = (n1 + TILE_DIM - 1) / kittens::TILE_DIM;
    }
    const dim3 block_dim{
        (unsigned int)(warps*NUM_WORKERS)
    };
    // Constant number of blocks in each grid dim
    const dim3 grid_dim{
        (unsigned int)b / B_TILE,
        (unsigned int)h / H_TILE
    };

    cudaFuncSetAttribute(
        fftconv_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    fftconv_tk<<<grid_dim, block_dim, mem_size>>>(x, O, twiddles, n);
}