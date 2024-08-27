//#define TORCH_COMPILE

#include "src/kittens.cuh"

using namespace kittens;

// In prod set both to 16
// Number of batches per SM
#define B_TILE 1
// Number of heads per SM
#define H_TILE 1

// Maximum number of warps than can run in parallel on a block
// Maybe increase so we can schedule other warps during a warp's IO time
#define NUM_WORKERS 4


__global__ void fftconv_tk(bf16 *u, bf16 *kf, bf16 *f, bf16 *finv, bf16 *t, bf16 *tinv, bf16 *o, int h, int n, int n1){
    int warps = (blockDim.x + kittens::WARP_THREADS - 1) / kittens::WARP_THREADS;
    auto warpid = kittens::warpid();
    int warp_rows = warps * kittens::TILE_DIM;
    // Number of row tiles each warp handles
    int row_tiles = (n1 + warp_rows - 1) / warp_rows;
    int col_tiles = (n1 + kittens::TILE_DIM - 1) / kittens::TILE_DIM;

    int h_stride = n;
    int b_stride = h * n;
    int h_start = blockIdx.y * H_TILE;
    int b_start = blockIdx.x * B_TILE;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    bf16 *f_smem, *finv_smem, *t_smem, *tinv_smem, *kf_smem, *x_smem, *xt_smem;
 
    // Creating smem array based on num warps makes it easier to load in smem
    kittens::st_cmplx_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&f_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();
    kittens::st_cmplx_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&finv_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();
    kittens::st_cmplx_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&t_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();
    kittens::st_cmplx_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&tinv_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();
    kittens::st_cmplx_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&kf_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();
    // Organize by warp into sets of row tiles (easier to load into smem this way)
    kittens::st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&x_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();
    // For ease of use, create separate SMEM bank for storing the intermediate transposed X
    // Since other warps may still need to read the old X
    kittens::st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle> (&xt_smem)[warps] = al.allocate<st_bf<row_tiles, col_tiles, ducks::st_layout::swizzle>, warps>();

    kittens::rt_bf_1x1 a_reg, accum1, accum2, accum3;
    kittens::rt_bf<1, 1, kittens::ducks::rt_layout::col> b_reg;

    kittens::load(f_smem[warpid], f + (row_idx * n1), n1);
    kittens::load(finv_smem[warpid], finv + (row_idx * n1), n1);
    // t and tinv need to be reshaped to n1xn1
    kittens::load(t_smem[warpid], t + (row_idx * n1), n1);
    kittens::load(tinv_smem[warpid], tinv + (row_idx * n1), n1);
    for (int h = 0; h < H_TILE; h++) {
        int h_offset = (h_start + h) * h_stride;
        // kf needs to be reshaped to n1xn1 then transposed in global
        kittens::load(kf_smem[warpid], kf + h_offset + (row_idx * n1), n1);
        for (int b = 0; b < B_TILE; b++) {
            int b_offset = (b_start + b) * b_stride;
            kittens::load(x_smem[warpid], x + h_offset + b_offset + (row_idx * n1), n1);
            // TODO set zero reg for complex portion of X
            __syncthreads(); // Wait for all smem everywhere to be loaded
            // Rows of X that each warp handles
            for (int r = 0; r < row_tiles; r++) {
                // Col tiles of output X
                for (int c0 = 0; c0 < col_tiles; c0++) {
                    accum3::zero();
                    // X <- XFinv
                    // Col tiles of X, row tiles of Finv
                    for (int c1 = 0; c1 < col_tiles; c1++) {
                        accum2::zero();
                        // X <- XF
                        // Col tiles of X, row tiles of F
                        for (int c2 = 0; c2 < col_tiles; c2++) {
                            // X <-- F^T X
                            accum1::zero();
                            // Col tile of F^T and row tile of X
                            for (int c3 = 0; c3 < col_tiles; c3++) {
                                auto f_st = kittens::subtile_inplace<1, 1>(f_smem[warpid], r*kittens::TILE_DIM, c3:kittens::TILE_DIM);
                                int row_warp = c3 / row_tiles;
                                auto x_st = kittens::subtile_inplace<1, 1>(x_smem[row_warp], (c3 % row_tiles)*kittens::TILE_DIM, c2*kittens::TILE_DIM);
                                
                                // F is symmetric so don't need to worry about transposing F
                                kittens::load(a_reg, f_st);
                                // Transpose x from row-layout in SRAM to col layout in register
                                kittens::load(b_reg, x_st);
                                kittens::mma_AB(accum1, a_reg, b_reg, accum1);
                            }
                            // X = X * t
                            auto t_st = kittens::subtile_inplace<1, 1>(t_smem[warpid], r*kittens::TILE_DIM, c2 * kittens::TILE_DIM);
                            kittens::load(a_reg, t_st);
                            kittens::mul(accum1, accum1, a_reg);
                            // X = XF
                            // Need to access col of data - calculate which row section of smem array its in
                            int row_warp = c2 / row_tiles;
                            auto f_st = kittens::subtile_inplace<1, 1>(f_smem[row_warp], (c2 % row_tiles)*kittens::TILE_DIM, c1 * kittens::TILE_DIM);
                            kittens::load(b_reg, f_st);
                            kittens::mma_AB(accum2, accum1, b_reg);
                        }
                        // X = X * k_f^T
                        auto kf_st = kittens::subtile_inplace<1, 1>(kf_smem[warpid], r*kittens::TILE_DIM, c1 * kittens::TILE_DIM);
                        kittens::load(a_reg, kf_st);
                        kittens::mul(accum2, accum2, a_reg);
                        int row_warp = c1 / row_tiles;
                        auto finv_st = kittens::subtile_inplace<1, 1>(finv_smem[row_warp], (c1 % row_tiles)*kittens::TILE_DIM, c0 * kittens::TILE_DIM);
                        kittens::load(a_reg, finv_st);
                        kittens::mma_AB(accum3, accum2, a_reg);
                    }
                    // Storing in transposed layout - rows/cols are opposite
                    // Need to transpose within the tile as well
                    //auto x_store_st = kittens::subtile_inplace<1, 1>(xt_smem[warpid], row_tile*kittens::TILE_DIM, c0*kittens::TILE_DIM);
                    int row_warp = c0 / row_tiles;
                    int col = warpid * row_tiles + r;
                    auto x_store_st = kittens::subtile_inplace<1, 1>(xt_smem[row_warp], (c0 % row_tile)*kittens::TILE_DIM, col*kittens::TILE_DIM);
                    //kittens::rt_bf<1, 1, kittens::ducks::rt_layout::col> xt_store_reg;
                    // Use existing col layout b_reg for transpose?
                    kittens::transpose_sep(b_reg, accum3);
                    kittens::store(x_store_st, b_reg);
                    // Stored one X col tile at a given row
                }
            } // All warps finished X computation and wrote X^T to SMEM, now load X^T from SMEM
            __syncthreads();
            for (int r = 0; r < row_tiles; r++) {
                // Col tiles of output Y^T
                for (int c0 = 0; c0 < col_tiles; c0++) {
                    // Col tiles of X^T, row tiles of Finv
                    accum2::zero();
                    for (int c1 = 0; c1 < col_tiles; c1++) {
                        auto xt_st = kittens::subtile_inplace<1, 1>(xt_smem[warpid], r*kittens::TILE_DIM, c0:kittens::TILE_DIM);
                        auto tinv_st = kittens::subtile_inplace<1, 1>(tinv_smem[warpid], r*kittens::TILE_DIM, c0 * kittens::TILE_DIM);
                        kittens::load(a_reg, xt_st);
                        kittens::load(accum1, tinv_st);
                        kittens::mul(a_reg, a_reg, accum1);

                        int row_warp = c1 / row_tiles;
                        auto finv_st = kittens::subtile_inplace<1, 1>(finv_smem[row_warp], (c1 % row_tiles)*kittens::TILE_DIM, c0 * kittens::TILE_DIM);
                        kittens::load(b_reg, finv_st);
                        kittens::mma_AB(accum2, a_reg, b_reg, accum2);
                    }
                    kittens::transpose_sep(b_reg, accum2);
                    int row_start = (warpid * row_tiles + r) * kittens::TILE_DIM;
                    int col_start = c0*kittens::TILE_DIM;
                    bf16 *_o = o + h_offset + b_offset + (row_start * n1) + col_start;
                    // TODO only store real component since we know the output will be real-valued
                    kittens::store(_o, b_reg, n1);
                }
            }
        // End of batch
        }
    // End of head
    }
}

void launch_fftconv_tk(bf16 *u, bf16 *kf, bf16 *f, bf16 *finv, bf16 *t, bf16 *tinv, bf16 *o, int h, int n, int n1, long mem_size) {
    int warps = NUM_WORKERS;
    // Can't fill the SM w/ warps, and each warp only has a single tile
    if (n1 / kittens::TILE_DIM < NUM_WORKERS) {
        warps = (n1 + TILE_DIM - 1) / kittens::TILE_DIM;
    }
    const dim3 block_dim{
        (unsigned int)(warps*kittens::WARP_THREADS)
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

    fftconv_tk<<<grid_dim, block_dim, mem_size>>>(u, kf, f, finv, t, tinv, o, h, n, n1);
}