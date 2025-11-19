/**
 * @file
 * @brief Group (collaborative warp) ops for handling tiles from and to parallel global memory.
 *
 * WARNING: This API is in an experimental stage.
 */

template <int TILE_ROWS, int TILE_COLS, kittens::reduce_op OP, kittens::ducks::pgl::all PGL>
__device__ inline static void all_reduce(PGL &pgl, const coord<ducks::default_type> &idx) {
    static_assert(std::is_same_v<typename PGL::dtype, bf16>, "Currently only bf16 is supported.");
    static_assert(TILE_COLS > 0 && TILE_COLS % (WARP_THREADS * 2) == 0, "TILE_COLS must be a multiple of 64 for best performance.");

    const int warps_per_row = TILE_COLS / (WARP_THREADS * 2);
    const int row_base = idx.r * TILE_ROWS;
    const int col_base = idx.c * TILE_COLS;
    const int warp_laneid = threadIdx.x % WARP_THREADS;

    for (int i = warpid(); i < TILE_ROWS * warps_per_row; i += GROUP_WARPS) {
        int row_idx = row_base + i / warps_per_row;
        int col_idx = col_base + (i % warps_per_row) * WARP_THREADS * 2 + warp_laneid * 2; // bf16x2
        bf16_2 tmp;
        bf16_2 *ptr = reinterpret_cast<bf16_2 *>(pgl.mc_ptr_at({idx.b, idx.d, row_idx, col_idx}));
        multimem<bf16_2>::ld_reduce<OP>(tmp, ptr);
        multimem<bf16_2>::st(ptr, tmp);
    }
}

template <int TILE_ROWS, int TILE_COLS, kittens::reduce_op OP, kittens::ducks::pgl::all PGL, kittens::ducks::gl::all GL>
__device__ inline static void reduce(GL &dst, const coord<ducks::default_type> &dst_idx, PGL &src, const coord<ducks::default_type> &src_idx) {
    static_assert(std::is_same_v<typename PGL::dtype, bf16>, "Currently only bf16 is supported.");
    static_assert(TILE_COLS > 0 && TILE_COLS % (WARP_THREADS * 2) == 0, "TILE_COLS must be a multiple of 64 for best performance.");

    const int warps_per_row = TILE_COLS / (WARP_THREADS * 2);
    const int src_row_base = src_idx.r * TILE_ROWS;
    const int src_col_base = src_idx.c * TILE_COLS;
    const int dst_row_base = dst_idx.r * TILE_ROWS;
    const int dst_col_base = dst_idx.c * TILE_COLS;
    const int warp_laneid = threadIdx.x % WARP_THREADS;

    for (int i = warpid(); i < TILE_ROWS * warps_per_row; i += GROUP_WARPS) {
        int src_row_idx = src_row_base + i / warps_per_row;
        int src_col_idx = src_col_base + (i % warps_per_row) * WARP_THREADS * 2 + warp_laneid * 2; // bf16x2
        int dst_row_idx = dst_row_base + i / warps_per_row;
        int dst_col_idx = dst_col_base + (i % warps_per_row) * WARP_THREADS * 2 + warp_laneid * 2; // bf16x2
        (void)dst_row_idx; // Suppress false positive unused warning
        (void)dst_col_idx; // Suppress false positive unused warning
        bf16_2 tmp;
        multimem<bf16_2>::ld_reduce<OP>(tmp, reinterpret_cast<bf16_2 *>(src.mc_ptr_at({src_idx.b, src_idx.d, src_row_idx, src_col_idx})));
        move<bf16_2>::stg(reinterpret_cast<bf16_2 *>(&dst[{dst_idx.b, dst_idx.d, dst_row_idx, dst_col_idx}]), tmp);
    }
}
