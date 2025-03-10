/**
 * @file
 * @brief Group-scope conversions between data layouts and types for register tiles.
 */

/* ----------  TRIANGULAR FILLS  ---------- */

/**
 * @brief Makes a register tile triangular by zeroing elements above the row index
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void tril(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    int row_offset = ((N_WARPS == 8) ? ((warpid()%4)*2 + warpid()/4) : warpid()) * dst.rows;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx   = row_offset + (i * dst.tile_size_row) + ((k % 2) * 8) + (kittens::laneid() / 4);
                const int global_col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((kittens::laneid() % 4) * 2);
                const int global_col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((kittens::laneid() % 4) * 2) + 1;

                if (global_col_idx_x <= global_row_idx + diagonal) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                                               { dst.tiles[i][j].data[k].x = val; }

                if (global_col_idx_y <= global_row_idx + diagonal) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                                               { dst.tiles[i][j].data[k].y = val; }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void tril(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    int row_offset = ((N_WARPS == 8) ? ((warpid()%4)*2 + warpid()/4) : warpid()) * dst.rows;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx_x = row_offset + (i * dst.tile_size_row) + ((k / 2) * 8) + ((kittens::laneid() % 4) * 2);
                const int global_row_idx_y = row_offset + (i * dst.tile_size_row) + ((k / 2) * 8) + ((kittens::laneid() % 4) * 2) + 1;
                const int global_col_idx   = (j * dst.tile_size_col) + ((k % 2) * 8) + (kittens::laneid() / 4);

                if (global_col_idx <= global_row_idx_x + diagonal) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                                               { dst.tiles[i][j].data[k].x = val; }
                if (global_col_idx <= global_row_idx_y + diagonal) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                                               { dst.tiles[i][j].data[k].y = val; }
            }
        }
    }
}

/**
 * @brief Makes a register tile triangular by zeroing elements below the row index
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void triu(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    int row_offset = ((N_WARPS == 8) ? ((warpid()%4)*2 + warpid()/4) : warpid()) * dst.rows;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx   = row_offset + (i * dst.tile_size_row) + ((k % 2) * 8) + (kittens::laneid() / 4);
                const int global_col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((kittens::laneid() % 4) * 2);
                const int global_col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((kittens::laneid() % 4) * 2) + 1;

                if (global_col_idx_x >= global_row_idx + diagonal) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                                               { dst.tiles[i][j].data[k].x = val; }

                if (global_col_idx_y >= global_row_idx + diagonal) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                                               { dst.tiles[i][j].data[k].y = val; }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void triu(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    int row_offset = ((N_WARPS == 8) ? ((warpid()%4)*2 + warpid()/4) : warpid()) * dst.rows;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx_x = row_offset + (i * dst.tile_size_row) + ((k / 2) * 8) + ((kittens::laneid() % 4) * 2);
                const int global_row_idx_y = row_offset + (i * dst.tile_size_row) + ((k / 2) * 8) + ((kittens::laneid() % 4) * 2) + 1;
                const int global_col_idx   = (j * dst.tile_size_col) + ((k % 2) * 8) + (kittens::laneid() / 4);

                if (global_col_idx >= global_row_idx_x + diagonal) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                                               { dst.tiles[i][j].data[k].x = val; }
                
                if (global_col_idx >= global_row_idx_y + diagonal) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                                               { dst.tiles[i][j].data[k].y = val; }
            }
        }
    }
}