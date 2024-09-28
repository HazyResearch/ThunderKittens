/**
 * @file
 * @brief Functions for a warpgroup to collaboratively transfer data directly between shared memory and registers and back.
 */

/**
 * @brief Collaboratively load data from a shared tile into register tiles split across a warpgroup.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%N_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");
    int local_warpid = warpid();
    using T2 = RT::dtype;
    using U  = ST::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = ::kittens::laneid();
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 2) {
                // handle the row-major layout for 16-bit types
                int row = (local_warpid*warp_height + i)*dst.tile_size + (warp_laneid / 4);
                int col = j*dst.tile_size + 2*(warp_laneid % 4);
                U2 tmp[4];
                move<U2>::lds(tmp[0], &src[{row+0, col+0}]);
                move<U2>::lds(tmp[1], &src[{row+8, col+0}]);
                move<U2>::lds(tmp[2], &src[{row+0, col+8}]);
                move<U2>::lds(tmp[3], &src[{row+8, col+8}]);
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 4) {
                // handle the row-major layout for 32-bit types
                int row = (local_warpid*warp_height + i)*dst.tile_size + (warp_laneid / 4);
                int col = j*dst.tile_size + 2*(warp_laneid % 4);
                int blit = (warp_laneid%4) / 2;
                U2 tmp[4];
                move<U>::lds(tmp[0].x, &src[{row+0, (col+0)^blit}]);
                move<U>::lds(tmp[0].y, &src[{row+0, (col+1)^blit}]);
                move<U>::lds(tmp[1].x, &src[{row+8, (col+0)^blit}]);
                move<U>::lds(tmp[1].y, &src[{row+8, (col+1)^blit}]);
                move<U>::lds(tmp[2].x, &src[{row+0, (col+8)^blit}]);
                move<U>::lds(tmp[2].y, &src[{row+0, (col+9)^blit}]);
                move<U>::lds(tmp[3].x, &src[{row+8, (col+8)^blit}]);
                move<U>::lds(tmp[3].y, &src[{row+8, (col+9)^blit}]);
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
                if(blit) {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        dst.tiles[i][j].data[k] = T2{dst.tiles[i][j].data[k].y, dst.tiles[i][j].data[k].x};
                    }
                }
            }
            else {
                // handle the column-major layout
                int row = (local_warpid*warp_height + i)*dst.tile_size + 2*(warp_laneid % 4);
                int col = j*dst.tile_size + (warp_laneid / 4);
                U2 tmp[4];
                move<U>::lds(tmp[0].x, &src[{row+0, col+0}]);
                move<U>::lds(tmp[0].y, &src[{row+1, col+0}]);
                move<U>::lds(tmp[1].x, &src[{row+0, col+8}]);
                move<U>::lds(tmp[1].y, &src[{row+1, col+8}]);
                move<U>::lds(tmp[2].x, &src[{row+8, col+0}]);
                move<U>::lds(tmp[2].y, &src[{row+9, col+0}]);
                move<U>::lds(tmp[3].x, &src[{row+8, col+8}]);
                move<U>::lds(tmp[3].y, &src[{row+9, col+8}]);
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
        }
    }
}


/**
 * @brief Collaboratively store data into a shared tile from register tiles split across a warpgroup.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::st::all ST, ducks::rt::all RT>
__device__ inline static void store(ST &dst, const RT &src) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%N_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");
    int local_warpid = warpid();
    using T2 = RT::dtype;
    using U  = ST::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = ::kittens::laneid();
    #pragma unroll
    for(int i = 0; i < warp_height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 2) {
                // handle the row-major layout
                int row = (local_warpid*warp_height + i)*src.tile_size + (warp_laneid / 4);
                int col = j*src.tile_size + 2*(warp_laneid % 4);
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
                move<U2>::sts(&dst[{row+0, col+0}], tmp[0]);
                move<U2>::sts(&dst[{row+8, col+0}], tmp[1]);
                move<U2>::sts(&dst[{row+0, col+8}], tmp[2]);
                move<U2>::sts(&dst[{row+8, col+8}], tmp[3]);
            }
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 4) {
                // handle the row-major layout for 32-bit types
                int row = (local_warpid*warp_height + i)*src.tile_size + (warp_laneid / 4);
                int col = j*src.tile_size + 2*(warp_laneid % 4);
                int blit = (warp_laneid%4) / 2;
                T2 reg_tmp[4];
                if(blit) {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        reg_tmp[k] = T2{src.tiles[i][j].data[k].y, src.tiles[i][j].data[k].x};
                    }
                }
                else {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        reg_tmp[k] = src.tiles[i][j].data[k];
                    }
                }
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(reg_tmp[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(reg_tmp[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(reg_tmp[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(reg_tmp[3]);
                move<U>::sts(&dst[{row+0, (col+0)^blit}], tmp[0].x);
                move<U>::sts(&dst[{row+0, (col+1)^blit}], tmp[0].y);
                move<U>::sts(&dst[{row+8, (col+0)^blit}], tmp[1].x);
                move<U>::sts(&dst[{row+8, (col+1)^blit}], tmp[1].y);
                move<U>::sts(&dst[{row+0, (col+8)^blit}], tmp[2].x);
                move<U>::sts(&dst[{row+0, (col+9)^blit}], tmp[2].y);
                move<U>::sts(&dst[{row+8, (col+8)^blit}], tmp[3].x);
                move<U>::sts(&dst[{row+8, (col+9)^blit}], tmp[3].y);
            }
            else {
                // handle the column-major layout
                int row = (local_warpid*warp_height + i)*src.tile_size + 2*(warp_laneid % 4);
                int col = j*src.tile_size + (warp_laneid / 4);
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
                move<U>::sts(&dst[{row+0, col+0}], tmp[0].x);
                move<U>::sts(&dst[{row+1, col+0}], tmp[0].y);
                move<U>::sts(&dst[{row+0, col+8}], tmp[1].x);
                move<U>::sts(&dst[{row+1, col+8}], tmp[1].y);
                move<U>::sts(&dst[{row+8, col+0}], tmp[2].x);
                move<U>::sts(&dst[{row+9, col+0}], tmp[2].y);
                move<U>::sts(&dst[{row+8, col+8}], tmp[3].x);
                move<U>::sts(&dst[{row+9, col+8}], tmp[3].y);
            }
        }
    }
}