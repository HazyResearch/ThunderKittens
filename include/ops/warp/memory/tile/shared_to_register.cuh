/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

// These probably need to be redone to reduce bank conflicts.
// They currently work fine with xor layout but it should be
// possible to reduce their bank conflicts with other layouts too.

/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                // handle the row-major layout
                int row = i*dst.tile_size + (laneid / 4);
                int col = j*dst.tile_size + 2*(laneid % 4);
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row+0, col+0}]));
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row+8, col+0}]));
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row+0, col+8}]));
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row+8, col+8}]));
            }
            else {
                // handle the column-major layout
                int row = i*dst.tile_size + 2*(laneid % 4);
                int col = j*dst.tile_size + (laneid / 4);
                dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src[{row+0, col+0}]);
                dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src[{row+1, col+0}]);
                dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src[{row+0, col+8}]);
                dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src[{row+1, col+8}]);
                dst.tiles[i][j].data[2].x = base_types::convertor<T, U>::convert(src[{row+8, col+0}]);
                dst.tiles[i][j].data[2].y = base_types::convertor<T, U>::convert(src[{row+9, col+0}]);
                dst.tiles[i][j].data[3].x = base_types::convertor<T, U>::convert(src[{row+8, col+8}]);
                dst.tiles[i][j].data[3].y = base_types::convertor<T, U>::convert(src[{row+9, col+8}]);
            }
        }
    }
}

/**
 * @brief Store data into a shared tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void store(ST &dst, const RT &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                // handle the row-major layout
                int row = i*src.tile_size + (laneid / 4);
                int col = j*src.tile_size + 2*(laneid % 4);
                *(U2*)(&dst[{row+0, col+0}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                *(U2*)(&dst[{row+8, col+0}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                *(U2*)(&dst[{row+0, col+8}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                *(U2*)(&dst[{row+8, col+8}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
            }
            else {
                // handle the column-major layout
                int row = i*src.tile_size + 2*(laneid % 4);
                int col = j*src.tile_size + (laneid / 4);
                dst[{row+0, col+0}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
                dst[{row+1, col+0}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
                dst[{row+0, col+8}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
                dst[{row+1, col+8}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
                dst[{row+8, col+0}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x);
                dst[{row+9, col+0}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y);
                dst[{row+8, col+8}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x);
                dst[{row+9, col+8}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y);
            }
        }
    }
}

}