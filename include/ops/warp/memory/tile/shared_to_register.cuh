/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

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
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 2) {
                // handle the row-major layout for 16-bit types
                int row = i*dst.tile_size + (laneid / 4);
                int col = j*dst.tile_size + 2*(laneid % 4);
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
                int row = i*dst.tile_size + (laneid / 4);
                int col = j*dst.tile_size + 2*(laneid % 4);
                int blit = (laneid%4)/2;
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
                        dst.tiles[i][j].data[k] = {dst.tiles[i][j].data[k].y, dst.tiles[i][j].data[k].x};
                    }
                }
            }
            else {
                // handle the column-major layout
                int row = i*dst.tile_size + 2*(laneid % 4);
                int col = j*dst.tile_size + (laneid / 4);
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
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 2) {
                // handle the row-major layout
                int row = i*src.tile_size + (laneid / 4);
                int col = j*src.tile_size + 2*(laneid % 4);
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
                int row = i*src.tile_size + (laneid / 4);
                int col = j*src.tile_size + 2*(laneid % 4);
                int blit = (laneid%4) / 2;
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
                int row = i*src.tile_size + 2*(laneid % 4);
                int col = j*src.tile_size + (laneid / 4);
                U tmp[8];
                tmp[0] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
                tmp[1] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
                tmp[2] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
                tmp[3] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
                tmp[4] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x);
                tmp[5] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y);
                tmp[6] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x);
                tmp[7] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y);
                move<U>::sts(&dst[{row+0, col+0}], tmp[0]);
                move<U>::sts(&dst[{row+1, col+0}], tmp[1]);
                move<U>::sts(&dst[{row+0, col+8}], tmp[2]);
                move<U>::sts(&dst[{row+1, col+8}], tmp[3]);
                move<U>::sts(&dst[{row+8, col+0}], tmp[4]);
                move<U>::sts(&dst[{row+9, col+0}], tmp[5]);
                move<U>::sts(&dst[{row+8, col+8}], tmp[6]);
                move<U>::sts(&dst[{row+9, col+8}], tmp[7]);
            }
        }
    }
}

}