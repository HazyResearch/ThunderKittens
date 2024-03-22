#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

/* ----------  COPIES  ---------- */

/**
 * @brief Copies elements from source to destination with optional type conversion.
 *
 * @tparam T Destination type.
 * @tparam U Source type.
 * @tparam _height Height of the source and destination.
 * @tparam _width Width of the source and destination.
 * @tparam L1 Layout of the destination.
 * @tparam L2 Layout of the source.
 * @param dst Reference to the destination.
 * @param src Reference to the source.
 */
template<typename T, typename U, int _height, int _width, ducks::st_layout::all L1, ducks::st_layout::all L2>
__device__ static inline void copy(st<T, _height, _width, L1> &dst, const st<U, _height, _width, L2> &src) {
    using T2 = base_types::packing<T>::packed_type;
    using U2 = base_types::packing<U>::packed_type;
    int lane = threadIdx.x % 32;
    if constexpr (std::is_same_v<L1, L2>) { // if same layout can just do a simple copy
        #pragma unroll
        for(int i = lane; i < dst.rows*dst.cols; i+=WARP_SIZE) {
            dst[i] = base_types::convertor<T, U>::convert(src[i]);
        }
    }
    else { // otherwise we need to actually do indexing calculations :(
        #pragma unroll
        for(int i = lane; i < dst.rows*dst.cols; i+=WARP_SIZE) {
            int row = i/dst.cols;
            int col = i%dst.cols;
            dst[{row, col}] = base_types::convertor<T, U>::convert(src[{row, col}]);
        }
    }
}

/* ----------  SUBTILE  ---------- */

/**
 * @brief Creates a subtile view into a larger tile, allowing for operations on a sub-section.
 *
 * @tparam subtile_height Height of the subtile.
 * @tparam subtile_width Width of the subtile.
 * @tparam ST Shared tile type.
 * @param src Reference to the source shared tile.
 * @param tile_row_offset Row offset into the source tile.
 * @param tile_col_offset Column offset into the source tile.
 * @return A subtile view into the source tile.
 */
template<int subtile_height, int subtile_width, ducks::st::all ST>
__device__ inline typename ST::subtile<subtile_height, subtile_width> subtile_inplace(ST &src, int tile_row_offset, int tile_col_offset) {
    return typename ST::subtile<subtile_height, subtile_width>(
        &src[0], subtile_height*16*tile_row_offset, subtile_width*16*tile_col_offset
    );
}

}
