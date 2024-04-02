#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

/* ----------  COPIES  ---------- */

template<typename T, typename U, int _height, int _width, ducks::st_layout::all L1, ducks::st_layout::all L2>
__device__ static inline void copy(st<T, _height, _width, L1> &dst, const st<U, _height, _width, L2> &src) {
    using T2 = base_types::packing<T>::packed_type;
    using U2 = base_types::packing<U>::packed_type;
    int lane = threadIdx.x % 32;
    if constexpr (std::is_same_v<L1, L2>) { // if same layout can just do a simple copy
        #pragma unroll
        for(int i = lane; i < dst.rows*dst.cols; i+=WARP_THREADS) {
            dst[i] = base_types::convertor<T, U>::convert(src[i]);
        }
    }
    else { // otherwise we need to actually do indexing calculations :(
        #pragma unroll
        for(int i = lane; i < dst.rows*dst.cols; i+=WARP_THREADS) {
            int row = i/dst.cols;
            int col = i%dst.cols;
            dst[{row, col}] = base_types::convertor<T, U>::convert(src[{row, col}]);
        }
    }
}

/* ----------  SUBTILE  ---------- */

template<int subtile_height, int subtile_width, ducks::st::all ST>
__device__ inline typename ST::subtile<subtile_height, subtile_width> subtile_inplace(ST &src, int tile_row_offset, int tile_col_offset) {
    return typename ST::subtile<subtile_height, subtile_width>(
        &src[0], subtile_height*16*tile_row_offset, subtile_width*16*tile_col_offset
    );
}

/* ----------  SUBVEC  ---------- */

template<int subvec_length, ducks::sv::all SV>
__device__ inline typename SV::subvec<subvec_length> &subvec_inplace(SV &src, int vec_offset) {
    return *(typename SV::subvec<subvec_length>*)(&src[vec_offset*16*subvec_length]);
}

}