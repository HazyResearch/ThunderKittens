#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

/* ----------  COPIES  ---------- */

template<typename T, typename U, int _height, int _width, st_layout L1, st_layout L2>
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

}