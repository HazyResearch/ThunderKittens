/**
 * @file
 * @brief Conversions on vectors stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

namespace detail {

// i am not smart enough to figure out these indices without these helpers :/
// again, blame nvidia for these stupid, stupid layouts
__device__ static inline int row_from_indices_dim2(int laneid, int inner_dim, int x_or_y) {
    return 8*inner_dim + (laneid%4)*2 + x_or_y;
}
__device__ static inline int row_from_indices_dim1(int laneid, int x_or_y) {
    return 8*x_or_y + (laneid/4);
}
__device__ static inline int canonical_src_lane_dim2(int row) {
    return (row/2)%4 + 4*(row%2); // draw even rows from 0...3 and odds from 4...7
}
__device__ static inline int canonical_src_lane_dim1(int row) {
    return (row*4)%32;
}

}

/**
 * @brief Copies data from one register vector to another.
 *
 * @tparam RV1 The type of the destination register vector.
 * @tparam RV2 The type of the source register vector.
 * @param dst[out] The destination register vector.
 * @param src[in] The source register vector to copy from.
 */
template<ducks::rv::all RV1, ducks::rv::all RV2>
__device__ static inline void copy(RV1 &dst, const RV2 &src) {
    static_assert(RV1::outer_dim == RV2::outer_dim, "Outer dimensions of the register vectors must be the same.");
    using D1 = RV1::dtype;
    using D2 = RV2::dtype;
    if constexpr (RV1::inner_dim == RV2::inner_dim) {
        #pragma unroll
        for(int i = 0; i < RV1::outer_dim; i++) {
            #pragma unroll
            for(int j = 0; j < RV1::inner_dim; j++) {
                dst[i][j] = base_types::convertor<D1, D2>::convert(src[i][j]);
            }
        }
    }
    // Inner dimensions are not the same, this is really a layout conversion.
    else if constexpr (RV1::inner_dim == 1 && RV2::inner_dim == 2) {
        // Convert from an unaligned vector layout to an aligned vector layout.
        int laneid = kittens::laneid();
        #pragma unroll
        for(int i = 0; i < RV1::outer_dim; i++) {
            dst[i][0].x = packed_shfl_sync(
                kittens::MASK_ALL,
                laneid < 4 ? src[i][0].x : src[i][0].y, // mirrors canonical_src_lane_dim2
                detail::canonical_src_lane_dim2(detail::row_from_indices_dim1(laneid, 0))
            );
            dst[i][0].y = packed_shfl_sync(
                kittens::MASK_ALL,
                laneid < 4 ? src[i][1].x : src[i][1].y, // mirrors canonical_src_lane_dim2
                detail::canonical_src_lane_dim2(detail::row_from_indices_dim1(laneid, 1))
            );
        }
    }
    else if constexpr (RV1::inner_dim == 2 && RV2::inner_dim == 1) {
        // Convert from an aligned vector layout to an unaligned vector layout.
        int laneid = kittens::laneid();
        #pragma unroll
        for(int i = 0; i < RV1::outer_dim; i++) {
            dst[i][0].x = packed_shfl_sync(
                kittens::MASK_ALL,
                src[i][0].x, // first 8 rows
                detail::canonical_src_lane_dim1(detail::row_from_indices_dim2(laneid, 0, 0))
            );
            dst[i][0].y = packed_shfl_sync(
                kittens::MASK_ALL,
                src[i][0].x, // first 8 rows
                detail::canonical_src_lane_dim1(detail::row_from_indices_dim2(laneid, 0, 1))
            );
            dst[i][1].x = packed_shfl_sync(
                kittens::MASK_ALL,
                src[i][0].y, // last 8 rows
                detail::canonical_src_lane_dim1(detail::row_from_indices_dim2(laneid, 1, 0))
            );
            dst[i][1].y = packed_shfl_sync(
                kittens::MASK_ALL,
                src[i][0].y, // last 8 rows
                detail::canonical_src_lane_dim1(detail::row_from_indices_dim2(laneid, 1, 1))
            );
        }
    }
    else {
        static_assert(RV1::inner_dim == RV2::inner_dim, "Something has gone deeply wrong with how register vectors were instantiated.");
    }
}

} // namespace kittens