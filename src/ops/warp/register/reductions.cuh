#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

// Row reduction. Written to give (hopefully) better ILP than before.
// row-major layout
template<typename op, ducks::rv::col_vec V, ducks::rt::row_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(std::is_same_v<typename V::layout, typename T::layout>); // compatible layout
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = V::dtype;

    const int leader = threadIdx.x & 0x1C; // 11100 in binary
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        dtype accum_top_row    = op::template op<dtype>(src.tiles[i][0].data[0], src.tiles[i][0].data[2]);
        dtype accum_bottom_row = op::template op<dtype>(src.tiles[i][0].data[1], src.tiles[i][0].data[3]);
        #pragma unroll
        for(int j = 1; j < src.width; j++) {
            #pragma unroll
            for(int k = 0; k < src.packed_per_tile; k+=2) {
                accum_top_row    = op::template op<dtype>(accum_top_row,    src.tiles[i][j].data[k+0]);
                accum_bottom_row = op::template op<dtype>(accum_bottom_row, src.tiles[i][j].data[k+1]);
            }
        }
        dtype accum_packed;
        accum_packed.x = op::template op<base_types::packing<dtype>::unpacked_type>(accum_top_row.x,    accum_top_row.y);
        accum_packed.y = op::template op<base_types::packing<dtype>::unpacked_type>(accum_bottom_row.x, accum_bottom_row.y);

        // Now we need to do a lil shuffle to make everyone happy.

        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 2));
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 1));

        accum_packed = packed_shfl_sync(MASK_ALL, accum_packed, leader);

        if(reset) {
            row_accum[i][0] = accum_packed;
        }
        else {
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_packed);
        }
    }
}
// col-major layout
template<typename op, ducks::rv::col_vec V, ducks::rt::col_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(std::is_same_v<typename V::layout, typename T::layout>); // compatible layout
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = V::dtype;

    const int leader = threadIdx.x & 0x3; // 00011 in binary
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        dtype accum_top_rows    = op::template op<dtype>(src.tiles[i][0].data[0], src.tiles[i][0].data[1]);
        dtype accum_bottom_rows = op::template op<dtype>(src.tiles[i][0].data[2], src.tiles[i][0].data[3]);
        #pragma unroll
        for(int j = 1; j < src.width; j++) {
            #pragma unroll
            for(int k = 0; k < src.packed_per_tile/2; k++) {
                accum_top_rows    = op::template op<dtype>(accum_top_rows,    src.tiles[i][j].data[k+0]);
                accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, src.tiles[i][j].data[k+2]);
            }
        }

        // Now we need to do a lil shuffle to make everyone happy.

        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 16));
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 8));
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 4));

        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 16));
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 8));
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 4));

        accum_top_rows    = packed_shfl_sync(MASK_ALL, accum_top_rows,    leader);
        accum_bottom_rows = packed_shfl_sync(MASK_ALL, accum_bottom_rows, leader);

        if(reset) {
            row_accum[i][0] = accum_top_rows;
            row_accum[i][1] = accum_bottom_rows;
        }
        else {
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_top_rows);
            row_accum[i][1] = op::template op<dtype>(src_accum[i][1], accum_bottom_rows);
        }
    }
}

// Col reduction.
// row-major layout
template<typename op, ducks::rv::row_vec V, ducks::rt::row_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(std::is_same_v<typename V::layout, typename T::layout>); // compatible layout
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = V::dtype;

    const int leader = threadIdx.x & 0x3; // 00011 in binary
    #pragma unroll
    for(int j = 0; j < src.width; j++) {
        dtype accum_left_cols  = op::template op<dtype>(src.tiles[0][j].data[0], src.tiles[0][j].data[1]);
        dtype accum_right_cols = op::template op<dtype>(src.tiles[0][j].data[2], src.tiles[0][j].data[3]);
        #pragma unroll
        for(int i = 1; i < src.height; i++) {
            #pragma unroll
            for(int k = 0; k < src.packed_per_tile/2; k++) {
                accum_left_cols  = op::template op<dtype>(accum_left_cols,  src.tiles[i][j].data[k+0]);
                accum_right_cols = op::template op<dtype>(accum_right_cols, src.tiles[i][j].data[k+2]);
            }
        }

        // Now we need to do a lil shuffle to make everyone happy.

        accum_left_cols = op::template op<dtype>(accum_left_cols, packed_shfl_down_sync(MASK_ALL, accum_left_cols, 16));
        accum_left_cols = op::template op<dtype>(accum_left_cols, packed_shfl_down_sync(MASK_ALL, accum_left_cols, 8));
        accum_left_cols = op::template op<dtype>(accum_left_cols, packed_shfl_down_sync(MASK_ALL, accum_left_cols, 4));

        accum_right_cols = op::template op<dtype>(accum_right_cols, packed_shfl_down_sync(MASK_ALL, accum_right_cols, 16));
        accum_right_cols = op::template op<dtype>(accum_right_cols, packed_shfl_down_sync(MASK_ALL, accum_right_cols, 8));
        accum_right_cols = op::template op<dtype>(accum_right_cols, packed_shfl_down_sync(MASK_ALL, accum_right_cols, 4));

        accum_left_cols  = packed_shfl_sync(MASK_ALL, accum_left_cols,  leader);
        accum_right_cols = packed_shfl_sync(MASK_ALL, accum_right_cols, leader);

        if(reset) {
            col_accum[j][0] = accum_left_cols;
            col_accum[j][1] = accum_right_cols;
        }
        else {
            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_left_cols);
            col_accum[j][1] = op::template op<dtype>(src_accum[j][1], accum_right_cols);
        }
    }
}
// col-major layout
template<typename op, ducks::rv::row_vec V, ducks::rt::col_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(std::is_same_v<typename V::layout, typename T::layout>); // compatible layout
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = V::dtype;
    const int leader = threadIdx.x & 0x1C; // 11100 in binary
    #pragma unroll
    for(int j = 0; j < src.width; j++) { // note now width is the outer loop
        dtype accum_left_col  = op::template op<dtype>(src.tiles[0][j].data[0], src.tiles[0][j].data[2]);
        dtype accum_right_col = op::template op<dtype>(src.tiles[0][j].data[1], src.tiles[0][j].data[3]);
        #pragma unroll
        for(int i = 1; i < src.height; i++) { // and height is the inner loop
            #pragma unroll
            for(int k = 0; k < src.packed_per_tile; k+=2) {
                accum_left_col  = op::template op<dtype>(accum_left_col,  src.tiles[i][j].data[k+0]);
                accum_right_col = op::template op<dtype>(accum_right_col, src.tiles[i][j].data[k+1]);
            }
        }
        dtype accum_packed;
        accum_packed.x = op::template op<base_types::packing<dtype>::unpacked_type>(accum_left_col.x,  accum_left_col.y);
        accum_packed.y = op::template op<base_types::packing<dtype>::unpacked_type>(accum_right_col.x, accum_right_col.y);

        // Now we need to do a lil shuffle to make everyone happy.

        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 2));
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 1));

        accum_packed = packed_shfl_sync(MASK_ALL, accum_packed, leader);

        if(reset) {
            col_accum[j][0] = accum_packed;
        }
        else {
            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_packed);
        }
    }
}


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// two-operand row reductions. (Accumulate and REPLACE.)
template<ducks::rv::col_vec V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src)  {
    row_reduce<base_ops::max, V, T, true>(row_accum, src, row_accum);
}
template<ducks::rv::col_vec V, ducks::rt::all T>
__device__ static inline void row_min(V &row_accum, const T &src)  {
    row_reduce<base_ops::min, V, T, true>(row_accum, src, row_accum);
}
template<ducks::rv::col_vec V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src)  {
    row_reduce<base_ops::sum, V, T, true>(row_accum, src, row_accum);
}
template<ducks::rv::col_vec V, ducks::rt::all T>
__device__ static inline void row_prod(V &row_accum, const T &src) {
    row_reduce<base_ops::mul, V, T, true>(row_accum, src, row_accum);
}
// three-operand row reductions. (Accumulate ONTO.)
template<ducks::rv::col_vec V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}
template<ducks::rv::col_vec V, ducks::rt::all T>
__device__ static inline void row_min(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::min, V, T, false>(row_accum, src, src_accum);
}
template<ducks::rv::col_vec V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}
template<ducks::rv::col_vec V, ducks::rt::all T>
__device__ static inline void row_prod(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::mul, V, T, false>(row_accum, src, src_accum);
}

// two-operand col reductions. (Accumulate and REPLACE.)
template<ducks::rv::row_vec V, ducks::rt::all T>
__device__ static inline void col_max(V &col_accum, const T &src)  {
    col_reduce<base_ops::max, V, T, true>(col_accum, src, col_accum);
}
template<ducks::rv::row_vec V, ducks::rt::all T>
__device__ static inline void col_min(V &col_accum, const T &src)  {
    col_reduce<base_ops::min, V, T, true>(col_accum, src, col_accum);
}
template<ducks::rv::row_vec V, ducks::rt::all T>
__device__ static inline void col_sum(V &col_accum, const T &src)  {
    col_reduce<base_ops::sum, V, T, true>(col_accum, src, col_accum);
}
template<ducks::rv::row_vec V, ducks::rt::all T>
__device__ static inline void col_prod(V &col_accum, const T &src) {
    col_reduce<base_ops::mul, V, T, true>(col_accum, src, col_accum);
}
// three-operand col reductions. (Accumulate ONTO.)
template<ducks::rv::row_vec V, ducks::rt::all T>
__device__ static inline void col_max(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::max, V, T, false>(col_accum, src, src_accum);
}
template<ducks::rv::row_vec V, ducks::rt::all T>
__device__ static inline void col_min(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::min, V, T, false>(col_accum, src, src_accum);
}
template<ducks::rv::row_vec V, ducks::rt::all T>
__device__ static inline void col_sum(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::sum, V, T, false>(col_accum, src, src_accum);
}
template<ducks::rv::row_vec V, ducks::rt::all T>
__device__ static inline void col_prod(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::mul, V, T, false>(col_accum, src, src_accum);
}

}