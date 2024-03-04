#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

/* ----------  Uniform tile maps (independent of layout)  ---------- */

// Uniform unary map on tile
template<typename op, rt_type T> // T2, w, h can be inferred from dst as long as op is specialized
__device__ static inline void unary_map(T &dst, const T &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k]);
            }
        }
    }
}

// Uniform map from packed to tile
template<typename op, rt_type T>
__device__ static inline void bin_map(T &dst, const T &src, const typename T::dtype &param) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k], param);
            }
        }
    }
}
// Uniform map from single to tile
template<typename op, rt_type T>
__device__ static inline void bin_map(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    // The optimizing compiler should eliminate this pack in the 32-bit case but not in the 16-bit case
    bin_map<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}
// Tile-to-tile element-wise map. Only enforces same layout.
template<typename op, rt_type T>
__device__ static inline void bin_map(T &dst, const T &lhs, const T &rhs) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(lhs.tiles[i][j].data[k], rhs.tiles[i][j].data[k]);
            }
        }
    }
}

/* ----------  Row tile maps  ----------*/

// row-major layout
template<typename op, rt_type_rowlayout T, rt_col_vec_type V>
__device__ static inline void row_map(T &dst, const T &src, const V &row_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(std::is_same_v<typename V::layout, typename T::layout>); // compatible layout
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        dtype packed_top_row    = base_types::packing<dtype>::pack(row_values[i][0].x); //  first value in eager mode
        dtype packed_bottom_row = base_types::packing<dtype>::pack(row_values[i][0].y); // second value in eager mode
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], packed_top_row);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(src.tiles[i][j].data[k+1], packed_bottom_row);
            }
        }
    }
}
// col-major layout
template<typename op, rt_type_collayout T, rt_col_vec_type V>
__device__ static inline void row_map(T &dst, const T &src, const V &row_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(std::is_same_v<typename V::layout, typename T::layout>); // compatible layout
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], row_values[i][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(src.tiles[i][j].data[k+2], row_values[i][1]);
            }
        }
    }
}


// Three-operand row map
// row-major layout
template<typename op, rt_type_rowlayout T, rt_col_vec_type V>
__device__ static inline void row_map(T &dst, const T &a, const T &b, const V &row_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(std::is_same_v<typename V::layout, typename T::layout>); // compatible layout
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        dtype packed_top_row    = base_types::packing<dtype>::pack(row_values[i][0].x); //  first value in eager mode
        dtype packed_bottom_row = base_types::packing<dtype>::pack(row_values[i][0].y); // second value in eager mode
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], packed_top_row);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(a.tiles[i][j].data[k+1], b.tiles[i][j].data[k+1], packed_bottom_row);
            }
        }
    }
}
// col-major layout
template<typename op, rt_type_collayout T, rt_col_vec_type V>
__device__ static inline void row_map(T &dst, const T &a, const T &b, const V &row_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(std::is_same_v<typename V::layout, typename T::layout>); // compatible layout
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], row_values[i][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(a.tiles[i][j].data[k+2], b.tiles[i][j].data[k+2], row_values[i][1]);
            }
        }
    }
}

/* ----------  Col major tile maps  ----------*/

// row-major layout
template<typename op, rt_type_rowlayout T, rt_row_vec_type V>
__device__ static inline void col_map(T &dst, const T &src, const V &col_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(std::is_same_v<typename V::layout, typename T::layout>); // compatible layout
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], col_values[j][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(src.tiles[i][j].data[k+2], col_values[j][1]);
            }
        }
    }
}
// col-major layout
template<typename op, rt_type_collayout T, rt_row_vec_type V>
__device__ static inline void col_map(T &dst, const T &src, const V &col_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(std::is_same_v<typename V::layout, typename T::layout>); // compatible layout
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        dtype packed_left_col  = base_types::packing<dtype>::pack(col_values[j][0].x); //  first value in eager mode
        dtype packed_right_col = base_types::packing<dtype>::pack(col_values[j][0].y); // second value in eager mode
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], packed_left_col);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(src.tiles[i][j].data[k+1], packed_right_col);
            }
        }
    }
}

// Three-operand col map
// row-major layout
template<typename op, rt_type_rowlayout T, rt_row_vec_type V>
__device__ static inline void col_map(T &dst, const T &a, const T &b, const V &col_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(std::is_same_v<typename V::layout, typename T::layout>); // compatible layout
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], col_values[j][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(a.tiles[i][j].data[k+2], b.tiles[i][j].data[k+2], col_values[j][1]);
            }
        }
    }
}
// col-major layout
template<typename op, rt_type_collayout T, rt_row_vec_type V>
__device__ static inline void col_map(T &dst, const T &a, const T &b, const V &col_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(std::is_same_v<typename V::layout, typename T::layout>); // compatible layout
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = T::dtype;
    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        dtype packed_left_col  = base_types::packing<dtype>::pack(col_values[j][0].x); //  first value in eager mode
        dtype packed_right_col = base_types::packing<dtype>::pack(col_values[j][0].y); // second value in eager mode
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], packed_left_col);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(a.tiles[i][j].data[k+1], b.tiles[i][j].data[k+1], packed_right_col);
            }
        }
    }
}


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during compile-time.
// So, syntax should just be kittens::add_row(tile, colvec);

// const maps
template<rt_type T>
__device__ static inline void zero(T &dst) {
    unary_map<base_ops::zero, T>(dst, dst);
}
template<rt_type T>
__device__ static inline void one(T &dst) {
    unary_map<base_ops::one, T>(dst, dst);
}
template<rt_type T>
__device__ static inline void pos_infty(T &dst) {
    unary_map<base_ops::pos_infty, T>(dst, dst);
}
template<rt_type T>
__device__ static inline void neg_infty(T &dst) {
    unary_map<base_ops::neg_infty, T>(dst, dst);
}

// unary maps
template<rt_type T>
__device__ static inline void exp(T &dst, const T &src) {
    unary_map<base_ops::exp, T>(dst, src);
}
template<rt_type T>
__device__ static inline void abs(T &dst, const T &src) {
    unary_map<base_ops::abs, T>(dst, src);
}
template<rt_type T>
__device__ static inline void relu(T &dst, const T &src) {
    unary_map<base_ops::relu, T>(dst, src);
}
template<rt_type T, typename U>
__device__ static inline void copy2(T &dst, const U &src) {
    bin_map<base_ops::copy, T>(dst, src);
}

// uniform binary maps
template<rt_type T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::max, T>(dst, lhs, rhs);
}
template<rt_type T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::min, T>(dst, lhs, rhs);
}
template<rt_type T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}
template<rt_type T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}
template<rt_type T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}
template<rt_type T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::div, T>(dst, lhs, rhs);
}

// row maps
template<rt_type T, rt_col_vec_type V>
__device__ static inline void add_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sum, T, V>(dst, src, row_values);
}
template<rt_type T, rt_col_vec_type V>
__device__ static inline void sub_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sub, T, V>(dst, src, row_values);
}
template<rt_type T, rt_col_vec_type V>
__device__ static inline void mul_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::mul, T, V>(dst, src, row_values);
}
template<rt_type T, rt_col_vec_type V>
__device__ static inline void div_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::div, T, V>(dst, src, row_values);
}

// col maps
template<rt_type T, rt_row_vec_type V>
__device__ static inline void add_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sum, T, V>(dst, src, col_values);
}
template<rt_type T, rt_row_vec_type V>
__device__ static inline void sub_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sub, T, V>(dst, src, col_values);
}
template<rt_type T, rt_row_vec_type V>
__device__ static inline void mul_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::mul, T, V>(dst, src, col_values);
}
template<rt_type T, rt_row_vec_type V>
__device__ static inline void div_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::div, T, V>(dst, src, col_values);
}

}