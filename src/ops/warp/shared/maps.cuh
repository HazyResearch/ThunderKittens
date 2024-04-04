#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

/* ----------  Uniform tile maps (independent of layout)  ---------- */

// Uniform unary map on tile
template<typename op, ducks::st::all T> // T2, w, h can be inferred from dst as long as op is specialized
__device__ static inline void unary_map(T &dst, const T &src) {

    int lane = threadIdx.x % 32; 
    #pragma unroll
    for(int i = lane; i < dst.rows*dst.cols; i += WARP_SIZE) {
        int row = (i/dst.cols); 
        int col = (i%dst.cols); 
        dst.data[col + (row * dst.cols)] = op::template op<typename T::dtype>(src.data[col + (row * dst.cols)]);
    }
}

// Uniform map from packed to tile
template<typename op, ducks::st::all T>
__device__ static inline void bin_map(T &dst, const T &src, const typename T::dtype &param) {

    int lane = threadIdx.x % 32; 
    #pragma unroll
    for(int i = lane; i < dst.rows*dst.cols; i += WARP_SIZE) {
        int row = (i/dst.cols); 
        int col = (i%dst.cols);  
        dst.data[col + (row * dst.cols)] = op::template op<typename T::dtype>(src.data[col + (row * dst.cols)], param);
    }
}
// Uniform map from single to tile
template<typename op, ducks::st::all T>
__device__ static inline void bin_map(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    // The optimizing compiler should eliminate this pack in the 32-bit case but not in the 16-bit case
    bin_map<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}
// Tile-to-tile element-wise map. Only enforces same layout.
template<typename op, ducks::st::all T>
__device__ static inline void bin_map(T &dst, const T &lhs, const T &rhs) {

    int lane = threadIdx.x % 32; 
    #pragma unroll
    for(int i = lane; i < dst.rows*dst.cols; i += WARP_SIZE) {
        int row = (i/dst.cols); 
        int col = (i%dst.cols); 
        dst.data[col + (row * dst.cols)] = op::template op<typename T::dtype>(lhs.data[col + (row * dst.cols)], rhs.data[col + (row * dst.cols)]);
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during compile-time.
// So, syntax should just be kittens::add_row(tile, colvec);

// const maps
template<ducks::st::all T>
__device__ static inline void zero(T &dst) {
    unary_map<base_ops::zero, T>(dst, dst);
}
template<ducks::st::all T>
__device__ static inline void one(T &dst) {
    unary_map<base_ops::one, T>(dst, dst);
}
template<ducks::st::all T>
__device__ static inline void pos_infty(T &dst) {
    unary_map<base_ops::pos_infty, T>(dst, dst);
}
template<ducks::st::all T>
__device__ static inline void neg_infty(T &dst) {
    unary_map<base_ops::neg_infty, T>(dst, dst);
}

// unary maps
template<ducks::st::all T>
__device__ static inline void exp(T &dst, const T &src) {
    unary_map<base_ops::exp, T>(dst, src);
}
template<ducks::st::all T>
__device__ static inline void abs(T &dst, const T &src) {
    unary_map<base_ops::abs, T>(dst, src);
}
template<ducks::st::all T>
__device__ static inline void relu(T &dst, const T &src) {
    unary_map<base_ops::relu, T>(dst, src);
}
template<ducks::st::all T, typename U>
__device__ static inline void copy(T &dst, const U &src) {
    bin_map<base_ops::copy2, T>(dst, src);
}

// uniform binary maps
template<ducks::st::all T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::max, T>(dst, lhs, rhs);
}
template<ducks::st::all T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::min, T>(dst, lhs, rhs);
}
template<ducks::st::all T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}
template<ducks::st::all T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}
template<ducks::st::all T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}
template<ducks::st::all T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::div, T>(dst, lhs, rhs);
}

}