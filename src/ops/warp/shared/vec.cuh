#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"


namespace kittens {


template<typename op, ducks::sv::all T>
__device__ static inline void unary_op(T &dst, const T &src) {
    __syncwarp();
    #pragma unroll
    for(auto cur = laneid(); cur < T::length; cur+=WARP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(src[cur]);
    }
}
template<typename op, ducks::sv::all T>
__device__ static inline void bin_op(T &dst, const T &lhs, const T &rhs) {
    __syncwarp();
    #pragma unroll
    for(auto cur = laneid(); cur < T::length; cur+=WARP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(lhs[cur], rhs[cur]);
    }
}
// And for passing in single floats / bf16's
template<typename op, ducks::sv::all T>
__device__ static inline void bin_op(T &dst, const T &src, const typename T::dtype &param) {
    __syncwarp();
    #pragma unroll
    for(auto cur = laneid(); cur < T::length; cur+=WARP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(src[cur], param);
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// ---- const ops ----

template<ducks::sv::all T>
__device__ static inline void zero(T &dst) {
    unary_op<base_ops::zero, T>(dst, dst);
}
template<ducks::sv::all T>
__device__ static inline void one(T &dst) {
    unary_op<base_ops::one, T>(dst, dst);}
template<ducks::sv::all T>
__device__ static inline void pos_infty(T &dst) {
    unary_op<base_ops::pos_infty, T>(dst, dst);
}
template<ducks::sv::all T>
__device__ static inline void neg_infty(T &dst) {
    unary_op<base_ops::neg_infty, T>(dst, dst);
}

// ---- unary ops ----

template<ducks::sv::all T, typename U>
__device__ static inline void copy(T &dst, const U &src) {
    bin_op<base_ops::copy2, T>(dst, dst, src); // the second arg is ignored here.
}
template<ducks::sv::all T>
__device__ static inline void exp(T &dst, const T &src) {
    unary_op<base_ops::exp, T>(dst, src);
}
template<ducks::sv::all T>
__device__ static inline void abs(T &dst, const T &src) {
    unary_op<base_ops::abs, T>(dst, src);
}
template<ducks::sv::all T>
__device__ static inline void relu(T &dst, const T &src) {
    unary_op<base_ops::relu, T>(dst, src);
}

// ---- binary ops ----

template<ducks::sv::all T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::max, T>(dst, lhs, rhs);
}
template<ducks::sv::all T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::min, T>(dst, lhs, rhs);
}
template<ducks::sv::all T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sum, T>(dst, lhs, rhs);
}
template<ducks::sv::all T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sub, T>(dst, lhs, rhs);
}
template<ducks::sv::all T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::mul, T>(dst, lhs, rhs);
}
template<ducks::sv::all T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::div, T>(dst, lhs, rhs);
}


}
