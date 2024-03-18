#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"


namespace kittens {


template<typename op, ducks::sv::all T>
__device__ 
__device__ static inline void st_unary_map(T &dst) {
    __syncwarp();
    #pragma unroll
    for(auto cur = laneid(); cur < T::length; cur+=WARP_SIZE) {
        auto col       = cur % T::length;
        auto row       = cur / T::length;
        auto idx       = row*T::length + col;
        dst[idx] = op::template op<typename T::dtype>(dst[idx]);
    }
}

template<ducks::sv::all T>
__device__ static inline void zero(T &dst)      { st_unary_map<base_ops::zero, T>(dst);      }
template<ducks::sv::all T>
__device__ static inline void one(T &dst)       { st_unary_map<base_ops::one, T>(dst);       }
template<ducks::sv::all T>
__device__ static inline void pos_infty(T &dst) { st_unary_map<base_ops::pos_infty, T>(dst); }
template<ducks::sv::all T>
__device__ static inline void neg_infty(T &dst) { st_unary_map<base_ops::neg_infty, T>(dst); }


}
