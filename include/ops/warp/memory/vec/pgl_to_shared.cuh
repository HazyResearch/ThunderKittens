#pragma once

#include "../util/reduce.cuh"

namespace kittens {

template<ReduceOp OP, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void ld_reduce_op(SV &dst, const PGL &src, int dev_id, const COORD &idx) {
    using U = typename PGL::dtype;
    using T = typename SV::dtype;
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = (dst.length + WARP_THREADS*elem_per_transfer - 1) / (WARP_THREADS*elem_per_transfer); // round up

    U *src_mc_ptr = src.mc_ptr_at(idx.template unit_coord<-1, 3>(), dev_id);
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    #pragma unroll
    for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=WARP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            float4 tmp;
            U *src_ptr = static_cast<U*>(src_mc_ptr) + i*elem_per_transfer;
            multimem_ld_reduce_op<T, OP>::apply_vec(&tmp, src_ptr);
            move<float4>::sts(dst_ptr + sizeof(typename SV::dtype)*i*elem_per_transfer, tmp);
        }
    }
}

template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void all_reduce_add(SV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<ReduceOp::ADD>(dst, src, dev_id, idx);
}

template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void all_reduce_min(SV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<ReduceOp::MIN>(dst, src, dev_id, idx);
}

template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void all_reduce_max(SV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<ReduceOp::MAX>(dst, src, dev_id, idx);
}

}