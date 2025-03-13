/**
 * @file
 * @brief Functions for reduction operations across a gang
 */
#pragma once

#include <cuda.h>
#include "util/ld_reduce.cuh"
#include "util/reduce.cuh"
 
namespace kittens {

/**
 * @brief Implementation of reduction operations across devices.
 * 
 * @tparam PGL A parallel global layout type.
 * @tparam ReduceOp Type of reduction operation.
 * @tparam StoreResult Whether to store the result back.
 */
template<ducks::pgl::all PGL, template<typename> class ReduceOp, bool StoreResult = true>
__device__ static inline void reduce_op(PGL &pgl, int dev_idx) {
    using T = typename PGL::dtype;
    
    constexpr int iters_per_warp = 32;
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(T);
    constexpr int nelem_per_iter = 4 * elem_per_transfer;
    constexpr int nelem_per_warp_per_iter = nelem_per_iter * WARP_THREADS;
    constexpr int nelem_per_warp = nelem_per_warp_per_iter * iters_per_warp;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_active_threads = gridDim.x * blockDim.x;
    
    int nelem = pgl.size / sizeof(T);
    int elements_per_thread = (nelem + total_active_threads - 1) / total_active_threads;
    int thread_start_idx = tid * elements_per_thread;
    
    #pragma unroll
    for (int element_ofs = 0; element_ofs < nelem_per_warp; element_ofs += nelem_per_iter) {
        int idx = thread_start_idx + element_ofs;
        if (idx < nelem && idx % elem_per_transfer == 0) {
            float4 val;
            float4 *ptr = reinterpret_cast<float4 *>(pgl[dev_idx] + idx);
            
            ReduceOp<float4>::op(val, ptr);
            
            if constexpr (StoreResult) {
                move<float4>::stg(ptr, val);
            }
        }
    }
}

template<ducks::pgl::all PGL>
__device__ static inline void all_reduce_add(PGL &pgl, int dev_idx) {
    reduce_op<PGL, multimem_ld_reduce<float4>::add, true>(pgl, dev_idx);
}

template<ducks::pgl::all PGL>
__device__ static inline void all_reduce_min(PGL &pgl, int dev_idx) {
    reduce_op<PGL, multimem_ld_reduce<float4>::min, true>(pgl, dev_idx);
}

template<ducks::pgl::all PGL>
__device__ static inline void all_reduce_max(PGL &pgl, int dev_idx) {
    reduce_op<PGL, multimem_ld_reduce<float4>::max, true>(pgl, dev_idx);
}

template<ducks::pgl::all PGL>
__device__ static inline void atomic_add(PGL &pgl, int dev_idx) {
    reduce_op<PGL, multimem_reduce<float4>::add, false>(pgl, dev_idx);
}

template<ducks::pgl::all PGL>
__device__ static inline void atomic_min(PGL &pgl, int dev_idx) {
    reduce_op<PGL, multimem_reduce<float4>::min, false>(pgl, dev_idx);
}

template<ducks::pgl::all PGL>
__device__ static inline void atomic_max(PGL &pgl, int dev_idx) {
    reduce_op<PGL, multimem_reduce<float4>::max, false>(pgl, dev_idx);
}

} // namespace kittens