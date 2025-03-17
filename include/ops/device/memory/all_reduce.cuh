/**
 * @file
 * @brief Functions for reduction operations called by a single device
 */
#pragma once

#include <cuda.h>
#include "util/ld_reduce.cuh"
#include "util/reduce.cuh"


namespace kittens {

enum class ReduceOperation {
    ADD,
    MIN,
    MAX
};

/**
 * @brief Implementation of reduction operations across devices.
 * 
 * @tparam ReduceOp Either a multimem_ld_reduce or multimem_reduce operation.
 * @tparam Op The operation to perform.
 * @tparam StoreResult Whether to store the result back to the global memory.
 * @tparam GL The global layout of the data.
 * @param p_o A parallel global layout object.
 * @param dev_idx The device index.
 */
template<ReduceOperation Op, typename GL>
__device__ static inline void ld_reduce_op(PglObj<GL> p_o) {
    using T = typename PglObj<GL>::dtype;

    int thread_idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int total_threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    
    int global_thread_idx = block_idx * total_threads_per_block + thread_idx;
    int warp_id = global_thread_idx / WARP_THREADS;
    int lane_id = global_thread_idx % WARP_THREADS;

    constexpr int iters_per_warp = 32;
    constexpr int N_per_iter = sizeof(float4) / sizeof(T);
    constexpr int N_per_warp_per_iter = N_per_iter * WARP_THREADS;
    constexpr int N_per_warp = iters_per_warp * N_per_warp_per_iter;
    int start_idx = N_per_warp * warp_id;

    for (int i = 0; i < 32; ++i) {
        int idx = start_idx + i * N_per_warp_per_iter + lane_id * N_per_iter;
        if (idx < p_o.nelem) {
            float4 val;
            T* ptr = static_cast<T*>(p_o.mc_ptr + p_o.nelem * p_o.dev_id) + idx;

            if constexpr (Op == ReduceOperation::ADD) {
                multimem_ld_reduce<T>::add(val, ptr);
            } else if constexpr (Op == ReduceOperation::MIN) {
                multimem_ld_reduce<T>::min(val, ptr);
            } else if constexpr (Op == ReduceOperation::MAX) {
                multimem_ld_reduce<T>::max(val, ptr);
            }
                
            move<float4>::stg((float4*)ptr, val);
        }
        __syncthreads();
    }
}

template <typename PGL_OBJ>
__device__ static inline void all_reduce_add(PGL_OBJ p_o) {
    ld_reduce_op<ReduceOperation::ADD>(p_o);
}

template <typename PGL_OBJ>
__device__ static inline void all_reduce_min(PGL_OBJ p_o) {
    ld_reduce_op<ReduceOperation::MIN>(p_o);
}

template <typename PGL_OBJ>
__device__ static inline void all_reduce_max(PGL_OBJ p_o) {
    ld_reduce_op<ReduceOperation::MAX>(p_o);
}

template<ReduceOperation Op, typename GL>
__device__ static inline void reduce_op(PglObj<GL> p_o) {
    using T = typename PglObj<GL>::dtype;

    printf("In reduce op for device %d\n", p_o.dev_id);

    int thread_idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int total_threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    
    int global_thread_idx = block_idx * total_threads_per_block + thread_idx;
    int warp_id = global_thread_idx / WARP_THREADS;
    int lane_id = global_thread_idx % WARP_THREADS;

    constexpr int iters_per_warp = 32;
    constexpr int N_per_iter = 1;
    constexpr int N_per_warp_per_iter = N_per_iter * WARP_THREADS;
    constexpr int N_per_warp = iters_per_warp * N_per_warp_per_iter;
    int start_idx = N_per_warp * warp_id;

    for (int i = 0; i < 32; ++i) {
        int idx = start_idx + i * N_per_warp_per_iter + lane_id * N_per_iter;
        if (idx < p_o.nelem) {
            T* ptr = static_cast<T*>(p_o.mc_ptr) + idx;

            if constexpr (Op == ReduceOperation::ADD) {
                multimem_reduce<T>::add(ptr, *ptr);
            } else if constexpr (Op == ReduceOperation::MIN) {
                multimem_reduce<T>::min(ptr, *ptr);
            } else if constexpr (Op == ReduceOperation::MAX) {
                multimem_reduce<T>::max(ptr, *ptr);
            }                
        }
        __syncthreads();
    }
}

template <typename PGL_OBJ>
__device__ static inline void atomic_add(PGL_OBJ p_o) {
    reduce_op<ReduceOperation::ADD>(p_o);
}

template <typename PGL_OBJ>
__device__ static inline void atomic_min(PGL_OBJ p_o) {
    reduce_op<ReduceOperation::MIN>(p_o);
}

template <typename PGL_OBJ>
__device__ static inline void atomic_max(PGL_OBJ p_o) {
    reduce_op<ReduceOperation::MAX>(p_o);
}

} // namespace kittens