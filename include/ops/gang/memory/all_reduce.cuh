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
 * @brief Performs an all-reduce addition operation across devices.
 *
 * @tparam PGL A parallel global layout type.
 * @param src A kittens::pgl object
 * @param dev_idx[in] The device index to operate on.
 */
template<ducks::pgl::all PGL>
__device__ static inline void all_reduce_add(PGL &pgl, int dev_idx) {
    using T = typename PGL::dtype;
    
    constexpr int iters_per_warp = 32;
    constexpr int nelem_per_iter = 4;
    constexpr int nelem_per_warp_per_iter = nelem_per_iter * WARP_THREADS;
    constexpr int nelem_per_warp = nelem_per_warp_per_iter * iters_per_warp;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = kittens::laneid();
    int total_active_threads = gridDim.x * blockDim.x;
    
    int nelem = pgl.size / sizeof(T);
    int elements_per_thread = (nelem + total_active_threads - 1) / total_active_threads;
    int thread_start_idx = tid * elements_per_thread;
    
    #pragma unroll
    for (int element_ofs = 0; element_ofs < nelem_per_warp; element_ofs += nelem_per_iter) {
        int idx = thread_start_idx + element_ofs;
        if (idx < nelem && idx % nelem_per_iter == 0) {
            float4 val;
            float4 *ptr = (float4 *)(pgl[dev_idx] + idx);

            multimem_ld_reduce<float4>::add(val, ptr);
            
            // TODO: need to ensure equivalence of this 
            // asm volatile(
            //     "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
            //     :: "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w)
            //     : "memory"
            // );
            move<float4>::stg(ptr, val);
        }
    }
}

/**
 * @brief Performs an all-reduce min operation across devices.
 *
 * @tparam PGL A parallel global layout type.
 * @param src A kittens::pgl object
 * @param dev_idx[in] The device index to operate on.
 */
template<ducks::pgl::all PGL>
__device__ static inline void all_reduce_min(PGL &pgl, int dev_idx) {
    using T = typename PGL::dtype;
    
    constexpr int iters_per_warp = 32;
    constexpr int nelem_per_iter = 4;
    constexpr int nelem_per_warp_per_iter = nelem_per_iter * WARP_THREADS;
    constexpr int nelem_per_warp = nelem_per_warp_per_iter * iters_per_warp;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = kittens::laneid();
    int total_active_threads = gridDim.x * blockDim.x;
    
    int nelem = pgl.size / sizeof(T);
    int elements_per_thread = (nelem + total_active_threads - 1) / total_active_threads;
    int thread_start_idx = tid * elements_per_thread;
    
    #pragma unroll
    for (int element_ofs = 0; element_ofs < nelem_per_warp; element_ofs += nelem_per_iter) {
        int idx = thread_start_idx + element_ofs;
        if (idx < nelem && idx % nelem_per_iter == 0) {
            float4 val;
            float4 *ptr = (float4 *)(pgl[dev_idx] + idx);

            multimem_ld_reduce<float4>::min(val, ptr);

            move<float4>::stg(ptr, val);
        }
    }
}

/**
 * @brief Performs an all-reduce max operation across devices.
 *
 * @tparam PGL A parallel global layout type.
 * @param src A kittens::pgl object
 * @param dev_idx[in] The device index to operate on.
 */
template<ducks::pgl::all PGL>
__device__ static inline void all_reduce_max(PGL &pgl, int dev_idx) {
    using T = typename PGL::dtype;
    
    constexpr int iters_per_warp = 32;
    constexpr int nelem_per_iter = 4;
    constexpr int nelem_per_warp_per_iter = nelem_per_iter * WARP_THREADS;
    constexpr int nelem_per_warp = nelem_per_warp_per_iter * iters_per_warp;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = kittens::laneid();
    int total_active_threads = gridDim.x * blockDim.x;
    
    int nelem = pgl.size / sizeof(T);
    int elements_per_thread = (nelem + total_active_threads - 1) / total_active_threads;
    int thread_start_idx = tid * elements_per_thread;
    
    #pragma unroll
    for (int element_ofs = 0; element_ofs < nelem_per_warp; element_ofs += nelem_per_iter) {
        int idx = thread_start_idx + element_ofs;
        if (idx < nelem && idx % nelem_per_iter == 0) {
            float4 val;
            float4 *ptr = (float4 *)(pgl[dev_idx] + idx);

            multimem_ld_reduce<float4>::max(val, ptr);

            move<float4>::stg(ptr, val);
        }
    }
}


/** 
 * @brief Performs an atomic add operation across devices.
 *
 * @tparam PGL A parallel global layout type.
 * @param src A kittens::pgl object
 * @param dev_idx[in] The device index to operate on.
 */
template<ducks::pgl::all PGL>
__device__ static inline void atomic_add(PGL &pgl, int dev_idx) {
    using T = typename PGL::dtype;
    
    constexpr int iters_per_warp = 32;
    constexpr int nelem_per_iter = 4;
    constexpr int nelem_per_warp_per_iter = nelem_per_iter * WARP_THREADS;
    constexpr int nelem_per_warp = nelem_per_warp_per_iter * iters_per_warp;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = kittens::laneid();
    int total_active_threads = gridDim.x * blockDim.x;
    
    int nelem = pgl.size / sizeof(T);
    int elements_per_thread = (nelem + total_active_threads - 1) / total_active_threads;
    int thread_start_idx = tid * elements_per_thread;
    
    #pragma unroll
    for (int element_ofs = 0; element_ofs < nelem_per_warp; element_ofs += nelem_per_iter) {
        int idx = thread_start_idx + element_ofs;
        if (idx < nelem && idx % nelem_per_iter == 0) {
            float4 val;
            float4 *ptr = (float4 *)(pgl[dev_idx] + idx);

            multimem_reduce<float4>::add(val, ptr);
        }
    }
}

/** 
 * @brief Performs an atomic min operation across devices.
 *
 * @tparam PGL A parallel global layout type.
 * @param src A kittens::pgl object
 * @param dev_idx[in] The device index to operate on.
 */
template<ducks::pgl::all PGL>
__device__ static inline void atomic_min(PGL &pgl, int dev_idx) {
    using T = typename PGL::dtype;
    
    constexpr int iters_per_warp = 32;
    constexpr int nelem_per_iter = 4;
    constexpr int nelem_per_warp_per_iter = nelem_per_iter * WARP_THREADS;
    constexpr int nelem_per_warp = nelem_per_warp_per_iter * iters_per_warp;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = kittens::laneid();
    int total_active_threads = gridDim.x * blockDim.x;
    
    int nelem = pgl.size / sizeof(T);
    int elements_per_thread = (nelem + total_active_threads - 1) / total_active_threads;
    int thread_start_idx = tid * elements_per_thread;
    
    #pragma unroll
    for (int element_ofs = 0; element_ofs < nelem_per_warp; element_ofs += nelem_per_iter) {
        int idx = thread_start_idx + element_ofs;
        if (idx < nelem && idx % nelem_per_iter == 0) {
            float4 val;
            float4 *ptr = (float4 *)(pgl[dev_idx] + idx);

            multimem_reduce<float4>::min(val, ptr);
        }
    }
}


/** 
 * @brief Performs an atomic max operation across devices.
 *
 * @tparam PGL A parallel global layout type.
 * @param src A kittens::pgl object
 * @param dev_idx[in] The device index to operate on.
 */
template<ducks::pgl::all PGL>
__device__ static inline void atomic_max(PGL &pgl, int dev_idx) {
    using T = typename PGL::dtype;
    
    constexpr int iters_per_warp = 32;
    constexpr int nelem_per_iter = 4;
    constexpr int nelem_per_warp_per_iter = nelem_per_iter * WARP_THREADS;
    constexpr int nelem_per_warp = nelem_per_warp_per_iter * iters_per_warp;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = kittens::laneid();
    int total_active_threads = gridDim.x * blockDim.x;
    
    int nelem = pgl.size / sizeof(T);
    int elements_per_thread = (nelem + total_active_threads - 1) / total_active_threads;
    int thread_start_idx = tid * elements_per_thread;
    
    #pragma unroll
    for (int element_ofs = 0; element_ofs < nelem_per_warp; element_ofs += nelem_per_iter) {
        int idx = thread_start_idx + element_ofs;
        if (idx < nelem && idx % nelem_per_iter == 0) {
            float4 val;
            float4 *ptr = (float4 *)(pgl[dev_idx] + idx);

            multimem_reduce<float4>::max(val, ptr);
        }
    }
}
} // namespace kittens