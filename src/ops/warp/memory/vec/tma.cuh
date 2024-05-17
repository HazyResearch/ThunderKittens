#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

#include <cuda.h>
#include <iostream>

namespace kittens {
namespace tma {

/* ----------   Create tensor map descriptor (HOST)  ---------- */

/**
* @brief Creates a tensor map for the given source vector.
*
* This function creates a tensor map (CUtensorMap) for the specified source shared vector type. The tensor map
* is used to describe the shape and layout of the tensor in memory. The function sets up the tensor
* map based on the provided source tensor pointer and the layout specified by the SV template parameter.
*
* @tparam SV The source tensor type, which must be TMA-compatible.
* @tparam num_vectors The number of vectors present in global memory.
* @param tma_map Pointer to the CUtensorMap object to be initialized.
* @param src Pointer to the source tensor data in global memory.
*/
template<ducks::sv::all SV>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map, const bf16 *src, int num_vectors) {
    
    constexpr uint32_t  tma_dim      = 1; 
    void                *global_addr = (void*)(src);

    constexpr CUtensorMapDataType     tma_format      = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; 
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      swizzle         = CU_TENSOR_MAP_SWIZZLE_NONE;

    uint64_t gmem_shape [1] = {SV::length * num_vectors};
    uint64_t gmem_stride[1] = {1};
    uint32_t smem_shape [1] = {SV::length};
    uint32_t smem_stride[1] = {1};

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    assert(smem_shape[0] <= 256); // smem_shape[0] elements must be <= 256

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = &gmem_stride[0]; 
    const uint32_t *smem_shape_ptr = &smem_shape[0];
    const uint32_t *smem_stride_ptr = &smem_stride[0];

    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        tma_format,
        tma_dim,
        global_addr,
        gmem_shape_ptr,
        gmem_stride_ptr, 
        smem_shape_ptr,
        smem_stride_ptr,
        tma_interleave,
        swizzle,
        tma_l2Promotion,
        tma_oobFill
    );

    const char *error_string;
    CUresult res = cuGetErrorString(result, &error_string);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Error: " << error_string << std::endl;
    }
};

/**
* @brief Allocates on the GPU and initializes a tensor map for the given source tensor.
*
* This function creates a tensor map (CUtensorMap) for the specified source shared vector type. The tensor map
* is used to describe the shape and layout of the tensor in memory. The function sets up the tensor
* map based on the provided source tensor pointer and the layout specified by the SV template parameter.
*
* @tparam SV The source tensor type, which must be TMA-compatible.
* @tparam num_vectors The number of vectors present in global memory.
* @param src Pointer to the source tensor data in global memory.
* @returns Pointer to the CUtensorMap object to be initialized.
*/
template<ducks::sv::all SV>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(const bf16 *src, int num_vectors) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host; // put it on the stack, why not.
    create_tensor_map<SV>(&tma_map_host, src, num_vectors);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

/* ----------   Prefetch Tensor Map  ---------- */

/**
 * @brief Prefetches data from global memory into a shared memory vector, along with the tensormap.
 *
 * @tparam SV A shared vector type with a TMA-compatible layout
 * @param[out] dst The destination shared memory vector.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in] vec_idx The index of the requested vector.
 */
template<ducks::sv::all SV>
__device__ static inline void prefetch(SV &dst, void const* const src_tma_map, int vec_idx) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);

        int32_t crd0 = vec_idx * (dst.length);

        asm volatile (
            "cp.async.bulk.prefetch.tensor.1d.L2.global.tile"
            " [%0, {%1}];"
            :
            : "l"(tma_ptr), "r"(crd0)
            : "memory"
        );
    }
}

/* ----------   Async load and store data from gmem/smem  ---------- */

/**
 * @brief Asynchronously stores data into global memory from a shared memory vector.
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam SV A shared vector type with a TMA-compatible layout
 * @param[out] dst_tma_map The destination tensormap address in global memory
 * @param[in] src The source shared memory vector.
 * @param[in] vec_idx The index of the vector destination.
 */
template<ducks::sv::all SV>
__device__ static inline void store_async(void *dst_tma_map, const SV &src, int vec_idx) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));

        int32_t crd0 = vec_idx * (src.length);
        
        asm volatile (
            "cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"
            " [%0, {%2}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr), "r"(crd0)
            : "memory"
        );
    }
}

/**
* @brief Asynchronously performs an add reduction and stores the result into global memory.
*
* This function performs an asynchronous add reduction operation using CUDA's cp.reduce.async.bulk.tensor instruction.
*
* @tparam SV A shared vector type with a TMA-compatible layout
* @param[out] dst_tma_map The destination tensormap address in global memory
* @param[in] src The source shared memory vector.
* @param[in] vec_idx The index of the vector destination.
*/
template<ducks::sv::all SV>
__device__ static inline void store_add_async(void *dst_tma_map, const SV &src, int vec_idx) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));

        int32_t crd0 = vec_idx * (src.length);
        
        asm volatile (
            "cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group"
            " [%0, {%2}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr), "r"(crd0)
            : "memory"
        );
    }
}

/**
* @brief Asynchronously performs an min reduction and stores the result into global memory.
*
* This function performs an asynchronous min reduction operation using CUDA's cp.reduce.async.bulk.tensor instruction.
*
* @tparam SV A shared vector type with a TMA-compatible layout
* @param[out] dst_tma_map The destination tensormap address in global memory
* @param[in] src The source shared memory vector.
* @param[in] vec_idx The index of the vector destination.
*/
template<ducks::sv::all SV>
__device__ static inline void store_min_async(void *dst_tma_map, const SV &src, int vec_idx) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));

        int32_t crd0 = vec_idx * (src.length);
        
        asm volatile (
            "cp.reduce.async.bulk.tensor.1d.global.shared::cta.min.tile.bulk_group"
            " [%0, {%2}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr), "r"(crd0)
            : "memory"
        );
    }
}

/**
* @brief Asynchronously performs an max reduction and stores the result into global memory.
*
* This function performs an asynchronous max reduction operation using CUDA's cp.reduce.async.bulk.tensor instruction.
*
* @tparam SV A shared vector type with a TMA-compatible layout
* @param[out] dst_tma_map The destination tensormap address in global memory
* @param[in] src The source shared memory vector.
* @param[in] vec_idx The index of the vector destination.
*/
template<ducks::sv::all SV>
__device__ static inline void store_max_async(void *dst_tma_map, const SV &src, int vec_idx) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));

        int32_t crd0 = vec_idx * (src.length);
        
        asm volatile (
            "cp.reduce.async.bulk.tensor.1d.global.shared::cta.max.tile.bulk_group"
            " [%0, {%2}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr), "r"(crd0)
            : "memory"
        );
    }
}

/**
 * @brief Asynchronously loads data from global memory into a shared memory vector.
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam SV A shared vector type with a TMA-compatible layout
 * @param[out] dst The destination shared memory vector.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in] vec_idx The index of the requested vector.
 * @param[in,out] bar The barrier used for synchronization of the asynchronous copy.
 */
template<ducks::sv::all SV>
__device__ static inline void load_async(SV &dst, void const* const src_tma_map, barrier& bar, int vec_idx) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
        uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));

        int32_t crd0 = vec_idx * (dst.length);

        asm volatile (
            "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3}], [%2];"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(crd0)
            : "memory"
        );
    }
}

} // namespace tma
} // namespace kittens