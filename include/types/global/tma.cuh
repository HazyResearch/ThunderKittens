#pragma once

#include <cuda.h>
#include <iostream>
#include <assert.h>
#include <functional> // for std::hash
#include <unordered_map>
#include "../../common/common.cuh"
#include "../shared/shared.cuh"

namespace kittens {
namespace tma {
namespace detail {

/* ----------   Create tile tensor map descriptor (HOST)  ---------- */

/**
* @brief Creates a tensor map for the given source tensor.
*
* This function creates a tensor map (CUtensorMap) for the specified source shared tile type. The tensor map
* is used to describe the shape and layout of the tensor in memory. The function sets up the tensor
* map based on the provided source tensor pointer and the layout specified by the ST template parameter.
*
* @tparam ST The source tensor type, which must be TMA-compatible.
* @tparam blocks_height The number of tiles present on the height axis in global memory.
* @tparam blocks_width The number of tiles present on the width axis in global memory. Defaults to 1.
* @param tma_map Pointer to the CUtensorMap object to be initialized.
* @param src Pointer to the source tensor data in global memory.
*/
template<ducks::st::all ST, int axis>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map, const typename ST::dtype *src, int batch, int depth, int rows, int cols) {
    using dtype = typename ST::dtype;
    static_assert(axis==0 || axis==1 || axis==2, "axis must be 0, 1, or 2");
    
    constexpr uint32_t  tma_dim = 5; // Always use all 5D
    void *global_addr = (void*)(src);

    constexpr CUtensorMapDataType     tma_format      = (
        std::is_same_v<dtype, bf16>  ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
        std::is_same_v<dtype, half>  ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 :
        std::is_same_v<dtype, float> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 :
        CUtensorMapDataType(-1)
    );
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      tma_swizzle     = (
        ST::swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  :
        ST::swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
        ST::swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : 
        CU_TENSOR_MAP_SWIZZLE_NONE
    );

    uint64_t gmem_shape [5] = {0, 0, 0, 0, 0};
    uint64_t gmem_stride[4] = {0, 0, 0, 0};
    uint32_t smem_shape [5] = {0, 0, 0, 0, 0};
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    constexpr uint64_t shared_tile_height = ST::rows; 
    constexpr uint64_t shared_tile_width  = ST::cols;

    constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);

    if constexpr (axis == 2) {
        gmem_shape[0] = swizzle_elements;
        gmem_shape[1] = (uint64_t)rows;
        gmem_shape[2] = (uint64_t)(cols+swizzle_elements-1) / swizzle_elements; // round up, note this can potentially screw up out of bounds access handling :/
        gmem_shape[3] = (uint64_t)depth;
        gmem_shape[4] = (uint64_t)batch;

        gmem_stride[0] = (uint64_t)cols * sizeof(dtype);
        gmem_stride[1] = ST::swizzle_bytes;
        gmem_stride[2] = (uint64_t)rows * cols * sizeof(dtype);
        gmem_stride[3] = (uint64_t)depth * rows * cols * sizeof(dtype);
    }
    else if constexpr (axis == 1) {
        gmem_shape[0] = swizzle_elements;
        gmem_shape[1] = (uint64_t)depth;
        gmem_shape[2] = (uint64_t)(cols+swizzle_elements-1) / swizzle_elements; // round up, note this can potentially screw up out of bounds access handling :/
        gmem_shape[3] = (uint64_t)rows;
        gmem_shape[4] = (uint64_t)batch;

        gmem_stride[0] = (uint64_t)rows * cols * sizeof(dtype);
        gmem_stride[1] = ST::swizzle_bytes;
        gmem_stride[2] = (uint64_t)cols * sizeof(dtype);
        gmem_stride[3] = (uint64_t)depth * rows * cols * sizeof(dtype);

    }
    else {
        gmem_shape[0] = swizzle_elements;
        gmem_shape[1] = (uint64_t)batch;
        gmem_shape[2] = (uint64_t)(cols+swizzle_elements-1) / swizzle_elements; // round up, note this can potentially screw up out of bounds access handling :/
        gmem_shape[3] = (uint64_t)rows;
        gmem_shape[4] = (uint64_t)depth;

        gmem_stride[0] = (uint64_t)depth * rows * cols * sizeof(dtype);
        gmem_stride[1] = ST::swizzle_bytes;
        gmem_stride[2] = (uint64_t)cols * sizeof(dtype);
        gmem_stride[3] = (uint64_t)rows * cols * sizeof(dtype);
    }
    smem_shape[0] = swizzle_elements;
    smem_shape[1] = shared_tile_height;
    smem_shape[2] = shared_tile_width / swizzle_elements;
    smem_shape[3] = 1;
    smem_shape[4] = 1;

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    assert(gmem_stride[0] % 16 == 0); // gmem_stride[0] elements must be a multiple of 16B
    assert(gmem_stride[1] % 16 == 0); // gmem_stride[1] elements must be a multiple of 16B
    assert(gmem_stride[2] % 16 == 0); // gmem_stride[2] elements must be a multiple of 16B
    assert(gmem_stride[3] % 16 == 0); // gmem_stride[2] elements must be a multiple of 16B

    assert(smem_shape[0] <= 256); // smem_shape[0] elements must be <= 256
    assert(smem_shape[1] <= 256); // smem_shape[1] elements must be <= 256
    assert(smem_shape[2] <= 256); // smem_shape[2] elements must be <= 256

    assert((smem_shape[0]*sizeof(dtype)) % 16 == 0); // if wgmma_interleave is none, then smem_shape[0] * sizeof(dtype) must be a multiple of 16B

    assert(smem_stride[0] <= 8); // smem_stride[0] must be less <= 8
    assert(smem_stride[1] <= 8); // smem_stride[1] must be less <= 8
    assert(smem_stride[2] <= 8); // smem_stride[2] must be less <= 8
    assert(smem_stride[3] <= 8); // smem_stride[3] must be less <= 8
    assert(smem_stride[4] <= 8); // smem_stride[3] must be less <= 8

    assert(smem_stride[0] == 1); // smem_stride[0] is ignored when wgmma_interleave is none

    if constexpr (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        assert(smem_shape[0] * sizeof(dtype) <= ST::swizzle_bytes);
    }

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
        tma_swizzle,
        tma_l2Promotion,
        tma_oobFill);

    const char *error_string;
    CUresult res = cuGetErrorString(result, &error_string);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Error in tile TMA descriptor creation: " << error_string << std::endl;
    }
}

/**
* @brief Allocates on the GPU and initializes a tensor map for the given source tensor.
*
* This function creates a tensor map (CUtensorMap) for the specified source shared tile type. The tensor map
* is used to describe the shape and layout of the tensor in memory. The function sets up the tensor
* map based on the provided source tensor pointer and the layout specified by the ST template parameter.
*
* @tparam ST The source tensor type, which must be TMA-compatible.
* @tparam blocks_height The number of tiles present on the height axis in global memory.
* @tparam blocks_width The number of tiles present on the width axis in global memory. Defaults to 1.
* @param src Pointer to the source tensor data in global memory.
* @returns Pointer to the CUtensorMap object to be initialized.
*/
template<ducks::st::all ST>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(const typename ST::dtype *src, int batch, int depth, int rows, int cols) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host; // put it on the stack, why not.
    create_tensor_map<ST>(&tma_map_host, src, batch, depth, rows, cols);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

/* ----------   Create vector tensor map descriptor (HOST)  ---------- */

// First, we need a template system to determine how to divide up a long shared vector into multiple subvectors.
// We have to do this because the first dimension for TMA is limited to 256 elements.
// Our goal is to find the largest multiple of 16 that is <= 256 and divides the vector length evenly.

template<typename SV, int D=16> struct find_vector_divider {
    static constexpr int value = (SV::length % (16*D) == 0 && (SV::length < 256 || ((16*D)*sizeof(typename SV::dtype)) % 128 == 0)) ?
        16*D : find_vector_divider<SV, D-1>::value;
};
template<typename SV> struct find_vector_divider<SV, 1> { static constexpr int value = 16; }; // base case
template<typename SV> constexpr int sv_tma_dim1 = find_vector_divider<SV>::value; // inner dim
template<typename SV> constexpr int sv_tma_dim2 = (SV::length / sv_tma_dim1<SV>);

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
template<ducks::sv::all SV, int axis>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map, const typename SV::dtype *src, int batch, int depth, int rows, int cols) {
    using dtype = typename SV::dtype;
    static_assert(axis == -1, "for vector TMA, row axis must be -1 as it's unused");
    static_assert(SV::length <= 256 || (SV::length*sizeof(dtype)) % 128 == 0);
    // There is technically a way around ^ that involves instantiating two separate TMA descriptors, one of size 256
    // and the other of size %256, but this is a fairly mild restriction and the other approach is a real PITA and incurs other costs.
    
    constexpr uint32_t  tma_dim     = 4;
    void               *global_addr = (void*)(src);

    constexpr CUtensorMapDataType     tma_format      = (
        std::is_same_v<dtype, bf16>  ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
        std::is_same_v<dtype, half>  ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 :
        std::is_same_v<dtype, float> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 :
        CUtensorMapDataType(-1)
    );
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      swizzle         = CU_TENSOR_MAP_SWIZZLE_NONE;

    constexpr uint64_t dim1 = sv_tma_dim1<SV>; // inner dim
    std::cout << "dim1: " << dim1 << std::endl;
    // constexpr uint64_t dim2 = sv_tma_dim2<SV>; outer dim, not used here.

    uint64_t gmem_shape [4] = {(uint64_t)cols, (uint64_t)rows, (uint64_t)depth, (uint64_t)batch};
    uint64_t gmem_stride[3] = {(uint64_t)cols*sizeof(dtype), (uint64_t)cols*rows*sizeof(dtype), (uint64_t)cols*rows*depth*sizeof(dtype)};
    uint32_t smem_shape [4] = {(uint32_t)dim1, 1, 1, 1};
    uint32_t smem_stride[4] = {1, 1, 1, 1};

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    assert(smem_shape[0] <= 256); // smem_shape[0] elements must be <= 256.

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
        std::cerr << "Error in vector TMA descriptor creation: " << error_string << std::endl;
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
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(const typename SV::dtype *src, int batch, int depth, int rows, int cols) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host; // put it on the stack, why not.
    create_tensor_map<SV>(&tma_map_host, src, batch, depth, rows, cols);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

} // namespace detail
} // namespace tma
} // namespace kittens