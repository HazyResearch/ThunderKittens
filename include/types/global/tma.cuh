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
template<ducks::st::all ST>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map, const typename ST::dtype *src, int batch, int depth, int rows, int cols) {
    using dtype = typename ST::dtype;
    
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

              uint64_t global_tile_height = (uint64_t)rows;
              uint64_t global_tile_width  = (uint64_t)cols; 
    constexpr uint64_t shared_tile_height = ST::rows; 
    constexpr uint64_t shared_tile_width  = ST::cols;

    constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);

    gmem_shape[0] = swizzle_elements;
    gmem_shape[1] = global_tile_height;
    gmem_shape[2] = (global_tile_width+swizzle_elements-1) / swizzle_elements; // round up, note this can potentially screw up out of bounds access handling :/
    gmem_shape[3] = (uint64_t)depth;
    gmem_shape[4] = (uint64_t)batch;

    gmem_stride[0] = global_tile_width * sizeof(dtype);
    gmem_stride[1] = ST::swizzle_bytes;
    gmem_stride[2] = global_tile_height * global_tile_width * sizeof(dtype);
    gmem_stride[3] = depth * global_tile_height * global_tile_width * sizeof(dtype);

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
        std::cerr << "Error creating tensor map: " << error_string << std::endl;
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
    static constexpr int value = (SV::length % (16*D) == 0) ? 16*D : find_vector_divider<SV, D-1>::value;
};
template<typename SV> struct find_vector_divider<SV, 1> { static constexpr int value = 16; }; // base case
template<typename SV> constexpr int sv_tma_dim1 = find_vector_divider<SV>::value;
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
template<ducks::sv::all SV>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map, const typename SV::dtype *src, int batch, int depth, int rows, int cols) {
    using dtype = typename SV::dtype;
    
    constexpr uint32_t  tma_dim      = 5;
    void                *global_addr = (void*)(src);

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

    constexpr uint64_t dim1 = sv_tma_dim1<SV>;
    constexpr uint64_t dim2 = sv_tma_dim2<SV>;

    int vec_wide = (cols + SV::length - 1) / SV::length; // round up, note this can potentially screw up out of bounds access handling :/
    uint64_t gmem_shape [5] = {(uint64_t)vec_wide*dim1, (uint64_t)vec_wide*dim2, (uint64_t)rows, (uint64_t)depth, (uint64_t)batch};
    uint64_t gmem_stride[4] = {dim1*sizeof(dtype), cols*sizeof(dtype), cols*rows*sizeof(dtype), cols*rows*depth*sizeof(dtype)};
    uint32_t smem_shape [5] = {dim1, dim2, 1, 1, 1};
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

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
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(const typename SV::dtype *src, int batch, int depth, int rows, int cols) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host; // put it on the stack, why not.
    create_tensor_map<SV>(&tma_map_host, src, batch, depth, rows, cols);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}


/* ----------  TMA descriptor cache (host-side)  ---------- */

// TMA cache
template<typename dtype> struct dtype_info      { static_assert(std::is_same_v<dtype, void>, "dtype is not a valid TK TMA type."); };
template<> struct dtype_info<float>             { static constexpr int id = 0; };
template<> struct dtype_info<half>              { static constexpr int id = 1; };
template<> struct dtype_info<bf16>              { static constexpr int id = 2; };
template<typename dtype> constexpr int dtype_id = dtype_info<dtype>::id;
template<typename S> struct shared_info         { static_assert(std::is_same_v<S, void>, "S is not a valid shared type."); };
template<kittens::ducks::st::all ST> struct shared_info<ST> {
    static constexpr int id = dtype_id<typename ST::dtype>;
    static constexpr int tile_rows = ST::rows, tile_cols = ST::cols;
};
template<kittens::ducks::sv::all SV> struct shared_info<SV> {
    static constexpr int id = dtype_id<typename SV::dtype>;
    static constexpr int tile_rows = 1, tile_cols = SV::length;
};
struct tma_cache_id {
    uint64_t base_ptr;
    int dtype_id, tile_rows, tile_cols, b, d, r, c;
    __host__ tma_cache_id() : base_ptr(0), dtype_id(0), tile_rows(0), tile_cols(0), b(0), d(0), r(0), c(0) {}
    __host__ tma_cache_id(void *_base_ptr, int _dtype_id, int _tile_rows, int _tile_cols, int _b, int _d, int _r, int _c):
        base_ptr((uint64_t)(_base_ptr)), dtype_id(_dtype_id), tile_rows(_tile_rows), tile_cols(_tile_cols), b(_b), d(_d), r(_r), c(_c) {}
    __host__ inline bool operator==(const tma_cache_id &other) const {
        return base_ptr == other.base_ptr &&
               dtype_id == other.dtype_id &&
               tile_rows == other.tile_rows &&
               tile_cols == other.tile_cols &&
               b == other.b &&
               d == other.d &&
               r == other.r &&
               c == other.c;
    }
};
} // namespace detail
} // namespace tma
} // namespace kittens
namespace std {
template<> struct hash<kittens::tma::detail::tma_cache_id> {
    std::size_t operator()(const kittens::tma::detail::tma_cache_id& id) const {
        // this is normally a terrible hash function, but the base_ptr makes it fine.
        return id.base_ptr ^ id.dtype_id ^ id.b ^ id.d ^ id.r ^ id.c ^ ((id.tile_rows/16) << 16) ^ (id.tile_cols/16);
    }
};
}
namespace kittens {
namespace tma {
struct tma_cache {
    std::unordered_map<detail::tma_cache_id, CUtensorMap*> cache;
    __host__ tma_cache() {}
    template<typename S> __host__ inline CUtensorMap* get_descriptor(void *base_ptr, int b, int d, int r, int c) {
        using info = detail::shared_info<S>;
        detail::tma_cache_id id(base_ptr, info::id, info::tile_rows, info::tile_cols, b, d, r, c);
        auto it = cache.find(id);
        if (it == cache.end()) {
            CUtensorMap *map = detail::allocate_and_create_tensor_map<S>((typename S::dtype*)base_ptr, b, d, r, c);
            cache.insert({id, map});
            return map;
        }
        return it->second;
    }
    __host__ inline void clear() {
        for (auto &pair : cache) {
            cudaFree(pair.second);
        }
        cache.clear();
    }
};
} // namespace tma
} // namespace kittens