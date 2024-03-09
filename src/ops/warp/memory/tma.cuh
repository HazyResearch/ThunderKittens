#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

#include <cuda.h>

namespace kittens {
namespace tma {

namespace detail {

template<typename T> concept st_naive_row_layout_type = (
    st_type<T> && 
    (
        std::is_same_v<typename T::layout, st_naive_row_layout>
    )
);

template<typename T> concept st_wgmma_row_layout_type = (
    st_type<T> && 
    (
        std::is_same_v<typename T::layout, st_wgmma_row_0b_layout> || 
        std::is_same_v<typename T::layout, st_wgmma_row_32b_layout>
    )
);

template<typename T> concept st_wgmma_col_t_layout_type = (
    st_type<T> && 
    (
        std::is_same_v<typename T::layout, st_wgmma_col_t_0b_layout> ||
        std::is_same_v<typename T::layout, st_wgmma_col_t_32b_layout>
    )
);

}; 

// TMA STEP 1 = Create Tensor Map outside kernel (host side)
template<detail::st_naive_row_layout_type ST, int num_blocks>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map, bf16 *src) {
    
    constexpr uint32_t  tma_dim     = 2; 
    void                *global_addr = reinterpret_cast<void*>(src);

    constexpr CUtensorMapDataType     tma_format      = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; 
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      tma_swizzle     = CU_TENSOR_MAP_SWIZZLE_NONE; 

    constexpr uint64_t global_tile_height = num_blocks * ST::rows;
    constexpr uint64_t global_tile_width  = ST::cols; 
    constexpr uint64_t shared_tile_height = ST::rows; 
    constexpr uint64_t shared_tile_width  = ST::cols; 

    constexpr uint64_t gmem_shape[tma_dim] = {global_tile_width, global_tile_height};
    constexpr uint64_t gmem_stride[tma_dim] = {1, shared_tile_width * sizeof(bf16)}; 

    constexpr uint32_t smem_shape[tma_dim] = {shared_tile_width, shared_tile_height};
    constexpr uint32_t smem_stride[tma_dim] = {1, 1};

    // if smem_stride[] = 1 for all dims, smem_shape is used to specify num elements to load

    // if smem_stride[i] != 1 for any i, smem_shape[i]/smem_stride[i] is used to specify num 
    //      elements to load along dimension i


    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);
    // ensure that the global address is always 16-byte aligned 

    static_assert(gmem_stride[1] % 16 == 0, "all elements of gmem_stride must be a multiple of 16B");

    static_assert(smem_shape[0] <= 256, "smem_shape[0] elements must be less than= 256");
    static_assert(smem_shape[1] <= 256, "smem_shape[1] elements must be less than= 256");

    static_assert(smem_shape[0] * sizeof(bf16) % 16 == 0, "if interleave is none, then smem_shape[0] * sizeof(bf16) must be a multiple of 16B");

    static_assert(smem_stride[0] <= 8, "smem_stride[0] must be less than= 8");
    static_assert(smem_stride[1] <= 8, "smem_stride[1] must be less than= 8");

    static_assert(smem_stride[0] == 1, "smem_stride[0] is ignored when interleave is none");

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = (&gmem_stride[0] + 1); 
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
        std::cerr << "Error: " << error_string << std::endl;
    }
}; 

template<detail::st_wgmma_row_layout_type ST, int num_blocks>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map, bf16 *src) {
    
    constexpr uint32_t  tma_dim     = 5; 
    void                *global_addr = reinterpret_cast<void*>(src);

    // static assert that we currently only support 0B swizzle
    static_assert(std::is_same_v<typename ST::layout, st_wgmma_row_0b_layout>, "Only 0B swizzle is currently supported");

    constexpr CUtensorMapDataType     tma_format      = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; 
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;  
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      tma_swizzle     = (std::is_same_v<typename ST::layout, st_wgmma_row_0b_layout>) ? 
                                                        CU_TENSOR_MAP_SWIZZLE_NONE : CU_TENSOR_MAP_SWIZZLE_32B; 

    constexpr uint64_t global_tile_height = num_blocks * ST::rows;
    constexpr uint64_t global_tile_width  = ST::cols; 
    constexpr uint64_t shared_tile_height = ST::rows; 
    constexpr uint64_t shared_tile_width  = ST::cols; 

    constexpr uint64_t gmem_shape[tma_dim] = {
        8, 
        8,
        2, 
        (global_tile_height/8),
        (global_tile_width/16)
    };
    
    constexpr uint64_t gmem_stride[tma_dim] = {
        sizeof(bf16), 
        global_tile_width * sizeof(bf16),
        8 * sizeof(bf16), 
        8 * global_tile_width * sizeof(bf16),
        (global_tile_height/8) * 2 * sizeof(bf16)
    };

    constexpr uint32_t smem_shape[tma_dim]  = {8, 8, 2, (shared_tile_height/8), (shared_tile_width/16)};
    constexpr uint32_t smem_stride[tma_dim] = {1, 1, 1, 1, 1};

    // if smem_stride[] = 1 for all dims, smem_shape is used to specify num elements to load

    // if smem_stride[i] != 1 for any i, smem_shape[i]/smem_stride[i] is used to specify num 
    //      elements to load along dimension i

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    static_assert(gmem_stride[1] % 16 == 0, "gmem_stride[1] elements must be a multiple of 16B");
    static_assert(gmem_stride[2] % 16 == 0, "gmem_stride[2] elements must be a multiple of 16B");
    static_assert(gmem_stride[3] % 16 == 0, "gmem_stride[3] elements must be a multiple of 16B");
    static_assert(gmem_stride[4] % 16 == 0, "gmem_stride[4] elements must be a multiple of 16B");

    if constexpr (tma_swizzle == CU_TENSOR_MAP_SWIZZLE_32B) {
        static_assert(gmem_stride[1] % 32 == 0, "if swizzle is 32b, then gmem_stride[1] elements must be a multiple of 32B");
        static_assert(gmem_stride[2] % 32 == 0, "if swizzle is 32b, then gmem_stride[2] elements must be a multiple of 32B");
        static_assert(gmem_stride[3] % 32 == 0, "if swizzle is 32b, then gmem_stride[3] elements must be a multiple of 32B");
        static_assert(gmem_stride[4] % 32 == 0, "if swizzle is 32b, then gmem_stride[4] elements must be a multiple of 32B");
    }

    static_assert(smem_shape[0] <= 256, "smem_shape[0] elements must be less than= 256");
    static_assert(smem_shape[1] <= 256, "smem_shape[1] elements must be less than= 256");
    static_assert(smem_shape[2] <= 256, "smem_shape[2] elements must be less than= 256");
    static_assert(smem_shape[3] <= 256, "smem_shape[3] elements must be less than= 256");
    static_assert(smem_shape[4] <= 256, "smem_shape[4] elements must be less than= 256");

    static_assert(smem_shape[0] * sizeof(bf16) % 16 == 0, "if interleave is none, then smem_shape[0] * sizeof(bf16) must be a multiple of 16B");

    static_assert(smem_stride[0] <= 8, "smem_stride[0] must be less than= 8");
    static_assert(smem_stride[1] <= 8, "smem_stride[1] must be less than= 8");
    static_assert(smem_stride[2] <= 8, "smem_stride[2] must be less than= 8");
    static_assert(smem_stride[3] <= 8, "smem_stride[3] must be less than= 8");
    static_assert(smem_stride[4] <= 8, "smem_stride[4] must be less than= 8");

    static_assert(smem_stride[0] == 1, "smem_stride[0] is ignored when interleave is none");

    if constexpr (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        constexpr int swizzle_size = 32;
        assert(smem_shape[0] * sizeof(bf16) <= swizzle_size);
    }

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = (&gmem_stride[0] + 1); 
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
        std::cerr << "Error: " << error_string << std::endl;
    }
}; 

template<detail::st_wgmma_col_t_layout_type ST, int num_blocks>
__host__ static inline void create_tensor_map(CUtensorMap* tma_map, bf16 *src) {

    constexpr int tma_dim  = 5;
    void      *global_addr = reinterpret_cast<void*>(src);

    // static assert that we currently only support 0B swizzle
    static_assert(std::is_same_v<typename ST::layout, st_wgmma_col_t_0b_layout>, "Only 0B swizzle is currently supported");

    constexpr CUtensorMapDataType     tma_format      = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      tma_swizzle     = (std::is_same_v<typename ST::layout, st_wgmma_col_t_0b_layout>) ? 
                                                        CU_TENSOR_MAP_SWIZZLE_NONE : CU_TENSOR_MAP_SWIZZLE_32B;
                                                        
    constexpr uint64_t global_tile_height = num_blocks * ST::rows;
    constexpr uint64_t global_tile_width  = ST::cols;
    constexpr uint64_t shared_tile_height = ST::rows;
    constexpr uint64_t shared_tile_width  = ST::cols;

    constexpr uint64_t gmem_shape[tma_dim] = {
        8, 
        8,
        2,
        (global_tile_width/8), 
        (global_tile_height/16)
    };

    constexpr uint64_t gmem_stride[tma_dim] = {
        sizeof(bf16), 
        global_tile_width * sizeof(bf16), 
        8 * global_tile_width * sizeof(bf16),
        8 * sizeof(bf16),
        8 * 2 * global_tile_width * sizeof(bf16),
    };

    constexpr uint32_t smem_shape[tma_dim]  = {
        8, 
        8,
        2,
        (shared_tile_width/8),
        (shared_tile_height/16)
    };
    constexpr uint32_t smem_stride[tma_dim] = {1, 1, 1, 1, 1};

    // if smem_stride[] = 1 for all dims, smem_shape is used to specify num elements to load

    // if smem_stride[i] != 1 for any i, smem_shape[i]/smem_stride[i] is used to specify num 
    //      elements to load along dimension i

    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);
    // ensure that the global address is always 16-byte aligned 

    static_assert(gmem_stride[1] % 16 == 0, "gmem_stride[1] elements must be a multiple of 16B");
    static_assert(gmem_stride[2] % 16 == 0, "gmem_stride[2] elements must be a multiple of 16B");
    static_assert(gmem_stride[3] % 16 == 0, "gmem_stride[3] elements must be a multiple of 16B");
    static_assert(gmem_stride[4] % 16 == 0, "gmem_stride[4] elements must be a multiple of 16B");

    if constexpr (tma_swizzle == CU_TENSOR_MAP_SWIZZLE_32B) {
        static_assert(gmem_stride[1] % 32 == 0, "if swizzle is 32b, then gmem_stride[1] elements must be a multiple of 32B");
        static_assert(gmem_stride[2] % 32 == 0, "if swizzle is 32b, then gmem_stride[2] elements must be a multiple of 32B");
        static_assert(gmem_stride[3] % 32 == 0, "if swizzle is 32b, then gmem_stride[3] elements must be a multiple of 32B");
        static_assert(gmem_stride[4] % 32 == 0, "if swizzle is 32b, then gmem_stride[4] elements must be a multiple of 32B");
    }

    static_assert(smem_shape[0] <= 256, "smem_shape[0] elements must be less than= 256");
    static_assert(smem_shape[1] <= 256, "smem_shape[1] elements must be less than= 256");
    static_assert(smem_shape[2] <= 256, "smem_shape[2] elements must be less than= 256");
    static_assert(smem_shape[3] <= 256, "smem_shape[3] elements must be less than= 256");
    static_assert(smem_shape[4] <= 256, "smem_shape[4] elements must be less than= 256");

    static_assert(smem_shape[0] * sizeof(bf16) % 16 == 0, "if interleave is none, then smem_shape[0] * sizeof(bf16) must be a multiple of 16B");

    static_assert(smem_stride[0] <= 8, "smem_stride[0] must be less than= 8");
    static_assert(smem_stride[1] <= 8, "smem_stride[1] must be less than= 8");
    static_assert(smem_stride[2] <= 8, "smem_stride[2] must be less than= 8");
    static_assert(smem_stride[3] <= 8, "smem_stride[3] must be less than= 8");
    static_assert(smem_stride[4] <= 8, "smem_stride[4] must be less than= 8");

    static_assert(smem_stride[0] == 1, "smem_stride[0] is ignored when interleave is none");

    if constexpr (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        int swizzle_size = 32; 
        assert(smem_shape[0] * sizeof(bf16) <= swizzle_size);
    }

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = (&gmem_stride[0] + 1); 
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
        std::cerr << "Error: " << error_string << std::endl;
    }
}

// TMA STEP 2 = Prefetch Tensor Map using prefetch inside kernel (device side)
template<int height, int width>
__device__ static inline void prefetch(st<bf16, height, width, st_naive_row_layout> &dst, void const* const src_tma_map, int tile_idx) {
    if (threadIdx.x % WARP_SIZE == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);

        int32_t crd0 = 0;  
        int32_t crd1 = tile_idx * (dst.rows); 

        asm volatile (
            "cp.async.bulk.prefetch.tensor.2d.L2.global.tile"
            " [%0, {%1, %2}];"
            :
            : "l"(tma_ptr),
            "r"(crd0), "r"(crd1)
            : "memory"
        );
    }
}

template<int height, int width, st_wgmma_row_layout wgmma_row_layout>
__device__ static inline void prefetch(st<bf16, height, width, wgmma_row_layout> &dst, void const* const src_tma_map, int tile_idx) {
    if (threadIdx.x % WARP_SIZE == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);

        int32_t crd0 = 0;  
        int32_t crd1 = 0; 
        int32_t crd2 = 0;
        int32_t crd3 = tile_idx * (dst.rows);
        int32_t crd4 = 0;

        asm volatile (
            "cp.async.bulk.prefetch.tensor.5d.L2.global.tile"
            " [%0, {%1, %2, %3, %4, %5}];"
            :
            : "l"(tma_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
            : "memory"
        );
    }
}

template<int height, int width, st_wgmma_col_layout wgmma_col_layout>
__device__ static inline void prefetch(st<bf16, height, width, wgmma_col_layout> &dst, void const* const src_tma_map, int tile_idx) {
    if (threadIdx.x % WARP_SIZE == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);

        int32_t crd0 = 0;  
        int32_t crd1 = 0; 
        int32_t crd2 = 0;
        int32_t crd3 = 0; 
        int32_t crd4 = tile_idx * (dst.rows);

        asm volatile (
            "cp.async.bulk.prefetch.tensor.5d.L2.global.tile"
            " [%0, {%1, %2, %3, %4, %5}];"
            :
            : "l"(tma_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
            : "memory"
        );
    }
}

// TMA STEP 3 = Async load and store data from gmem/smem
template<int height, int width>
__device__ static inline void store_async(void *dst_tma_map, const st<bf16, height, width, st_naive_row_layout> &src, int tile_idx) {
    if (threadIdx.x % WARP_SIZE == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));

        int32_t crd0 = 0;  
        int32_t crd1 = tile_idx * (src.rows); 

        asm volatile (
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1)
            : "memory"
        );
    }
}
template<int height, int width>
__device__ static inline void load_async(st<bf16, height, width, st_naive_row_layout> &dst, void const* const src_tma_map, int tile_idx, uint64_t& barrier) {
    if (threadIdx.x % WARP_SIZE == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));

        int32_t crd0 = 0;  
        int32_t crd1 = tile_idx * (dst.rows); 

        asm volatile (
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4}], [%2];"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(crd0), "r"(crd1)
            : "memory"
        );
    }
}

template<int height, int width, st_wgmma_row_layout wgmma_row_layout>
__device__ static inline void store_async(void *dst_tma_map, const st<bf16, height, width, wgmma_row_layout> &src, int tile_idx) {
    if (threadIdx.x % WARP_SIZE == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));

        int32_t crd0 = 0;  
        int32_t crd1 = 0; 
        int32_t crd2 = 0;
        int32_t crd3 = tile_idx * (src.rows);
        int32_t crd4 = 0;

        asm volatile (
            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
            : "memory"
        );
    }
}
template<int height, int width, st_wgmma_row_layout wgmma_row_layout>
__device__ static inline void load_async(st<bf16, height, width, wgmma_row_layout> &dst, void const* const src_tma_map, int tile_idx, uint64_t& barrier) {
    if (threadIdx.x % WARP_SIZE == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));

        int32_t crd0 = 0;  
        int32_t crd1 = 0; 
        int32_t crd2 = 0;
        int32_t crd3 = tile_idx * (dst.rows);
        int32_t crd4 = 0;

        asm volatile (
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
            : "memory"
        );
    }
}

template<int height, int width, st_wgmma_col_layout wgmma_col_layout>
__device__ static inline void store_async(void *dst_tma_map, const st<bf16, height, width, wgmma_col_layout> &src, int tile_idx) {
    if (threadIdx.x % WARP_SIZE == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));

        int32_t crd0 = 0;  
        int32_t crd1 = 0; 
        int32_t crd2 = 0;
        int32_t crd3 = 0;
        int32_t crd4 = tile_idx * (src.rows);

        asm volatile (
            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
            : "memory"
        );
    }
}
template<int height, int width, st_wgmma_col_layout wgmma_col_layout>
__device__ static inline void load_async(st<bf16, height, width, wgmma_col_layout> &dst, void const* const src_tma_map, int tile_idx, uint64_t& barrier) {
    if (threadIdx.x % WARP_SIZE == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));

        int32_t crd0 = 0;  
        int32_t crd1 = 0; 
        int32_t crd2 = 0;
        int32_t crd3 = 0;
        int32_t crd4 = tile_idx * (dst.rows);

        asm volatile (
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
            : "memory"
        );
    }
}

/// Barrier functions for async load/store
__device__ static inline void init_barrier(uint64_t& barrier, int tc) {
    if (threadIdx.x % WARP_SIZE == 0) {
        void const* const ptr = &barrier;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile ("mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(bar_ptr), "r"(tc));
    }
}

__device__ static inline void set_barrier_bytes(uint64_t& barrier, uint32_t bytes) {
    if (threadIdx.x % WARP_SIZE == 0) {
        void const* const ptr = &barrier;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(bar_ptr), "r"(bytes));
    }
}

__device__ static inline void arrive_wait(uint64_t& barrier, int kPhaseBit) {
    void const* const ptr = &barrier;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

// TMA (store async) STEP 4 = Commit group
__device__ static inline void commit_group() {
    if (threadIdx.x % WARP_SIZE == 0) {
        asm volatile("cp.async.bulk.commit_group;");
    } 
}

// TMA (store async) STEP 5 = Wait for store complete
template <int N>
__device__ static inline void wait_for_store_complete() {
    asm volatile (
        "cp.async.bulk.wait_group %0;"
        :
        : "n"(N)
        : "memory"
    );
    __syncwarp();
}

}
}