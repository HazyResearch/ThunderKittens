#pragma once

#include "../../../common/common.cuh"
#include "../../../types/shared/shared.cuh"

#include <cuda.h>

namespace kittens {
namespace tma {

template<st_layout layout>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map, bf16 *src, int blocks, 
                                              int global_tile_height, int global_tile_width, 
                                              int shared_tile_height, int shared_tile_width) {}; 

template<>
__host__ inline void create_tensor_map<st_naive_row_layout>(CUtensorMap *tma_map, bf16 *src, int blocks, 
                                              int global_tile_height, int global_tile_width, 
                                              int shared_tile_height, int shared_tile_width) {
    uint32_t tma_dim     = 2; 
    void    *global_addr = reinterpret_cast<void*>(src);

    CUtensorMapDataType     tma_format      = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; 
    CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    CUtensorMapSwizzle      tma_swizzle     = CU_TENSOR_MAP_SWIZZLE_NONE;

    uint64_t gmem_shape[tma_dim] = {global_tile_width, global_tile_height};
    uint64_t gmem_stride[tma_dim] = {1, shared_tile_width * sizeof(bf16)}; 

    uint32_t smem_shape[tma_dim] = {shared_tile_width, shared_tile_height};
    uint32_t smem_stride[tma_dim] = {1, 1}; 

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    // if interleave is 32b, global address must be 32-byte aligned
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert((reinterpret_cast<uint64_t>(global_addr) & 0b11111) == 0);
    }

    // if interleave is none, then dim >= 3
    if (tma_interleave != CU_TENSOR_MAP_INTERLEAVE_NONE) {
        assert(tma_dim >= 3);
    }

    // all elements of gmem_shape must be non-zero
    for (int i = 0; i < tma_dim; i++) {
        assert(gmem_shape[i] != 0);
    }

    for (int i = 0; i < tma_dim; i++) {
        // ignore first value when interleave is none
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            continue; 
        }

        // all elements of gmem_stride must be a multiple of 16
        assert(gmem_stride[i] % 16 == 0);

        // multile of 32 for 32B interleave
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
            assert(gmem_stride[i] % 32 == 0);
        }
    }

    // all smem_shape elements must be non-zero and less than= 256
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_shape[i] != 0);
        assert(smem_shape[i] <= 256);
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            assert(smem_shape[0] * sizeof(bf16) % 16 == 0);
        }
    }

    // all smem_stride elements must be non-zero and less than= 8
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_stride[i] != 0);
        assert(smem_stride[i] <= 8);
    }

    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        int swizzle_size = 32; 
        switch (tma_swizzle) {
            case CU_TENSOR_MAP_SWIZZLE_32B:
                swizzle_size = 32;
                break;
            case CU_TENSOR_MAP_SWIZZLE_64B:
                swizzle_size = 64;
                break;
            case CU_TENSOR_MAP_SWIZZLE_128B:
                swizzle_size = 128;
                break;
            default:
                assert(false);
        }
        assert(smem_shape[0] * sizeof(bf16) <= swizzle_size);
    }

    // if interleave is 32B, then swizzle must be 32B
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert(tma_swizzle == CU_TENSOR_MAP_SWIZZLE_32B);
    }

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE) ? (&gmem_stride[0] + 1) : (&gmem_stride[0]);
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
    
    assert(result == CUDA_SUCCESS);
}
template<>
__host__ inline void create_tensor_map<st_xor_row_layout>(CUtensorMap *tma_map, bf16 *src, int blocks, 
                                              int global_tile_height, int global_tile_width, 
                                              int shared_tile_height, int shared_tile_width) {
    uint32_t tma_dim     = 2; 
    void    *global_addr = reinterpret_cast<void*>(src);

    CUtensorMapDataType     tma_format      = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; 
    CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    CUtensorMapSwizzle      tma_swizzle     = CU_TENSOR_MAP_SWIZZLE_64B; 

    uint64_t gmem_shape[tma_dim] = {global_tile_width, global_tile_height};
    uint64_t gmem_stride[tma_dim] = {1, shared_tile_width * sizeof(bf16)}; 

    uint32_t smem_shape[tma_dim] = {shared_tile_width, shared_tile_height};
    uint32_t smem_stride[tma_dim] = {1, 1}; 

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    // if interleave is 32b, global address must be 32-byte aligned
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert((reinterpret_cast<uint64_t>(global_addr) & 0b11111) == 0);
    }

    // if interleave is none, then dim >= 3
    if (tma_interleave != CU_TENSOR_MAP_INTERLEAVE_NONE) {
        assert(tma_dim >= 3);
    }

    // all elements of gmem_shape must be non-zero
    for (int i = 0; i < tma_dim; i++) {
        assert(gmem_shape[i] != 0);
    }

    for (int i = 0; i < tma_dim; i++) {
        // ignore first value when interleave is none
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            continue; 
        }

        // all elements of gmem_stride must be a multiple of 16
        assert(gmem_stride[i] % 16 == 0);

        // multile of 32 for 32B interleave
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
            assert(gmem_stride[i] % 32 == 0);
        }
    }

    // all smem_shape elements must be non-zero and less than= 256
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_shape[i] != 0);
        assert(smem_shape[i] <= 256);
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            assert(smem_shape[0] * sizeof(bf16) % 16 == 0);
        }
    }

    // all smem_stride elements must be non-zero and less than= 8
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_stride[i] != 0);
        assert(smem_stride[i] <= 8);
    }

    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        int swizzle_size = 32; 
        switch (tma_swizzle) {
            case CU_TENSOR_MAP_SWIZZLE_32B:
                swizzle_size = 32;
                break;
            case CU_TENSOR_MAP_SWIZZLE_64B:
                swizzle_size = 64;
                break;
            case CU_TENSOR_MAP_SWIZZLE_128B:
                swizzle_size = 128;
                break;
            default:
                assert(false);
        }
        assert(smem_shape[0] * sizeof(bf16) <= swizzle_size);
    }

    // if interleave is 32B, then swizzle must be 32B
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert(tma_swizzle == CU_TENSOR_MAP_SWIZZLE_32B);
    }

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE) ? (&gmem_stride[0] + 1) : (&gmem_stride[0]);
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
    
    assert(result == CUDA_SUCCESS);
}

template<>
__host__ inline void create_tensor_map<st_wgmma_row_0b_layout>(CUtensorMap *tma_map, bf16 *src, int blocks, 
                                                          int global_tile_height, int global_tile_width, 
                                                          int shared_tile_height, int shared_tile_width) {
    uint32_t tma_dim     = 5; 
    void    *global_addr = reinterpret_cast<void*>(src);

    CUtensorMapDataType     tma_format      = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; 
    CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    CUtensorMapSwizzle      tma_swizzle     = CU_TENSOR_MAP_SWIZZLE_NONE; 

    uint64_t gmem_shape[tma_dim] = {
        8, 
        8,
        2, 
        (global_tile_height/8),
        (global_tile_width/16)
    };
    uint64_t gmem_stride[tma_dim] = {
        sizeof(bf16), 
        global_tile_width * sizeof(bf16),
        8 * sizeof(bf16), 
        8 * global_tile_width * sizeof(bf16),
        8 * 2 * sizeof(bf16)
    };


    uint32_t smem_shape[tma_dim]  = {8, 8, 2, (shared_tile_height/8), (shared_tile_width/16)};
    uint32_t smem_stride[tma_dim] = {1, 1, 1, 1, 1};

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    // if interleave is 32b, global address must be 32-byte aligned
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert((reinterpret_cast<uint64_t>(global_addr) & 0b11111) == 0);
    }

    // if interleave is none, then dim >= 3
    if (tma_interleave != CU_TENSOR_MAP_INTERLEAVE_NONE) {
        assert(tma_dim >= 3);
    }

    // all elements of gmem_shape must be non-zero
    for (int i = 0; i < tma_dim; i++) {
        assert(gmem_shape[i] != 0);
    }

    for (int i = 0; i < tma_dim; i++) {
        // ignore first value when interleave is none
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            continue; 
        }

        // all elements of gmem_stride must be a multiple of 16
        assert(gmem_stride[i] % 16 == 0);

        // multile of 32 for 32B interleave
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
            assert(gmem_stride[i] % 32 == 0);
        }
    }

    // all smem_shape elements must be non-zero and less than= 256
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_shape[i] != 0);
        assert(smem_shape[i] <= 256);
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            assert(smem_shape[0] * sizeof(bf16) % 16 == 0);
        }
    }

    // all smem_stride elements must be non-zero and less than= 8
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_stride[i] != 0);
        assert(smem_stride[i] <= 8);
    }

    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        int swizzle_size = 32; 
        switch (tma_swizzle) {
            case CU_TENSOR_MAP_SWIZZLE_32B:
                swizzle_size = 32;
                break;
            case CU_TENSOR_MAP_SWIZZLE_64B:
                swizzle_size = 64;
                break;
            case CU_TENSOR_MAP_SWIZZLE_128B:
                swizzle_size = 128;
                break;
            default:
                assert(false);
        }
        assert(smem_shape[0] * sizeof(bf16) <= swizzle_size);
    }

    // if interleave is 32B, then swizzle must be 32B
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert(tma_swizzle == CU_TENSOR_MAP_SWIZZLE_32B);
    }

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE) ? (&gmem_stride[0] + 1) : (&gmem_stride[0]);
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
    
    assert(result == CUDA_SUCCESS);
}
template<>
__host__ inline void create_tensor_map<st_wgmma_row_32b_layout>(CUtensorMap *tma_map, bf16 *src, int blocks, 
                                                          int global_tile_height, int global_tile_width, 
                                                          int shared_tile_height, int shared_tile_width) {
    uint32_t tma_dim     = 5; 
    void    *global_addr = reinterpret_cast<void*>(src);

    CUtensorMapDataType     tma_format      = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; 
    CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    CUtensorMapSwizzle      tma_swizzle     = CU_TENSOR_MAP_SWIZZLE_32B; 

    uint64_t gmem_shape[tma_dim] = {
        8, 
        8,
        2, 
        (global_tile_height/8),
        (global_tile_width/16)
    };
    uint64_t gmem_stride[tma_dim] = {
        sizeof(bf16), 
        global_tile_width * sizeof(bf16),
        8 * sizeof(bf16), 
        8 * global_tile_width * sizeof(bf16),
        8 * 2 * sizeof(bf16)
    };


    uint32_t smem_shape[tma_dim]  = {8, 8, 2, (shared_tile_height/8), (shared_tile_width/16)};
    uint32_t smem_stride[tma_dim] = {1, 1, 1, 1, 1};

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    // if interleave is 32b, global address must be 32-byte aligned
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert((reinterpret_cast<uint64_t>(global_addr) & 0b11111) == 0);
    }

    // if interleave is none, then dim >= 3
    if (tma_interleave != CU_TENSOR_MAP_INTERLEAVE_NONE) {
        assert(tma_dim >= 3);
    }

    // all elements of gmem_shape must be non-zero
    for (int i = 0; i < tma_dim; i++) {
        assert(gmem_shape[i] != 0);
    }

    for (int i = 0; i < tma_dim; i++) {
        // ignore first value when interleave is none
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            continue; 
        }

        // all elements of gmem_stride must be a multiple of 16
        assert(gmem_stride[i] % 16 == 0);

        // multile of 32 for 32B interleave
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
            assert(gmem_stride[i] % 32 == 0);
        }
    }

    // all smem_shape elements must be non-zero and less than= 256
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_shape[i] != 0);
        assert(smem_shape[i] <= 256);
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            assert(smem_shape[0] * sizeof(bf16) % 16 == 0);
        }
    }

    // all smem_stride elements must be non-zero and less than= 8
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_stride[i] != 0);
        assert(smem_stride[i] <= 8);
    }

    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        int swizzle_size = 32; 
        switch (tma_swizzle) {
            case CU_TENSOR_MAP_SWIZZLE_32B:
                swizzle_size = 32;
                break;
            case CU_TENSOR_MAP_SWIZZLE_64B:
                swizzle_size = 64;
                break;
            case CU_TENSOR_MAP_SWIZZLE_128B:
                swizzle_size = 128;
                break;
            default:
                assert(false);
        }
        assert(smem_shape[0] * sizeof(bf16) <= swizzle_size);
    }

    // if interleave is 32B, then swizzle must be 32B
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert(tma_swizzle == CU_TENSOR_MAP_SWIZZLE_32B);
    }

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE) ? (&gmem_stride[0] + 1) : (&gmem_stride[0]);
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
    
    assert(result == CUDA_SUCCESS);
}
template<>
__host__ inline void create_tensor_map<st_wgmma_row_64b_layout>(CUtensorMap *tma_map, bf16 *src, int blocks, 
                                                          int global_tile_height, int global_tile_width, 
                                                          int shared_tile_height, int shared_tile_width) {
    uint32_t tma_dim     = 5; 
    void    *global_addr = reinterpret_cast<void*>(src);

    CUtensorMapDataType     tma_format      = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; 
    CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    CUtensorMapSwizzle      tma_swizzle     = CU_TENSOR_MAP_SWIZZLE_64B; 

    uint64_t gmem_shape[tma_dim] = {
        8, 
        8,
        2, 
        (global_tile_height/8),
        (global_tile_width/16)
    };
    uint64_t gmem_stride[tma_dim] = {
        sizeof(bf16), 
        global_tile_width * sizeof(bf16),
        8 * sizeof(bf16), 
        8 * global_tile_width * sizeof(bf16),
        8 * 2 * sizeof(bf16)
    };


    uint32_t smem_shape[tma_dim]  = {8, 8, 2, (shared_tile_height/8), (shared_tile_width/16)};
    uint32_t smem_stride[tma_dim] = {1, 1, 1, 1, 1};

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    // if interleave is 32b, global address must be 32-byte aligned
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert((reinterpret_cast<uint64_t>(global_addr) & 0b11111) == 0);
    }

    // if interleave is none, then dim >= 3
    if (tma_interleave != CU_TENSOR_MAP_INTERLEAVE_NONE) {
        assert(tma_dim >= 3);
    }

    // all elements of gmem_shape must be non-zero
    for (int i = 0; i < tma_dim; i++) {
        assert(gmem_shape[i] != 0);
    }

    for (int i = 0; i < tma_dim; i++) {
        // ignore first value when interleave is none
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            continue; 
        }

        // all elements of gmem_stride must be a multiple of 16
        assert(gmem_stride[i] % 16 == 0);

        // multile of 32 for 32B interleave
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
            assert(gmem_stride[i] % 32 == 0);
        }
    }

    // all smem_shape elements must be non-zero and less than= 256
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_shape[i] != 0);
        assert(smem_shape[i] <= 256);
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            assert(smem_shape[0] * sizeof(bf16) % 16 == 0);
        }
    }

    // all smem_stride elements must be non-zero and less than= 8
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_stride[i] != 0);
        assert(smem_stride[i] <= 8);
    }

    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        int swizzle_size = 32; 
        switch (tma_swizzle) {
            case CU_TENSOR_MAP_SWIZZLE_32B:
                swizzle_size = 32;
                break;
            case CU_TENSOR_MAP_SWIZZLE_64B:
                swizzle_size = 64;
                break;
            case CU_TENSOR_MAP_SWIZZLE_128B:
                swizzle_size = 128;
                break;
            default:
                assert(false);
        }
        assert(smem_shape[0] * sizeof(bf16) <= swizzle_size);
    }

    // if interleave is 32B, then swizzle must be 32B
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert(tma_swizzle == CU_TENSOR_MAP_SWIZZLE_32B);
    }

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE) ? (&gmem_stride[0] + 1) : (&gmem_stride[0]);
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
    
    assert(result == CUDA_SUCCESS);
}
template<>
__host__ inline void create_tensor_map<st_wgmma_row_128b_layout>(CUtensorMap *tma_map, bf16 *src, int blocks, 
                                                          int global_tile_height, int global_tile_width, 
                                                          int shared_tile_height, int shared_tile_width) {
    uint32_t tma_dim     = 5; 
    void    *global_addr = reinterpret_cast<void*>(src);

    CUtensorMapDataType     tma_format      = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; 
    CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    CUtensorMapSwizzle      tma_swizzle     = CU_TENSOR_MAP_SWIZZLE_128B; 

    uint64_t gmem_shape[tma_dim] = {
        8, 
        8,
        2, 
        (global_tile_height/8),
        (global_tile_width/16)
    };
    uint64_t gmem_stride[tma_dim] = {
        sizeof(bf16), 
        global_tile_width * sizeof(bf16),
        8 * sizeof(bf16), 
        8 * global_tile_width * sizeof(bf16),
        8 * 2 * sizeof(bf16)
    };


    uint32_t smem_shape[tma_dim]  = {8, 8, 2, (shared_tile_height/8), (shared_tile_width/16)};
    uint32_t smem_stride[tma_dim] = {1, 1, 1, 1, 1};

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    // if interleave is 32b, global address must be 32-byte aligned
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert((reinterpret_cast<uint64_t>(global_addr) & 0b11111) == 0);
    }

    // if interleave is none, then dim >= 3
    if (tma_interleave != CU_TENSOR_MAP_INTERLEAVE_NONE) {
        assert(tma_dim >= 3);
    }

    // all elements of gmem_shape must be non-zero
    for (int i = 0; i < tma_dim; i++) {
        assert(gmem_shape[i] != 0);
    }

    for (int i = 0; i < tma_dim; i++) {
        // ignore first value when interleave is none
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            continue; 
        }

        // all elements of gmem_stride must be a multiple of 16
        assert(gmem_stride[i] % 16 == 0);

        // multile of 32 for 32B interleave
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
            assert(gmem_stride[i] % 32 == 0);
        }
    }

    // all smem_shape elements must be non-zero and less than= 256
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_shape[i] != 0);
        assert(smem_shape[i] <= 256);
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            assert(smem_shape[0] * sizeof(bf16) % 16 == 0);
        }
    }

    // all smem_stride elements must be non-zero and less than= 8
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_stride[i] != 0);
        assert(smem_stride[i] <= 8);
    }

    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        int swizzle_size = 32; 
        switch (tma_swizzle) {
            case CU_TENSOR_MAP_SWIZZLE_32B:
                swizzle_size = 32;
                break;
            case CU_TENSOR_MAP_SWIZZLE_64B:
                swizzle_size = 64;
                break;
            case CU_TENSOR_MAP_SWIZZLE_128B:
                swizzle_size = 128;
                break;
            default:
                assert(false);
        }
        assert(smem_shape[0] * sizeof(bf16) <= swizzle_size);
    }

    // if interleave is 32B, then swizzle must be 32B
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert(tma_swizzle == CU_TENSOR_MAP_SWIZZLE_32B);
    }

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE) ? (&gmem_stride[0] + 1) : (&gmem_stride[0]);
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
    
    assert(result == CUDA_SUCCESS);
}

template<>
__host__ inline void create_tensor_map<st_wgmma_col_0b_layout>(CUtensorMap *tma_map, bf16 *src, int blocks, 
                                                          int global_tile_height, int global_tile_width, 
                                                          int shared_tile_height, int shared_tile_width) {
    uint32_t tma_dim     = 5; 
    void    *global_addr = reinterpret_cast<void*>(src);

    CUtensorMapDataType     tma_format      = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; 
    CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    CUtensorMapSwizzle      tma_swizzle     = CU_TENSOR_MAP_SWIZZLE_NONE; 

    uint64_t gmem_shape[tma_dim] = {
        8, 
        8,
        (global_tile_width/8), 
        2,
        (global_tile_height/16)
    };
    uint64_t gmem_stride[tma_dim] = {
        sizeof(bf16), 
        8 * sizeof(bf16),
        global_tile_width * sizeof(bf16), 
        8 * 2 * sizeof(bf16),
        8 * global_tile_width * sizeof(bf16),
    };


    uint32_t smem_shape[tma_dim]  = {
        8, 
        8,
        (shared_tile_width/8), 
        2,
        (shared_tile_height/16)
    };
    uint32_t smem_stride[tma_dim] = {1, 1, 1, 1, 1};

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    // if interleave is 32b, global address must be 32-byte aligned
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert((reinterpret_cast<uint64_t>(global_addr) & 0b11111) == 0);
    }

    // if interleave is none, then dim >= 3
    if (tma_interleave != CU_TENSOR_MAP_INTERLEAVE_NONE) {
        assert(tma_dim >= 3);
    }

    // all elements of gmem_shape must be non-zero
    for (int i = 0; i < tma_dim; i++) {
        assert(gmem_shape[i] != 0);
    }

    for (int i = 0; i < tma_dim; i++) {
        // ignore first value when interleave is none
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            continue; 
        }

        // all elements of gmem_stride must be a multiple of 16
        assert(gmem_stride[i] % 16 == 0);

        // multile of 32 for 32B interleave
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
            assert(gmem_stride[i] % 32 == 0);
        }
    }

    // all smem_shape elements must be non-zero and less than= 256
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_shape[i] != 0);
        assert(smem_shape[i] <= 256);
        if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && i == 0) {
            assert(smem_shape[0] * sizeof(bf16) % 16 == 0);
        }
    }

    // all smem_stride elements must be non-zero and less than= 8
    for (int i = 0; i < tma_dim; i++) {
        assert(smem_stride[i] != 0);
        assert(smem_stride[i] <= 8);
    }

    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        int swizzle_size = 32; 
        switch (tma_swizzle) {
            case CU_TENSOR_MAP_SWIZZLE_32B:
                swizzle_size = 32;
                break;
            case CU_TENSOR_MAP_SWIZZLE_64B:
                swizzle_size = 64;
                break;
            case CU_TENSOR_MAP_SWIZZLE_128B:
                swizzle_size = 128;
                break;
            default:
                assert(false);
        }
        assert(smem_shape[0] * sizeof(bf16) <= swizzle_size);
    }

    // if interleave is 32B, then swizzle must be 32B
    if (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
        assert(tma_swizzle == CU_TENSOR_MAP_SWIZZLE_32B);
    }

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE) ? (&gmem_stride[0] + 1) : (&gmem_stride[0]);
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
    
    assert(result == CUDA_SUCCESS);
}


template<int height, int width>
__device__ static inline void store_async(const st<bf16, height, width, st_naive_row_layout> &src, void *dst_tma_map, int tile_idx) {
    if (threadIdx.x % 32 == 0) {
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
    if (threadIdx.x % 32 == 0) {
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

// Implementation for st_xor_row_layout
template<int height, int width>
__device__ static inline void store_async(const st<bf16, height, width, st_xor_row_layout> &src, void *dst_tma_map, int tile_idx) {
    if (threadIdx.x % 32 == 0) {
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
__device__ static inline void load_async(st<bf16, height, width, st_xor_row_layout> &dst, void const* const src_tma_map, int tile_idx, uint64_t& barrier) {
    if (threadIdx.x % 32 == 0) {
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
__device__ static inline void store_async(const st<bf16, height, width, wgmma_row_layout> &src, void *dst_tma_map, int tile_idx) {
    if (threadIdx.x % 32 == 0) {
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
    if (threadIdx.x % 32 == 0) {
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
__device__ static inline void store_async(const st<bf16, height, width, wgmma_col_layout> &src, void *dst_tma_map, int tile_idx) {
    if (threadIdx.x % 32 == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));

        int32_t crd0 = 0;  
        int32_t crd1 = 0; 
        int32_t crd2 = 0;
        int32_t crd3 = tile_idx * (src.cols);
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
template<int height, int width, st_wgmma_col_layout wgmma_col_layout>
__device__ static inline void load_async(st<bf16, height, width, wgmma_col_layout> &dst, void const* const src_tma_map, int tile_idx, uint64_t& barrier) {
    if (threadIdx.x % 32 == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));

        int32_t crd0 = 0;  
        int32_t crd1 = 0; 
        int32_t crd2 = 0;
        int32_t crd3 = tile_idx * (dst.cols);
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

__device__ static inline void init_barrier(uint64_t& barrier, int tc) {
    if (threadIdx.x % 32 == 0) {
        void const* const ptr = &barrier;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile ("mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(bar_ptr), "r"(tc));
    }
}

__device__ static inline void set_barrier_bytes(uint64_t& barrier, uint32_t bytes) {
    if (threadIdx.x % 32 == 0) {
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

template<int height, int width>
__device__ static inline void prefetch(st<bf16, height, width, st_naive_row_layout> &dst, void const* const src_tma_map, int tile_idx) {
    if (threadIdx.x % 32 == 0) {
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

template<int height, int width>
__device__ static inline void prefetch(st<bf16, height, width, st_xor_row_layout> &dst, void const* const src_tma_map, int tile_idx) {
    if (threadIdx.x % 32 == 0) {
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
    if (threadIdx.x % 32 == 0) {
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
    if (threadIdx.x % 32 == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);

        int32_t crd0 = 0;  
        int32_t crd1 = 0; 
        int32_t crd2 = 0;
        int32_t crd3 = tile_idx * (dst.cols);
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

__device__ static inline void commit_group() {
    if (threadIdx.x % 32 == 0) {
        asm volatile("cp.async.bulk.commit_group;");
    } 
}

template <int N>
__device__ static inline void wait_for_store_read() {
    asm volatile (
        "cp.async.bulk.wait_group.read %0;"
        :
        : "n"(N)
        : "memory"
    );
    __syncwarp();
}

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