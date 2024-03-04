#pragma once

#include "../common/common.cuh"
#include "../shared_tile/shared_tile.cuh"

// CUTLASS 5D copy
// struct SM90_TMA_LOAD_5D
// {
//   CUTE_HOST_DEVICE static void
//   copy(void const* const desc_ptr, uint64_t& smem_mbar,
//        void const* const smem_ptr,
//        int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
//   {
// #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
//     uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
//     uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
//     uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
//     asm volatile (
//       "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes"
//       " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
//       :
//       : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
//         "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
//       : "memory");
// #else
//     CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
// #endif
//   }
// };

// CUTLASS 2D copy
// struct SM90_TMA_LOAD_2D
// {
//   CUTE_HOST_DEVICE static void
//   copy(void const* const desc_ptr, uint64_t& smem_mbar,
//        void const* const smem_ptr,
//        int32_t const& crd0, int32_t const& crd1)
//   {
// #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
//     uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
//     uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
//     uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
//     asm volatile (
//       "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
//       " [%0], [%1, {%3, %4}], [%2];"
//       :
//       : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
//         "r"(crd0), "r"(crd1)
//       : "memory");
// #else
//     CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
// #endif
//   }
// };

#include <cuda.h>

namespace kittens {
namespace tma {

// __device__ static inline void execute_2d_load(st<bf16, height, width, st_naive_row_layout> &dst, CUtensorMap &tma_map, uint64_t &barrier,
//                                                 int32_t const& crd0, int32_t const& crd1)
// {
    
// }

// Implementation for st_naive_row_layout
template<int height, int width>
__device__ static inline void load_async(st<bf16, height, width, st_naive_row_layout> &dst, const bf16 *src, const int row_stride, uint64_t &barrier) {
    using tmaTensorMap = CUtensorMap;

    tmaTensorMap            tma_map         = {0};
    void*                   global_address  = (void*)src;
    CUtensorMapDataType     tma_format      = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; // hardcoded in the same way as global_to_shared.cuh
    constexpr int           tma_dim         = 2;
    
    uint32_t global_dim[2]; 
    global_dim[0] = dst.rows; 
    global_dim[1] = dst.cols; 
    const uint64_t* g_dims = reinterpret_cast<const uint64_t*>(&global_dim);

    uint32_t global_strides[2];
    global_strides[0] = 1; 
    global_strides[1] = dst.cols * sizeof(bf16); 
    const uint64_t* g_strides = reinterpret_cast<const uint64_t*>(&global_strides);

    uint32_t smem_dim[2]; 
    smem_dim[0] = dst.rows; 
    smem_dim[1] = dst.cols; 

    uint32_t smem_strides[2]; 
    smem_strides[0] = 1;
    smem_strides[1] = 1; 
    
    CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapSwizzle      tma_swizzle     = CU_TENSOR_MAP_SWIZZLE_NONE;
    CUtensorMapL2promotion  tma_l2Promo     = CU_TENSOR_MAP_L2_PROMOTION_L2_128B; // try 64B and 256B - is this something we give the user control over?
    CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map,
        tma_format,
        tma_dim,
        global_address,
        g_dims, 
        g_strides, 
        smem_dim, 
        smem_strides, 
        tma_interleave,
        tma_swizzle,
        tma_l2Promo,
        tma_oobFill
    );

    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(&tma_map);

    int32_t crd0 = 0;  
    int32_t crd1 = 0; 

    asm volatile (
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
          "r"(crd0), "r"(crd1)
        : "memory"
    );
}

// Implementation for st_xor_row_layout
template<int height, int width>
__device__ void static inline load_async(st<bf16, height, width, st_xor_row_layout> &dst, const bf16 *src, const int row_stride, uint64_t& barrier) {
    // Implementation for st_xor_row_layout
    return;
}

// Implementation for st_wgmma_row_0b_layout
template<int height, int width>
__device__ void static inline load_async(st<bf16, height, width, st_wgmma_row_0b_layout> &dst, const bf16 *src, const int row_stride, uint64_t& barrier) {
    // Implementation for st_wgmma_row_0b_layout
    return;
}


// Implementation for st_wgmma_row_32b_layout
template<int height, int width>
__device__ void static inline load_async(st<bf16, height, width, st_wgmma_row_32b_layout> &dst, const bf16 *src, const int row_stride, uint64_t& barrier) {
    // Implementation for st_wgmma_row_32b_layout
    return;
}

// Implementation for st_wgmma_row_64b_layout
template<int height, int width>
__device__ void static inline load_async(st<bf16, height, width, st_wgmma_row_64b_layout> &dst, const bf16 *src, const int row_stride, uint64_t& barrier) {
    // Implementation for st_wgmma_row_64b_layout
    return;
}

// Implementation for st_wgmma_row_128b_layout
template<int height, int width>
__device__ void static inline load_async(st<bf16, height, width, st_wgmma_row_128b_layout> &dst, const bf16 *src, const int row_stride, uint64_t& barrier) {
    // Implementation for st_wgmma_row_128b_layout
    return;
}

// Implementation for st_wgmma_col_0b_layout
template<int height, int width>
__device__ void static inline load_async(st<bf16, height, width, st_wgmma_col_0b_layout> &dst, const bf16 *src, const int row_stride, uint64_t& barrier) {
    // Implementation for st_wgmma_col_0b_layout
    return;
}

// Implementation for st_wgmma_col_32b_layout
template<int height, int width>
__device__ void static inline load_async(st<bf16, height, width, st_wgmma_col_32b_layout> &dst, const bf16 *src, const int row_stride, uint64_t& barrier) {
    // Implementation for st_wgmma_col_32b_layout
    return;
}

// Implementation for st_wgmma_col_64b_layout
template<int height, int width>
__device__ void static inline load_async(st<bf16, height, width, st_wgmma_col_64b_layout> &dst, const bf16 *src, const int row_stride, uint64_t& barrier) {
    // Implementation for st_wgmma_col_64b_layout
    return;
}

// Implementation for st_wgmma_col_128b_layout
template<int height, int width>
__device__ void static inline load_async(st<bf16, height, width, st_wgmma_col_128b_layout> &dst, const bf16 *src, const int row_stride, uint64_t& barrier) {
    // Implementation for st_wgmma_col_128b_layout
    return;
}

// template<int height, int width, st_row_layout layout>
// __device__ static inline void store_async(bf16 *dst, const st<bf16, height, width, layout> &src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
//     // each thread needs to do 1 call per width*height
//     // attempting to improve striping into dram
//     // each lane of the warp should store sequential into dram

//     int laneid = threadIdx.x % 32;

//     // we can handle this many rows each time we run a memcpy_async
//     int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
//     int memcpy_per_row = src.cols / elem_per_memcpy;
//     int total_calls = src.height * src.width;

//     #pragma unroll
//     for(int i = 0; i < total_calls; i++) {

//         int idx = i * 32 + laneid;
        
//         int row = idx / memcpy_per_row;
//         int col = (idx*elem_per_memcpy) % src.cols;

//         cuda::memcpy_async(
//             (void*)(&dst[row*row_stride + col]),
//             (void*)(&src[{row, col}]),
//             cuda::aligned_size_t<16>(sizeof(float4)),
//             barrier
//         );
//     }
// }

__device__ static inline void set_barrier_bytes(uint64_t& barrier, uint32_t bytes) {
    void const* const ptr = &barrier;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_ptr), "r"(bytes));
}

__device__ static inline void init_barrier(uint64_t& barrier, int tc, uint32_t bytes) {
    void const* const ptr = &barrier;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile ("mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(tc));
    
    set_barrier_bytes(barrier, bytes);
}

}
}