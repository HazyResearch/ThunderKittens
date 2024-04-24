#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
namespace detail {


// wgmma helpers

// see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

template<int width, int height, ducks::st_layout::wgmma_normal L>
__device__ inline uint64_t normal_descriptor(uint64_t start_addr, int chunk_idx);

template<int width, int height>
__device__ inline uint64_t normal_descriptor<width, height, ducks::st_layout::wgmma_0b>(uint64_t start_addr, int chunk_idx) {
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(start_addr + chunk_idx*(128 * 2));
    desc |= matrix_descriptor_encode((uint64_t)128) << 16;
    desc |= matrix_descriptor_encode((uint64_t)256*width) << 32;
    return desc;
}

template<int width, int height, ducks::st_layout::wgmma_transposed L>
__device__ inline uint64_t transposed_descriptor(uint64_t start_addr, int chunk_idx);

template<int width, int height>
__device__ inline uint64_t transposed_descriptor<width, height, ducks::st_layout::wgmma_0b>(uint64_t start_addr, int chunk_idx) {
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(start_addr + chunk_idx*(256*width * 2));
    desc |= matrix_descriptor_encode((uint64_t)256*width) << 16;
    desc |= matrix_descriptor_encode((uint64_t)128) << 32;
    return desc;
}


template<ducks::st::all ST>
__device__ inline uint64_t normal_descriptor(const ST &tile, int chunk_idx) {
    static_assert(wgmma_normal<typename ST::layout>, "Tile must have a transposable wgmma layout to be called here.");
    return normal_descriptor<ST::underlying_width, ST::underlying_height, typename ST::layout>((uint64_t)(tile.data), chunk_idx);
}
template<ducks::st::all ST>
__device__ inline uint64_t transposed_descriptor(const ST &tile, int chunk_idx) {
    static_assert(wgmma_transposed<typename ST::layout>, "Tile must have a transposable wgmma layout to be called here.");
    return transposed_descriptor<ST::underlying_width, ST::underlying_height, typename ST::layout>((uint64_t)(tile.data), chunk_idx);
}

// template<wgmma_layout layout>
// __device__ inline uint64_t matrix_descriptor(uint64_t start_addr) {
//     uint64_t desc = 0x0000000000000000;
//     desc |= matrix_descriptor_encode(start_addr);
//     desc |= matrix_descriptor_encode((uint64_t)128) << 16;
//     desc |= matrix_descriptor_encode((uint64_t)256) << 32;
//     uint64_t base_offset = 0;
//     if constexpr (layout::swizzling_mode == 3) {
//         if((uint64_t)(start_addr) % 256 != 0) {
//             base_offset = (start_addr >> 0x7) & 0x7;
//         }
//     }
//     if constexpr (layout::swizzling_mode == 2) {
//         if((uint64_t)(start_addr) % 512 != 0) {
//             base_offset = (start_addr >> 0x7) & 0x7;
//         }
//     }
//     if constexpr (layout::swizzling_mode == 1) {
//         if((uint64_t)(start_addr) % 1024 != 0) {
//             base_offset = (start_addr >> 0x7) & 0x7;
//         }
//     }
//     desc |= ((uint64_t)base_offset) << 49;
//     desc |= ((uint64_t)layout::swizzling_mode) << 62;
//     return desc;
// }

// templated wrapper for PTX
template<int width, int trans_b>
struct wgmma_base {
    __device__ static inline void rt_st(
        rt_fl<1, width, ducks::rt_layout::row> &dst,
        const rt_bf<1, 1, ducks::rt_layout::row> & a_rt,
        const uint64_t b_st_desc
    );
    __device__ static inline void st_st(
        rt_fl<1, width, ducks::rt_layout::row> &dst,
        const uint64_t a_st_desc,
        const uint64_t b_st_desc
    );
};

#include "4x1.impl"
#include "4x2.impl"
#include "4x3.impl"
#include "4x4.impl"

// can add bigger ones later
// #include "4x5.impl"
// #include "4x6.impl"
// #include "4x7.impl"
#include "4x8.impl"

}
}