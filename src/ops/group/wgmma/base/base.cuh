#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
namespace detail {

// see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

// wgmma helpers
template<int height, int width, ducks::st_layout::all L>
struct wgmma_descriptor { static_assert("Asbtract wgmma descriptor struct should never be instantiated."); };

template<int height, int width>
struct wgmma_descriptor<height, width, ducks::st_layout::interleave> {
    __device__ static inline uint64_t normal(uint64_t start_addr, int chunk_idx) {
        uint64_t desc = 0x0000000000000000;
        desc |= matrix_descriptor_encode(start_addr + chunk_idx*(128 * 2));
        desc |= matrix_descriptor_encode((uint64_t)128) << 16;
        desc |= matrix_descriptor_encode((uint64_t)256*width) << 32;
        return desc;
    }
    __device__ static inline uint64_t transposed(uint64_t start_addr, int chunk_idx) {
        uint64_t desc = 0x0000000000000000;
        desc |= matrix_descriptor_encode(start_addr + chunk_idx*(256*width * 2));
        desc |= matrix_descriptor_encode((uint64_t)256*width) << 16;
        desc |= matrix_descriptor_encode((uint64_t)128) << 32;
        return desc;
    }
};
template<int height, int width>
struct wgmma_descriptor<height, width, ducks::st_layout::xor_swizzle> {
    __device__ static inline uint64_t normal(uint64_t start_addr, int chunk_idx) {
        uint64_t desc = 0x0000000000000000;
        desc |= matrix_descriptor_encode(start_addr + chunk_idx*(128 * 2));
        desc |= matrix_descriptor_encode((uint64_t)128) << 16;
        desc |= matrix_descriptor_encode((uint64_t)256*width) << 32;
        return desc;
    }
    __device__ static inline uint64_t transposed(uint64_t start_addr, int chunk_idx) {
        uint64_t desc = 0x0000000000000000;
        desc |= matrix_descriptor_encode(start_addr + chunk_idx*(256*width * 2));
        desc |= matrix_descriptor_encode((uint64_t)256*width) << 16;
        desc |= matrix_descriptor_encode((uint64_t)128) << 32;
        return desc;
    }
};

template<int transpose, ducks::st::all ST>
__device__ static inline uint64_t make_descriptor(const ST &tile, int chunk_idx) {
    if constexpr (transpose) {
        static_assert(ducks::st_layout::wgmma_transposed<typename ST::layout>, "Tile must have a transposable wgmma layout to be used here.");
        return wgmma_descriptor<ST::underlying_height, ST::underlying_width, typename ST::layout>::transposed((uint64_t)(tile.data), chunk_idx);
    }
    else {
        static_assert(ducks::st_layout::wgmma_normal<typename ST::layout>, "Tile must have a normal wgmma layout to be used here.");
        return wgmma_descriptor<ST::underlying_height, ST::underlying_width, typename ST::layout>::normal((uint64_t)(tile.data), chunk_idx);
    }
}
// templated wrapper for PTX
template<int width, int trans_a, int trans_b>
struct wgmma_base {
    __device__ static inline void rt_st(
        rt_fl<1, width, ducks::rt_layout::row> &dst,
        const rt_bf<1, 1, ducks::rt_layout::row> & a_rt,
        const uint64_t b_st_desc,
        int scale_d = 1
    );
    __device__ static inline void st_st(
        rt_fl<1, width, ducks::rt_layout::row> &dst,
        const uint64_t a_st_desc,
        const uint64_t b_st_desc,
        int scale_d = 1
    );
    template<ducks::st::all ST> __device__ static inline uint64_t a_desc(const ST &tile, int chunk_idx) {
        return make_descriptor<trans_a>(tile, chunk_idx);
    }
    template<ducks::st::all ST> __device__ static inline uint64_t b_desc(const ST &tile, int chunk_idx) {
        return make_descriptor<trans_b>(tile, chunk_idx);
    }
};

#include "4x1.impl"
#include "4x2.impl"
#include "4x3.impl"
#include "4x4.impl"

// can add bigger ones later, just annoying
// #include "4x5.impl"
// #include "4x6.impl"
// #include "4x7.impl"
#include "4x8.impl"

}
}