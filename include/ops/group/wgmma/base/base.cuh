#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
namespace ducks {
namespace wgmma {
template<typename T>
concept normal = (
    std::is_same_v<T, st_layout::wgmma_interleave>   ||
    std::is_same_v<T, st_layout::wgmma_swizzle> 
);
template<typename T>
concept transposed = (
    std::is_same_v<T, st_layout::wgmma_interleave>   // ||
);
template<typename T> concept st_normal     = ducks::st::all<T> && normal<typename T::layout>;
template<typename T> concept st_transposed = ducks::st::all<T> && transposed<typename T::layout>;
}
}
namespace wgmma {

// see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

// wgmma helpers
template<int height, int width, ducks::st_layout::all L>
struct descriptor { static_assert("Asbtract wgmma descriptor struct should never be instantiated."); };

template<int height, int width>
struct descriptor<height, width, ducks::st_layout::wgmma_interleave> {
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
struct descriptor<height, width, ducks::st_layout::wgmma_swizzle> {
    __device__ static inline uint64_t normal(uint64_t start_addr, int chunk_idx) {
        uint64_t desc = 0x0000000000000000;
        if constexpr (width%4 == 0) {
            desc |= matrix_descriptor_encode(start_addr + (chunk_idx%4)*32 + (chunk_idx/4)*height*2048);
            desc |= matrix_descriptor_encode((uint64_t)16) << 16;
            desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
            desc |= 1llu << 62; // set wgmma_swizzle mode
        }
        else if constexpr (width%2 == 0) {
            desc |= matrix_descriptor_encode(start_addr + (chunk_idx%2)*32 + (chunk_idx/2)*height*1024);
            desc |= matrix_descriptor_encode((uint64_t)16) << 16;
            desc |= matrix_descriptor_encode((uint64_t)512) << 32;
            desc |= 2llu << 62; // set wgmma_swizzle mode
        }
        else {
            desc |= matrix_descriptor_encode(start_addr + chunk_idx*height*512);
            desc |= matrix_descriptor_encode((uint64_t)16) << 16;
            desc |= matrix_descriptor_encode((uint64_t)256) << 32;
            desc |= 3llu << 62; // set wgmma_swizzle mode
        }
        return desc;
    }
};

template<int transpose, ducks::st::all ST>
__device__ static inline uint64_t make_descriptor(const ST &tile, int chunk_idx) {
    if constexpr (transpose) {
        static_assert(ducks::wgmma::transposed<typename ST::layout>, "Tile must have a transposable wgmma layout to be used here.");
        return descriptor<ST::underlying_height, ST::underlying_width, typename ST::layout>::transposed((uint64_t)(tile.data), chunk_idx);
    }
    else {
        static_assert(ducks::wgmma::normal<typename ST::layout>, "Tile must have a normal wgmma layout to be used here.");
        return descriptor<ST::underlying_height, ST::underlying_width, typename ST::layout>::normal((uint64_t)(tile.data), chunk_idx);
    }
}
// templated wrapper for PTX
template<int width, int trans_a, int trans_b>
struct base {
    template<typename T_D, typename T_AB> __device__ static inline void rt_st(
        rt_fl<1, width, ducks::rt_layout::row> &dst,
        const rt_bf<1, 1, ducks::rt_layout::row> & a_rt,
        const uint64_t b_st_desc,
        int scale_d = 1
    );
    template<typename T_D, typename T_AB> __device__ static inline void st_st(
        rt_fl<1, width, ducks::rt_layout::row> &dst,
        const uint64_t a_st_desc,
        const uint64_t b_st_desc,
        int scale_d = 1
    );
    // ----- DESCRIPTORS ----- //
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
#include "4x6.impl"
// #include "4x7.impl"
#include "4x8.impl"
#include "4x16.impl"

}
}