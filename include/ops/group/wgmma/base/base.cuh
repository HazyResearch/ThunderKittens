#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
namespace ducks {
namespace wgmma {
struct normal_descriptor_identifier {}; // abstract base type
struct transposed_descriptor_identifier {}; // abstract base type

template<typename T>
concept normal_layout = (
    std::is_same_v<T, st_layout::wgmma_interleave>   ||
    std::is_same_v<T, st_layout::wgmma_swizzle> 
);
template<typename T>
concept transposed_layout = (
    std::is_same_v<T, st_layout::wgmma_interleave>   ||
    std::is_same_v<T, st_layout::wgmma_swizzle> 
);
template<typename T> concept st_normal     = ducks::st::all<T> && normal_layout<typename T::layout>;
template<typename T> concept st_transposed = ducks::st::all<T> && transposed_layout<typename T::layout>;
}
}
namespace wgmma {

// see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

template<kittens::ducks::st::all _ST>
struct normal_descriptor {
    using identifier = ducks::wgmma::normal_descriptor_identifier;
    using ST = _ST;
    static constexpr int height = ST::height;
    static constexpr int width  = ST::width;
    using T = ST::T;
    static_assert(kittens::ducks::wgmma::st_normal<ST>, "Layout must be a normal wgmma layout.");
    uint64_t base_desc;
    __device__ inline normal_descriptor(const ST &tile) {
        base_desc = matrix_descriptor_encode((uint64_t)(&tile.data[0]));
        if constexpr (std::is_same_v<typename ST::layout, ducks::st_layout::wgmma_swizzle>) {
            if constexpr (ST::width%4 == 0) {
                base_desc |= matrix_descriptor_encode((uint64_t)16) << 16;
                base_desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
                base_desc |= 1llu << 62; // set wgmma_swizzle mode
            }
            else if constexpr (ST::width%2 == 0) {
                base_desc |= matrix_descriptor_encode((uint64_t)16) << 16;
                base_desc |= matrix_descriptor_encode((uint64_t)512) << 32;
                base_desc |= 2llu << 62; // set wgmma_swizzle mode
            }
            else {
                base_desc |= matrix_descriptor_encode((uint64_t)16) << 16;
                base_desc |= matrix_descriptor_encode((uint64_t)256) << 32;
                base_desc |= 3llu << 62; // set wgmma_swizzle mode
            }
        }
        else {
            base_desc |= matrix_descriptor_encode((uint64_t)128) << 16;
            base_desc |= matrix_descriptor_encode((uint64_t)256*ST::width) << 32;
            // no swizzle mode
        }
    }
    __device__ inline normal_descriptor(const normal_descriptor<ST> &other) : base_desc(other.base_desc) {} // copy constructor
    __device__ inline uint64_t chunk_descriptor(int chunk_idx) {
        if constexpr (std::is_same_v<typename ST::layout, ducks::st_layout::wgmma_swizzle>) {
            if constexpr (ST::width%4 == 0) {
                return base_desc + matrix_descriptor_encode((chunk_idx%4)*32 + (chunk_idx/4)*ST::height*2048);
            }
            else if constexpr (ST::width%2 == 0) {
                return base_desc + matrix_descriptor_encode((chunk_idx%2)*32 + (chunk_idx/2)*ST::height*1024);
            }
            else {
                return base_desc + matrix_descriptor_encode(chunk_idx*ST::height*512);
            }
        }
        else {
            return base_desc + matrix_descriptor_encode(chunk_idx*(128 * 2));
        }
    }
};
template<kittens::ducks::st::all _ST>
struct transposed_descriptor {
    using identifier = ducks::wgmma::transposed_descriptor_identifier;
    using ST = _ST;
    static constexpr int height = ST::height;
    static constexpr int width  = ST::width;
    using T = ST::T;
    static_assert(kittens::ducks::wgmma::st_transposed<ST>, "Layout must be a transposed wgmma layout.");
    uint64_t base_desc;
    __device__ inline transposed_descriptor(const ST &tile) {
        base_desc  = matrix_descriptor_encode((uint64_t)(&tile.data[0]));
        if constexpr (std::is_same_v<typename ST::layout, ducks::st_layout::wgmma_swizzle>) {
            if constexpr (ST::width%4 == 0) {
                base_desc |= matrix_descriptor_encode((uint64_t)1024*ST::height*2) << 16;
                base_desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
                base_desc |= 1llu << 62; // set wgmma_swizzle mode
            }
            else if constexpr (ST::width%2 == 0) {
                base_desc |= matrix_descriptor_encode((uint64_t)512*ST::height*2) << 16;
                base_desc |= matrix_descriptor_encode((uint64_t)512) << 32;
                base_desc |= 2llu << 62; // set wgmma_swizzle mode
            }
            else {
                base_desc |= matrix_descriptor_encode((uint64_t)256*ST::height*2) << 16;
                base_desc |= matrix_descriptor_encode((uint64_t)256) << 32;
                base_desc |= 3llu << 62; // set wgmma_swizzle mode
            }
        }
        else {
            base_desc |= matrix_descriptor_encode((uint64_t)256*ST::width) << 16;
            base_desc |= matrix_descriptor_encode((uint64_t)128) << 32;
        }
    }
    __device__ inline transposed_descriptor(const transposed_descriptor<ST> &other) : base_desc(other.base_desc) {} // copy constructor
    __device__ inline uint64_t chunk_descriptor(int chunk_idx) {
        if constexpr (std::is_same_v<typename ST::layout, ducks::st_layout::wgmma_swizzle>) {
            if constexpr (ST::width%4 == 0) {
                return base_desc + matrix_descriptor_encode(chunk_idx*2048);
            }
            else if constexpr (ST::width%2 == 0) {
                return base_desc + matrix_descriptor_encode(chunk_idx*1024);
            }
            else {
                return base_desc + matrix_descriptor_encode(chunk_idx*512);
            }
        }
        else {
            return base_desc + matrix_descriptor_encode(chunk_idx*(256*ST::width * 2));
        }
    }
};
template<kittens::ducks::st::all ST, int transpose> using descriptor = std::conditional_t<transpose, transposed_descriptor<ST>, normal_descriptor<ST>>;

// templated wrapper for PTX
template<typename T_D, typename T_AB, int width, int trans_a, int trans_b>
struct base {
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
};

#include "4x1.impl"
#include "4x2.impl"
#include "4x3.impl"
#include "4x4.impl"
#include "4x6.impl"
#include "4x8.impl"
#include "4x12.impl"
#include "4x16.impl"

// can add more later, just annoying

}
namespace ducks {
namespace wgmma {
template<typename T> concept input_normal     = st_normal<T> ||
    (requires {typename T::identifier;} && std::is_same_v<typename T::identifier, normal_descriptor_identifier>);
template<typename T> concept input_transposed = st_transposed<T> ||
    (requires {typename T::identifier;} && std::is_same_v<typename T::identifier, transposed_descriptor_identifier>);
namespace detail {
template<typename T> struct st_getter { using type = typename T::ST; };
template<ducks::st::all T> struct st_getter<T> { using type = T; };
template<typename T> using get_st = typename st_getter<T>::type;
}
}
}
}