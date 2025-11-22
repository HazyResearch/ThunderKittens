/**
 * @file
 * @brief The ThunderKittens shared tile descriptors, used for Hopper and Blackwell tensor cores.
 */

#pragma once

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)

#include "../../common/common.cuh"
#include "st.cuh"
#include "cst.cuh"

namespace kittens {
namespace ducks {
namespace st_descriptor {
struct identifier {};
}
}

namespace detail {

// See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

template<int width, int height, bool transpose, bool swizzle>
__device__ static inline uint64_t st_descriptor_raw(uint64_t addr) {
#ifdef KITTENS_BLACKWELL
    // see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-shared-memory-descriptor
    uint64_t desc = matrix_descriptor_encode(addr) | (1llu<<46); // needed for blackwell shared memory descriptors
#else
    uint64_t desc = matrix_descriptor_encode(addr);
#endif
    if constexpr (transpose) { // transpose mode
        if constexpr (!swizzle) {
            // desc |= matrix_descriptor_encode(??) << 16;
            // desc |= matrix_descriptor_encode(??) << 32;
            // desc |= 0llu << 62; // set wgmma_swizzle mode
        } else if constexpr (width%4 == 0) {
            desc |= matrix_descriptor_encode((uint64_t)2048*height) << 16;
            desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
            desc |= 1llu << 62; // set wgmma_swizzle mode
        }
        else if constexpr (width%2 == 0) {
            desc |= matrix_descriptor_encode((uint64_t)1024*height) << 16;
            desc |= matrix_descriptor_encode((uint64_t)512) << 32;
            desc |= 2llu << 62; // set wgmma_swizzle mode
        }
        else {
            desc |= matrix_descriptor_encode((uint64_t)512*height) << 16;
            desc |= matrix_descriptor_encode((uint64_t)256) << 32;
            desc |= 3llu << 62; // set wgmma_swizzle mode
        }
    }
    else { // normal mode
        if constexpr (!swizzle) {
            // desc |= matrix_descriptor_encode(??) << 16;
            // desc |= matrix_descriptor_encode(??) << 32;
            // desc |= 0llu << 62; // set wgmma_swizzle mode
        } else if constexpr (width%4 == 0) {
            desc |= matrix_descriptor_encode((uint64_t)16) << 16;   // this line doesn't matter
            desc |= matrix_descriptor_encode((uint64_t)1024) << 32; // 128 byte swizzle x 8 for core matrix rows
            desc |= 1llu << 62; // set wgmma_swizzle mode
        }
        else if constexpr (width%2 == 0) {
            desc |= matrix_descriptor_encode((uint64_t)16) << 16;  // this line doesn't matter
            desc |= matrix_descriptor_encode((uint64_t)512) << 32; // 64 byte swizzle x 8 for core matrix rows
            desc |= 2llu << 62; // set wgmma_swizzle mode
        }
        else {
            desc |= matrix_descriptor_encode((uint64_t)16) << 16;  // this line doesn't matter
            desc |= matrix_descriptor_encode((uint64_t)256) << 32; // 32 byte swizzle x 8 for core matrix rows
            desc |= 3llu << 62; // set wgmma_swizzle mode
        }
    }
    return desc;
}

} // namespace detail

template<kittens::ducks::st::all _ST, int transpose>
struct st_descriptor {
    using identifier = ducks::st_descriptor::identifier;
    using ST = _ST;
    static constexpr int height = ST::height;
    static constexpr int width  = ST::width;
    using T = ST::T;
    uint64_t base_desc;
    __device__ inline st_descriptor(const ST &tile) : base_desc(detail::st_descriptor_raw<ST::width, ST::height, transpose, ST::swizzle>((uint64_t)(&tile.data[0]))) {}
    __device__ inline st_descriptor(const st_descriptor<ST, transpose> &other) : base_desc(other.base_desc) {} // copy constructor
    __device__ inline uint64_t chunk_descriptor(int chunk_idx) {
        if constexpr (transpose) { // transpose mode
            if constexpr (!ST::swizzle) {
                // return base_desc + detail::matrix_descriptor_encode(chunk_idx*??);
                return 0;
            } else if constexpr (ST::width%4 == 0) {
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*2048);
            }
            else if constexpr (ST::width%2 == 0) {
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*1024);
            }
            else {
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*512);
            }
        }
        else { // normal mode
            if constexpr (!ST::swizzle) {
                // return base_desc + detail::matrix_descriptor_encode(chunk_idx*??);
                return 0;
            } else if constexpr (ST::width%4 == 0) {
                return base_desc + detail::matrix_descriptor_encode((chunk_idx%4)*32 + (chunk_idx/4)*ST::height*2048);
            }
            else if constexpr (ST::width%2 == 0) {
                return base_desc + detail::matrix_descriptor_encode((chunk_idx%2)*32 + (chunk_idx/2)*ST::height*1024);
            }
            else {
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*ST::height*512);
            }
        }
    }
};

namespace ducks {
namespace st_descriptor {
// input refers to either an ST directly or to a pre-generated descriptor, which can save cycles in certain situations.
template<typename T> concept input = ducks::st::all<T> || (requires {typename T::identifier;} && std::is_same_v<typename T::identifier, ducks::st_descriptor::identifier>);
template<typename T> concept complex_input = ducks::cst::all<T>;
namespace detail {
template<typename T> struct st_getter { using type = typename T::ST; };
template<ducks::st::all T> struct st_getter<T> { using type = T; };
template<ducks::cst::all T> struct st_getter<T> { using type = T::component; };
template<typename T> using get_st = typename st_getter<T>::type;
} // namespace detail
} // namespace st_descriptor
} // namespace ducks

} // namespace kittens

#endif