/**
 * @file
 * @brief The ThunderKittens shared memory descriptors, used for Hopper and Blackwell tensor cores.
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

template<typename T, int rows, int cols, bool MN_major, bool swizzle>
__device__ static inline uint64_t matrix_descriptor_raw(uint64_t addr) {
    static constexpr int height = rows / kittens::TILE_ROW_DIM<T>;
    static constexpr int width = cols / kittens::TILE_COL_DIM<T>;
#ifdef KITTENS_BLACKWELL
    // see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-shared-memory-descriptor
    uint64_t desc = matrix_descriptor_encode(addr) | (1llu<<46); // needed for blackwell shared memory descriptors
#else
    uint64_t desc = matrix_descriptor_encode(addr);
#endif
    if constexpr (MN_major) { // MN major mode (i.e., K x M for A matrix, K x N for B matrix)
        if constexpr (!swizzle) {
            desc |= matrix_descriptor_encode(1) << 16; // not used
            desc |= matrix_descriptor_encode(1) << 32; // not used
            desc |= 0llu << 62; // set no swizzle mode
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
    else { // K major mode (i.e., M x K for A matrix, N x K for B matrix)
        if constexpr (!swizzle) {
            desc |= matrix_descriptor_encode(sizeof(T)) << 16; // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-leading-dimension-byte-offset
            desc |= matrix_descriptor_encode(sizeof(T) * cols * 8) << 32; // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-stride-dimension-byte-offset
            desc |= 0llu << 62; // set no swizzle mode
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

template<kittens::ducks::st::all _ST, int MN_major>
struct st_descriptor {
    using identifier = ducks::st_descriptor::identifier;
    using ST = _ST;
    using T = typename ST::T;
    static constexpr int rows = ST::rows;
    static constexpr int cols = ST::cols;
    static constexpr int height = ST::height;
    static constexpr int width  = ST::width;
    static constexpr bool swizzle = ST::swizzle;
    uint64_t base_desc;
    __device__ inline st_descriptor(const ST &tile) : base_desc(detail::matrix_descriptor_raw<T, rows, cols, MN_major, swizzle>((uint64_t)(&tile.data[0]))) {}
    __device__ inline st_descriptor(const st_descriptor<ST, MN_major> &other) : base_desc(other.base_desc) {} // copy constructor
    __device__ inline uint64_t chunk_descriptor(int chunk_idx) {
        // Return the n-th chunk along the K dimension.
        // In MMA instructions, K per tensor core call is always 32 bytes
        //   ex. Hopper: K=32 for FP8, K=16 for BF16/FP16, K=8 for TF32)
        //   ex. Blackwell: K=64 for FP4, K=32 for FP8, K=16 for BF16/FP16, K=8 for TF32 (*For FP4, K=96 also possible, but we don't support yet)
        // So for MN-major, this is same as asking "how to forward 32 bytes worth of elements (=K elements) in the stride dimension?"
        // And for K-major, "how to forward K elements in the leading dimension?"
        if constexpr (MN_major) { // MN major mode (i.e., K x M for A matrix, K x N for B matrix)
            if constexpr (!swizzle) {
                // For no swizzle mode, this is just moving along the row dimension; easy!
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*cols*(32/sizeof(T)));
            } else if constexpr (ST::width%4 == 0) { // 128B swizzle: 
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*2048);
            }
            else if constexpr (ST::width%2 == 0) {
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*1024);
            }
            else {
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*512);
            }
        }
        else { // K major mode (i.e., M x K for A matrix, N x K for B matrix)
            if constexpr (!swizzle) {
                // For no swizzle mode, this is just moving along the column dimension; easy!
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*32);
            } else if constexpr (ST::width%4 == 0) {
                // 128B swizzle: 4 chunks fit within swizzle bytes; move on to next every 4 chunks (rows * 128B swizzle bytes)
                return base_desc + detail::matrix_descriptor_encode((chunk_idx%4)*32 + (chunk_idx/4)*ST::height*2048);
            }
            else if constexpr (ST::width%2 == 0) {
                // 64B swizzle: 2 chunks fit within swizzle bytes; move on to next every 2 chunks (rows * 64B swizzle bytes)
                return base_desc + detail::matrix_descriptor_encode((chunk_idx%2)*32 + (chunk_idx/2)*ST::height*1024);
            }
            else {
                // 32B swizzle: Entire chunk fits within swizzle bytes; move on to next on every chunk (rows * 32B swizzle bytes)
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