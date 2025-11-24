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

template <typename T>
__device__ static inline uint64_t matrix_descriptor_raw(
    T *addr,
    uint32_t leading_dim_offset,
    uint32_t stride_dim_offset,
    uint32_t swizzle_mode
) {
#ifdef KITTENS_BLACKWELL
    // see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-shared-memory-descriptor
    return matrix_descriptor_encode(reinterpret_cast<uint64_t>(addr)) | 
           (1llu << 46) | // needed for blackwell shared memory descriptors
#else
    return matrix_descriptor_encode(reinterpret_cast<uint64_t>(addr)) |
#endif
           matrix_descriptor_encode((uint64_t)leading_dim_offset) << 16 |
           matrix_descriptor_encode((uint64_t)stride_dim_offset) << 32 |
           (uint64_t)swizzle_mode << 62;
}

} // namespace detail

template<kittens::ducks::st::all _ST, int MN_major>
struct st_descriptor {
    using identifier = ducks::st_descriptor::identifier;
    using ST = _ST;
    using T = typename ST::T;
    static constexpr int rows = ST::rows;
    static constexpr int cols = ST::cols;
    static constexpr bool swizzle = ST::swizzle;
    static_assert(swizzle, "Non-swizzled descriptor is not supported yet.");
    uint64_t base_desc;
    __device__ inline st_descriptor(const ST &tile) {
        if constexpr (MN_major) { // MN major mode (i.e., K x M for A matrix, K x N for B matrix)
            if constexpr ((ST::cols/TILE_COL_DIM<T>)%4 == 0)
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 2048*ST::rows/TILE_ROW_DIM<T>, 1024, 1);
            else if constexpr ((ST::cols/TILE_COL_DIM<T>)%2 == 0)
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 1024*ST::rows/TILE_ROW_DIM<T>, 512, 2);
            else
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 512*ST::rows/TILE_ROW_DIM<T>, 256, 3);
        }
        else { // K major mode (i.e., M x K for A matrix, N x K for B matrix)
            if constexpr ((ST::cols/TILE_COL_DIM<T>)%4 == 0)
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 16 /* does not matter */, 1024, 1);
            else if constexpr ((ST::cols/TILE_COL_DIM<T>)%2 == 0)
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 16 /* does not matter */, 512, 2);
            else
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 16 /* does not matter */, 256, 3);
        }
    }
    __device__ inline st_descriptor(const st_descriptor<ST, MN_major> &other) : base_desc(other.base_desc) {} // copy constructor
    __device__ inline uint64_t chunk_descriptor(int chunk_idx) {
        // Return the n-th chunk along the K dimension.
        // In MMA instructions, K per tensor core call is always 32 bytes
        //   ex. Hopper: K=32 for FP8, K=16 for BF16/FP16, K=8 for TF32)
        //   ex. Blackwell: K=64 for FP4, K=32 for FP8, K=16 for BF16/FP16, K=8 for TF32 (*For FP4, K=96 also possible, but we don't support yet)
        // So for MN-major, this is same as asking "how to forward 32 bytes worth of elements (=K elements) in the stride dimension?"
        // And for K-major, "how to forward K elements in the leading dimension?"
        if constexpr (MN_major) { // MN major mode (i.e., K x M for A matrix, K x N for B matrix)
            if constexpr ((ST::cols/TILE_COL_DIM<T>)%4 == 0) { // 128B swizzle: 
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*2048);
            }
            else if constexpr ((ST::cols/TILE_COL_DIM<T>)%2 == 0) {
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*1024);
            }
            else {
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*512);
            }
        }
        else { // K major mode (i.e., M x K for A matrix, N x K for B matrix)
            if constexpr ((ST::cols/TILE_COL_DIM<T>)%4 == 0) {
                // 128B swizzle: 4 chunks fit within swizzle bytes; move on to next every 4 chunks (rows * 128B swizzle bytes)
                return base_desc + detail::matrix_descriptor_encode((chunk_idx%4)*32 + (chunk_idx/4)*(ST::rows/TILE_ROW_DIM<T>)*2048);
            }
            else if constexpr ((ST::cols/TILE_COL_DIM<T>)%2 == 0) {
                // 64B swizzle: 2 chunks fit within swizzle bytes; move on to next every 2 chunks (rows * 64B swizzle bytes)
                return base_desc + detail::matrix_descriptor_encode((chunk_idx%2)*32 + (chunk_idx/2)*(ST::rows/TILE_ROW_DIM<T>)*1024);
            }
            else {
                // 32B swizzle: Entire chunk fits within swizzle bytes; move on to next on every chunk (rows * 32B swizzle bytes)
                return base_desc + detail::matrix_descriptor_encode(chunk_idx*(ST::rows/TILE_ROW_DIM<T>)*512);
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
