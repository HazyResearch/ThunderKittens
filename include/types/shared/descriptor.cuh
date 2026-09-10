/**
 * @file
 * @brief The ThunderKittens shared memory descriptors, used for Hopper and Blackwell tensor cores.
 */

#pragma once

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)

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

#ifdef KITTENS_SM107
constexpr uint64_t DESCRIPTOR_ADDR_MASK = 0x7FFF;
#else
constexpr uint64_t DESCRIPTOR_ADDR_MASK = 0x3FFF;
#endif
constexpr uint64_t DESCRIPTOR_OFFSET_MASK = 0x3FFF;

__device__ static inline uint64_t matrix_descriptor_encode_addr(uint64_t x) { return (x >> 4) & DESCRIPTOR_ADDR_MASK; }
__device__ static inline uint64_t matrix_descriptor_encode_offset(uint64_t x) { return (x >> 4) & DESCRIPTOR_OFFSET_MASK; }

template <typename T>
__device__ static inline uint64_t matrix_descriptor_raw(
    T *addr,
    uint32_t leading_dim_offset,
    uint32_t stride_dim_offset,
    uint32_t swizzle_mode
) {
#ifdef KITTENS_SM10X
    // see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-shared-memory-descriptor
    return matrix_descriptor_encode_addr(reinterpret_cast<uint64_t>(addr)) |
           (1llu << 46) | // needed for blackwell shared memory descriptors
#else
    return matrix_descriptor_encode_addr(reinterpret_cast<uint64_t>(addr)) |
#endif
           matrix_descriptor_encode_offset((uint64_t)leading_dim_offset) << 16 |
           matrix_descriptor_encode_offset((uint64_t)stride_dim_offset) << 32 |
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
        // See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-leading-dimension-byte-offset
        if constexpr (MN_major) { // MN major mode (i.e., K x M for A matrix, K x N for B matrix)
            if constexpr (ST::swizzle_bytes == 128) // 128B swizzle mode
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 2048*ST::rows/TILE_ROW_DIM<T>, 1024, 1);
            else if constexpr (ST::swizzle_bytes == 64) // 64B swizzle mode
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 1024*ST::rows/TILE_ROW_DIM<T>, 512, 2);
            else // 32B swizzle mode
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 512*ST::rows/TILE_ROW_DIM<T>, 256, 3);
        }
        else { // K major mode (i.e., M x K for A matrix, N x K for B matrix)
            if constexpr (ST::swizzle_bytes == 128) // 128B swizzle mode
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 16 /* does not matter */, 1024, 1);
            else if constexpr (ST::swizzle_bytes == 64) // 64B swizzle mode
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 16 /* does not matter */, 512, 2);
            else // 32B swizzle mode
                base_desc = detail::matrix_descriptor_raw(&tile.data[0], 16 /* does not matter */, 256, 3);
        }
    }
    __device__ inline st_descriptor(const st_descriptor<ST, MN_major> &other) : base_desc(other.base_desc) {} // copy constructor
    template<int chunk_bytes=32>
    __device__ inline uint64_t chunk_descriptor(int chunk_idx) {
        // Return the n-th chunk along the K dimension, where chunk_bytes is the number of
        // operand bytes consumed per row by one tensor-core call. The default 32-byte chunk
        // covers the legacy MMA shapes; packed FP4 K96 uses 48 bytes (see
        // chunk_descriptor_k96), while SM107 packed FP4 K128 and FP8 K64 use 64 bytes.
        // For MN-major, advance chunk_bytes in the stride dimension; for K-major, advance
        // chunk_bytes in the leading dimension while respecting the swizzle-atom boundary.
#if defined(KITTENS_SM107)
        static_assert(chunk_bytes == 32 || chunk_bytes == 48 || chunk_bytes == 64, "SM107 chunk descriptors support 32-, 48-, or 64-byte chunks.");
#elif defined(KITTENS_SM103)
        static_assert(chunk_bytes == 32 || chunk_bytes == 48, "SM103 chunk descriptors support 32- or 48-byte chunks.");
#else
        static_assert(chunk_bytes == 32, "Chunk descriptors larger than 32 bytes require SM103 or SM107.");
#endif
        if constexpr (chunk_bytes == 48) {
#if defined(KITTENS_SM103) || defined(KITTENS_SM107)
            static_assert(!MN_major && std::is_same_v<T, fp4e2m1_2> && ST::swizzle_bytes == 128,
                          "48-byte chunks require K-major packed FP4 with 128B swizzling.");
#endif
            // K96 chunks that cross a 128B swizzle atom need a continuation box.
            const int start_byte = chunk_idx*48;
            const int edge_gap = 128 - start_byte%128;
            const int cont_byte = start_byte + (edge_gap < 48 ? edge_gap : 32);
            const uint64_t start_desc = base_desc + detail::matrix_descriptor_encode_addr(start_byte%128 + (start_byte/128)*(ST::rows/TILE_ROW_DIM<T>)*2048);
            const uint64_t cont_desc  = base_desc + detail::matrix_descriptor_encode_addr(cont_byte%128 + (cont_byte/128)*(ST::rows/TILE_ROW_DIM<T>)*2048);
            return (start_desc & ~(detail::DESCRIPTOR_ADDR_MASK << 16)) | ((cont_desc & detail::DESCRIPTOR_ADDR_MASK) << 16) | (1ull << 52);
        }
        else {
            if constexpr (chunk_bytes == 64) {
#ifdef KITTENS_SM107
                static_assert(!MN_major &&
                              (std::is_same_v<T, fp4e2m1_2> || std::is_same_v<T, fp8e4m3> || std::is_same_v<T, fp8e5m2>) &&
                              (ST::swizzle_bytes == 128 || ST::swizzle_bytes == 64),
                              "64-byte chunks require K-major packed FP4 or FP8 with 64B or 128B swizzling.");
#endif
            }
            constexpr int atom_stride = (ST::rows/TILE_ROW_DIM<T>)*ST::swizzle_bytes*16;
            if constexpr (MN_major) {
                return base_desc + detail::matrix_descriptor_encode_addr(chunk_idx*ST::swizzle_bytes*16);
            }
            else {
                constexpr int chunks_per_atom = chunk_bytes <= ST::swizzle_bytes ? ST::swizzle_bytes/chunk_bytes : 1;
                return base_desc + detail::matrix_descriptor_encode_addr((chunk_idx%chunks_per_atom)*chunk_bytes + (chunk_idx/chunks_per_atom)*atom_stride);
            }
        }
    }
#if defined(KITTENS_SM103) || defined(KITTENS_SM107)
    __device__ inline uint64_t chunk_descriptor_k96(int chunk_idx) {
        return chunk_descriptor<48>(chunk_idx);
    }
#endif
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
