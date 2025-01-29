/**
 * @file
 * @brief The ThunderKittens tensor memory struct.
 */

#pragma once

#include "../../common/common.cuh"

/* ----------  MAIN TMEM STRUCT  ---------- */

// these are helper structs for type inference
namespace kittens {
namespace ducks {
/**
 * @namespace tmem
 * 
 * @brief The namespace where concepts and abstract types for shared tiles live.
 */
namespace tmem {
/**
 * @brief A dummy type used to identify tensor memory.
 */
struct identifier {};
/**
* @brief Concept for all tmem tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as tmem::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::tmem::identifier
} // namespace tmem
} // namespace ducks

/**
 * @brief Shared memory tile structure for various data types and layouts.
 *
 * @tparam T The data type of the elements in the tile. Not packed!
 * @tparam _rows The height of the tile.
 * @tparam _cols The width of the tile.
 */
template<typename _T, int _rows, int _cols>
struct tmem {
    using identifier = ducks::tmem::identifier; ///< Type identifier for shared memory tile.
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.

    static constexpr int rows    = _rows;
    static constexpr int cols    = _cols;
    static constexpr int height  = rows / kittens::TILE_ROW_DIM<T>;
    static constexpr int width   = cols / kittens::TILE_COL_DIM<T>;

    uint32_t addr;

    __device__ inline tmem(uint32_t addr) : addr(addr) {}

    template<ducks::tmem::all TM> __device__ inline TM subtile(int row_offset, int col_offset) const {
        return TM(addr + (row_offset << 16) + col_offset/(4/(uint32_t)sizeof(T))); // in units of the tile's data type.
    }
    template<int transpose> __device__ inline uint32_t chunk_addr(int chunk) const {
        if constexpr (transpose) {
            if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, half>) {
                return addr + ((16 * chunk) << 16);
            }
            else {
                static_assert(sizeof(T) == 999, "Currently unsupported type for input to an mma.");
            }
        }
        else {
            if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, half>) {
                return addr + (16 * chunk / (4/(uint32_t)sizeof(T)));
            }
            else {
                static_assert(sizeof(T) == 999, "Currently unsupported type for input to an mma.");
            }
        }
    } 

};

template<int nblocks> constexpr int num_tmem_cols = ((512/nblocks) / 32) * 32;
template<int nblocks=1, int ncta=1> __device__ auto allocate_tmem() {
    __shared__ uint32_t addr;
    constexpr int cols = num_tmem_cols<nblocks>;
    static_assert(cols>0 && cols%32==0, "cols must be a multiple of 32");
    if constexpr (ncta == 1) {
        if(warpid() == 0) {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32  [%0], %1;\n"
            ::  "l"((uint64_t)&addr), "n"(cols)
            );
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
        }
    }
    else {
        if(warpid() == 0) {
            asm volatile(
                "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32  [%0], %1;\n"
            ::  "l"((uint64_t)&addr), "n"(cols)
            );
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n");
        }
    }
    asm volatile("tcgen05.fence::before_thread_sync;\n");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;\n");
    return tmem<float, 128, cols>(addr);
};

} // namespace kittens