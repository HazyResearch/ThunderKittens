/**
 * @file
 * @brief An aggregate header file for all the tensor types defined by ThunderKittens.
 */

#pragma once

#include "tt.cuh"

// A thin wrapper that allows for certain compile-time checks to be performed when allocating tensor memory.
namespace kittens {
namespace ducks {
/**
 * @namespace tensor_allocator
 * 
 * @brief The namespace where concepts and abstract types for tensor memory allocation live.
 */
namespace tensor_allocator {
/**
 * @brief A dummy type used to identify tensor memory.
 */
struct identifier {};
/**
* @brief Concept for all tensor_allocator types.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as tensor_allocator::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::tt::identifier
} // namespace tensor_allocator
} // namespace ducks

template<int _cols> struct tensor_allocator {
    using identifier = ducks::tensor_allocator::identifier;
    static constexpr int cols = _cols;
    uint32_t addr;
    template<ducks::tt::all TT, int col_offset> __device__ inline void check_bounds() {
        static_assert(col_offset >= 0 && col_offset + TT::cols <= cols, "Tile allocation extends out of bounds of the tensor allocator!");
    }
    __device__ inline tensor_allocator(uint32_t _addr) : addr(_addr) {}
    __device__ inline uint32_t get_addr(int superlane, int col_offset) const { return addr + ((superlane*16) << 16) + col_offset; }
    template<ducks::tt::half TT> __device__ inline auto allocate(int superlane, int col_offset) {
#ifndef NDEBUG
        if(col_offset + TT::cols > cols) {
            printf("Tile allocation extends out of bounds of the tensor allocator! col_offset: %d, TT::cols: %d, allocator cols: %d\n", col_offset, TT::cols, cols);
            asm volatile("trap;");
        }
        if(superlane < 0 || superlane > 1) {
            printf("Superlane must be 0 or 1! superlane: %d\n", superlane);
            asm volatile("trap;");
        }
#endif
        return TT(get_addr(superlane, col_offset));
    }
    template<ducks::tt::full TT> __device__ inline auto allocate(int col_offset) {
#ifndef NDEBUG
        if(col_offset + TT::cols > cols) {
            printf("Tile allocation extends out of bounds of the tensor allocator! col_offset: %d, TT::cols: %d, allocator cols: %d\n", col_offset, TT::cols, cols);
            asm volatile("trap;");
        }
#endif
        return TT(get_addr(0, col_offset));
    }
};
template<int nblocks> constexpr int num_tensor_memory_cols = ((512/nblocks) / 32) * 32;
template<int nblocks=1, int ncta=1> __device__ auto allocate_tensor_memory() {
    __shared__ uint32_t addr;
    constexpr int cols = num_tensor_memory_cols<nblocks>;
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
    asm volatile("bar.sync 0;\n");
    asm volatile("tcgen05.fence::after_thread_sync;\n");
    return tensor_allocator<cols>(addr);
};

} // namespace kittens