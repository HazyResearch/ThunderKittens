/**
 * @file
 * @brief Allocation utilities for ThunderKittens.
 */

#pragma once

#include "common.cuh"
#include "../types/types.cuh"

namespace kittens {

/* ----------  SHARED MEMORY UTILS  ---------- */

// sizeof() can be unreliable when working with references to objects
// plus, template magic allows arrays of these objects to be copied, too.
namespace detail {
template<typename T, size_t... dims> struct size_info;
template<ducks::st::all ST> struct size_info<ST> {
    static constexpr size_t elements = ST::num_elements;
    static constexpr size_t bytes    = ST::num_elements * sizeof(typename ST::dtype);
};
template<ducks::sv::all SV> struct size_info<SV> {
    static constexpr size_t elements = SV::length;
    static constexpr size_t bytes    = SV::length * sizeof(typename SV::dtype);
};
template<typename T, size_t dim, size_t... rest_dims> struct size_info<T, dim, rest_dims...> {
    static constexpr size_t elements = dim*size_info<T, rest_dims...>::elements;
    static constexpr size_t bytes    = dim*size_info<T, rest_dims...>::bytes;
};
}
template<typename T, size_t... dims> constexpr size_t size_elements = detail::size_info<T, dims...>::elements;
template<typename T, size_t... dims> constexpr size_t size_bytes    = detail::size_info<T, dims...>::bytes;

/**
 * @brief Dummy structure for alignment purposes. Needed for WGMMA and TMA calls.
 */
struct KITTENS_DEFAULT_ALIGN alignment_dummy { int dummy; };
/**
 * @brief Very simple allocator for dynamic shared memory. Advances pointer and tracks alignments.
 * @tparam default_alignment The default alignment this allocator will enforce. If <=0 (default -1) it will not align.
 */
template<int default_alignment=-1> // 
struct shared_allocator {
    int *ptr;

    private:
        // Recursive template to generate N-dimensional array type
        template<typename A, size_t... dims>
        struct variadic_array;
        template<typename A, size_t first_dim, size_t... rest_dims>
        struct variadic_array<A, first_dim, rest_dims...> {
            using type = typename variadic_array<A, rest_dims...>::type[first_dim];
        };
        template<typename A>
        struct variadic_array<A> {
            using type = A;
        };
        template<typename A, size_t... dims> using variadic_array_t = typename variadic_array<A, dims...>::type;

        template<int alignment>
        __device__ inline void align_ptr() {
            if constexpr (alignment > 0) {
                uint64_t p = reinterpret_cast<uint64_t>(ptr);
                if(p%alignment != 0) ptr = (int*)(p + (alignment-(p%alignment)));
            }
        }

    public:
        /**
        * @brief Construct a new shared allocator using a pointer to extern shared memory.
        * @param[in] _ptr Pointer to the start of the extern shared memory.
        */
        __device__ shared_allocator(int *_ptr): ptr(_ptr) {}
        /**
        * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
        * @tparam A The type of the object to allocate.
        * @tparam dims... A list of dimensions for the N-dimensional array.
        * @return Reference to the allocated object.
        */
        template<typename A, size_t... dims> 
        __device__ inline variadic_array_t<A, dims...>& allocate() {
            align_ptr<default_alignment>();
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += size_bytes<A, dims...>/sizeof(int);
            return *p;
        }
        /**
        * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
        * @tparam alignment An alignment to enforce for this particular object.
        * @tparam A The type of the object to allocate.
        * @tparam dims... A list of dimensions for the N-dimensional array.
        * @return Reference to the allocated object.
        */
        template<int alignment, typename A, size_t... dims> 
        __device__ inline variadic_array_t<A, dims...>& allocate() {
            align_ptr<alignment>();
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += size_bytes<A, dims...>/sizeof(int);
            return *p;
        }
};
#ifdef KITTENS_HOPPER
/**
 * @brief A wrapper for an allocator that enforces sufficient alignment to be used for TMA loads and stores.
 */
using tma_allocator = shared_allocator<128>;
using tma_swizzle_allocator = shared_allocator<1024>; // swizzled TMA modes require up to 1024 byte alignments :/
#endif

} // namespace kittens