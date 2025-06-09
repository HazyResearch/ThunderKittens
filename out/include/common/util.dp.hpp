/**
 * @file
 * @brief General utilities for ThunderKittens.
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdint.h>
#include <type_traits>
#include <concepts>
#include <memory>

// CUDA driver API
/*
DPCT1001:463: The statement could not be removed.
*/
/*
DPCT1000:464: Error handling if-stmt was detected but could not be rewritten.
*/
/*
DPCT1009:466: SYCL reports errors using exceptions and does not use error codes.
Please replace the "get_error_string_dummy(...)" with a real error-handling
function.
*/
#define CUCHECK(cmd) do {                                                      \
        int err = cmd;                                                         \
        if (err != 0) {                                                        \
            const char *errStr;                                                \
            errStr = dpct::get_error_string_dummy(err);                        \
            fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n", __FILE__,       \
                    __LINE__, errStr);                                         \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// CUDA runtime API
/*
DPCT1009:468: SYCL reports errors using exceptions and does not use error codes.
Please replace the "get_error_string_dummy(...)" with a real error-handling
function.
*/
#define CUDACHECK(cmd) do {                                                    \
        dpct::err0 err = cmd;                                                  \
                                                                               \
    } while (0)

/**
 * @namespace kittens
 *
 * @brief The main namespace of ThunderKittens.
 */
namespace kittens {

/* ----------  GENERAL CONSTANTS FOR KITTENS  ---------- */

/**
 * @brief Tile dimension constant.
 */
template<typename T> constexpr int TILE_COL_DIM = sizeof(T) == 1 ? 32 : 16;
template<typename T> constexpr int TILE_ROW_DIM = 16;
/**
 * @brief Tile num elements constant calculated as TILE_DIM squared.
 */
template<typename T> constexpr int TILE_ELEMENTS{TILE_COL_DIM<T>*TILE_ROW_DIM<T>};
/**
 * @brief Constant representing number of threads in a warp.
 */
constexpr int WARP_THREADS{32};
/**
 * @brief Constant representing number of threads in a warpgroup of four warps.
 */
constexpr int WARPGROUP_THREADS{128};
/**

 * @brief Constant representing number of warps in a warpgroup of four warps.
 */
constexpr int WARPGROUP_WARPS{4};
/**

 * @brief Get the warp ID of the current thread.
 * @return The warp ID.
 */
__dpct_inline__ int warpid() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
               2) >>
               5;
}
/**
 * @brief Get the warpgroup ID of the current thread.
 * @return The warpgroup ID.
 */
__dpct_inline__ int warpgroupid() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
               2) >>
               7;
}
/**
 * @brief Get the lane ID of the current thread within its warp.
 * @return The lane ID.
 */
__dpct_inline__ int laneid() {
    return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
               0x1f;
}

#if defined(KITTENS_HOPPER)
constexpr int MAX_SHARED_MEMORY = 227000;
#elif defined(KITTENS_A100)
constexpr int MAX_SHARED_MEMORY = 164000;
#elif defined(KITTENS_4090)
constexpr int MAX_SHARED_MEMORY = 100000;
#endif

struct transpose {
    static constexpr int N = 0; // not transposed
    static constexpr int T = 1; // transposed
};
struct axis {
    static constexpr int ROW = 0; // row axis of a tile
    static constexpr int COL = 1; // column axis of a tile
};

/* ----------  TYPE HELPERS  ---------- */

/**
 * @namespace ducks
 *
 * @brief ThunderKittens' namespace for template metaprogramming..
 * 
 * This includes primarily dummy types and concept wrappers, along
 * with a few additional utilities.
 */
namespace ducks {

/**
 * @brief A type representing an empty default for a template.
 */
struct default_type {};

// This macro can't be done as a template, so it doesn't really have a location in kittens.
#define typeof(A) typename std::remove_const<typename std::remove_reference<decltype(A)>::type>::type

}

/* ----------  SHUFFLE UTILS  ---------- */

/**
 * @brief Mask constant for all active threads in a warp.
 */
static constexpr uint32_t MASK_ALL = 0xFFFFFFFF;

/**
 * @brief Perform a shuffle down operation on a packed type synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask[in] The mask of active threads.
 * @param f[in] The value to be shuffled.
 * @param delta[in] The number of positions to shuffle down.
 * @return The result of the shuffle operation.
 */
template<typename T>
static inline T packed_shfl_down_sync(uint32_t mask, const T &f, int delta) {
    /*
    DPCT1108:368: '__shfl_down_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    return dpct::experimental::shift_sub_group_left(
        mask, sycl::ext::oneapi::this_work_item::get_sub_group(), f, delta);
}
template <>
inline sycl::float2 packed_shfl_down_sync<sycl::float2>(uint32_t mask,
                                                        const sycl::float2 &f,
                                                        int delta) {
    sycl::float2 r;
    /*
    DPCT1108:0: '__shfl_down_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    r.x() = dpct::experimental::shift_sub_group_left(
        mask, sycl::ext::oneapi::this_work_item::get_sub_group(), f.x(), delta);
    /*
    DPCT1108:1: '__shfl_down_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    r.y() = dpct::experimental::shift_sub_group_left(
        mask, sycl::ext::oneapi::this_work_item::get_sub_group(), f.y(), delta);
    return r;
}
/**
 * @brief Perform a packed shuffle operation synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask[in] The mask of active threads.
 * @param f[in] The value to be shuffled.
 * @param src[in] The source lane from which to shuffle.
 * @return The result of the shuffle operation.
 */
template<typename T>
static inline T packed_shfl_sync(uint32_t mask, const T &f, int src) {
    /*
    DPCT1108:343: '__shfl_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    return dpct::experimental::select_from_sub_group(
        mask, sycl::ext::oneapi::this_work_item::get_sub_group(), f, src);
}
template <>
inline sycl::float2
packed_shfl_sync<sycl::float2>(uint32_t mask, const sycl::float2 &f, int src) {
    sycl::float2 r;
    /*
    DPCT1108:2: '__shfl_sync' was migrated with the experimental feature masked
    sub_group function which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    r.x() = dpct::experimental::select_from_sub_group(
        mask, sycl::ext::oneapi::this_work_item::get_sub_group(), f.x(), src);
    /*
    DPCT1108:3: '__shfl_sync' was migrated with the experimental feature masked
    sub_group function which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    r.y() = dpct::experimental::select_from_sub_group(
        mask, sycl::ext::oneapi::this_work_item::get_sub_group(), f.y(), src);
    return r;
}

/* ----------  SHARED MEMORY UTILS  ---------- */

// namespace ducks {
// namespace sb {
// struct identifier {};
// }
// }

// template<typename Args...>
// struct sb {
//     using identifier = ducks::sb::identifier;
//     Args... args;
// };

// namespace ducks {
// namespace sb {
// template<typename T> concept all = requires {
//     typename T::identifier;
// } && std::is_same_v<T::identifier, identifier>;
// }
// }

// Joyously stolen from https://github.com/NVIDIA/cutlass/blob/5c447dd84f8ae0e1d48ff9a2eae26ce8c4958101/include/cute/container/alignment.hpp#L51
#if defined(SYCL_LANGUAGE_VERSION)
#define KITTENS_ALIGN_AS(n) __dpct_align__(n)
#else
#define KITTENS_ALIGN_AS(n) alignas(n)
#endif

#ifdef KITTENS_HOPPER
#define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(128)
#else
#define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(16)
#endif

/**
 * @brief Dummy structure for alignment purposes. Needed for WGMMA and TMA calls.
 */
struct KITTENS_DEFAULT_ALIGN alignment_dummy { int dummy; };
/**
 * @brief Very simple allocator for dynamic shared memory. Advances pointer and tracks alignments.
 * @tparam default_alignment The default alignment this allocator will enforce. If <=0 (default -1) it will not align.
 */
#ifdef KITTENS_HOPPER
template<int default_alignment=1024> 
#else
template<int default_alignment=16> 
#endif
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
        template<typename A, size_t... dims> 
        using variadic_array_t = typename variadic_array<A, dims...>::type;

        template<int alignment>
        inline void align_ptr() {
            if constexpr (alignment > 0) {
                uint64_t p = reinterpret_cast<uint64_t>(ptr);
                if(p % alignment != 0) {
                    ptr = (int*)(p + (alignment-(p%alignment)));
                }
            }
        }

    public:
        /**
        * @brief Construct a new shared allocator using a pointer to extern shared memory.
        * @param[in] _ptr Pointer to the start of the extern shared memory.
        */
        shared_allocator(int *_ptr): ptr(_ptr) {}
        /**
        * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
        * @tparam A The type of the object to allocate.
        * @tparam dims... A list of dimensions for the N-dimensional array.
        * @return Reference to the allocated object.
        */
        template<typename A, size_t... dims> 
        inline variadic_array_t<A, dims...>& allocate() {
            // static_assert(sizeof(A) % default_alignment == 0, "Type is not aligned properly for array allocation");
            align_ptr<default_alignment>();
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
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
        inline variadic_array_t<A, dims...>& allocate() {
            // static_assert(sizeof(A) % alignment == 0, "Type is not aligned properly for array allocation");
            align_ptr<alignment>();
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
            return *p;
        }
};
#ifdef KITTENS_HOPPER
/**
 * @brief A wrapper for an allocator that enforces sufficient alignment to be used for TMA loads and stores.
 */
using tma_allocator = shared_allocator<1024>;
using tma_swizzle_allocator = tma_allocator; // swizzled TMA modes require up to 1024 byte alignments :/
#endif

} // namespace kittens