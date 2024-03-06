#pragma once

#include <stdint.h>
#include <type_traits>

namespace kittens {

/**
 * @file util.cuh
 * @brief Utility functions and concepts for general constants, boolean type utilities, shuffle operations, and shared memory allocation.
 */

/* ----------  GENERAL CONSTANTS FOR KITTENS  ---------- */

/**
 * @brief Tile dimension constant.
 */
constexpr int TILE_DIM{16};

/**
 * @brief Tile size constant calculated as TILE_DIM squared.
 */
constexpr int TILE_SIZE{TILE_DIM*TILE_DIM};

/**
 * @brief Warp size constant.
 */
constexpr int WARP_SIZE{32};

/**
 * @brief Number of warps constant.
 */
constexpr int N_WARPS{8}; // default

/**
 * @brief Threads per block constant calculated as N_WARPS multiplied by WARP_SIZE.
 */
constexpr int THREADS_PER_BLOCK{N_WARPS*WARP_SIZE};

/**
 * @brief Get the warp ID of the current thread.
 * @return The warp ID.
 */
__device__ __forceinline__ int warp_id() {return threadIdx.x >> 5; } 

/**
 * @brief Get the lane ID of the current thread within its warp.
 * @return The lane ID.
 */
__device__ __forceinline__ int laneid() {return threadIdx.x & 0x1f; }

/* ----------  BOOL TYPE UTILS  ---------- */

/**
 * @brief Structure to check if a type is a boolean type.
 * @tparam T The type to check.
 */
template<typename T>
struct is_bool_type : std::false_type {};

template<>
struct is_bool_type<std::true_type> : std::true_type {};

template<>
struct is_bool_type<std::false_type> : std::true_type {};

/**
 * @brief Utility structure to get the opposite boolean type.
 * @tparam B The boolean type to negate.
 */
template<typename B> struct not_type      { using type = std::true_type;  };

/**
 * @brief Specialization of not_type for std::true_type.
 */
template<> struct not_type<std::true_type> { using type = std::false_type; };

/* ----------  SHUFFLE UTILS  ---------- */

/**
 * @brief Mask constant for all active threads.
 */
static constexpr uint32_t MASK_ALL = 0xFFFFFFFF;

/**
 * @brief Perform a shuffle down operation synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask The mask of active threads.
 * @param f The value to be shuffled.
 * @param delta The number of positions to shuffle down.
 * @return The result of the shuffle operation.
 */
template<typename T>
__device__ static inline T packed_shfl_down_sync(uint32_t mask, const T &f, int delta) {
    static_assert(std::is_arithmetic<T>::value, "Shuffle operations are only supported for arithmetic types.");
    return __shfl_down_sync(mask, f, delta);
}

/**
 * @brief Perform a shuffle operation synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask The mask of active threads.
 * @param f The value to be shuffled.
 * @param src The source lane from which to shuffle.
 * @return The result of the shuffle operation.
 */
template<typename T>
__device__ static inline T packed_shfl_sync(uint32_t mask, const T &f, int src) {
    static_assert(std::is_arithmetic<T>::value, "Shuffle operations are only supported for arithmetic types.");
    return __shfl_sync(mask, f, src);
}

/* ----------  SHARED MEMORY UTILS  ---------- */

/**
 * @brief Dummy structure for alignment purposes.
 */
struct alignas(128) alignment_dummy { int dummy; };

/**
 * @brief Allocator for shared memory.
 */
struct shared_allocator {

    int *ptr; ///< Pointer to the current position in shared memory.

    /**
     * @brief Construct a new shared allocator.
     * @param _ptr Pointer to the start of the shared memory.
     */
    __device__ shared_allocator(int * _ptr): ptr(_ptr) {}

    /**
     * @brief Allocate shared memory for a single object of type A.
     * @tparam A The type of the object to allocate.
     * @return Reference to the allocated object.
     */
    template<typename A> 
    __device__ inline A& allocate() {
        A*p = reinterpret_cast<A*>(ptr);
        ptr += sizeof(A)/sizeof(int);
        return *p;
    }

    /**
     * @brief Allocate shared memory for an array of objects of type A.
     * @tparam A The type of the objects to allocate.
     * @tparam N The number of objects in the array.
     * @return Reference to the allocated array.
     */
    template<typename A, size_t N> 
    __device__ inline A (&allocate())[N] {
        using at = A[N];
        at*p = reinterpret_cast<at*>(ptr);
        ptr += sizeof(at)/sizeof(int);
        return *p;
    }

    /**
     * @brief Allocate shared memory for a two-dimensional array of objects of type A.
     * @tparam A The type of the objects to allocate.
     * @tparam N The number of objects in the first dimension.
     * @tparam M The number of objects in the second dimension.
     * @return Reference to the allocated two-dimensional array.
     */
    template<typename A, size_t N, size_t M> 
    __device__ inline A (&allocate())[N][M] {
        using at = A[N][M];
        at*p = reinterpret_cast<at*>(ptr);
        ptr += sizeof(at)/sizeof(int);
        return *p;
    }
};

}
