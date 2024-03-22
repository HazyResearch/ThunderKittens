#pragma once

#include <stdint.h>
#include <type_traits>
#include <concepts>
#include <memory>

/**
 * @file util.cuh
 * @brief Utility functions and constants for CUDA operations within the kittens namespace.
 */

namespace kittens {

/* ----------  GENERAL CONSTANTS FOR KITTENS  ---------- */

/**
 * @brief Tile dimension for matrix operations.
 */
constexpr int TILE_DIM{16};

/**
 * @brief Total size of a tile, computed as TILE_DIM squared.
 */
constexpr int TILE_SIZE{TILE_DIM*TILE_DIM};

/**
 * @brief Size of a warp.
 */
constexpr int WARP_SIZE{32};

/**
 * @brief Number of warps.
 */
constexpr int N_WARPS{8}; // default

/**
 * @brief Number of threads per block.
 */
constexpr int THREADS_PER_BLOCK{N_WARPS*WARP_SIZE};

/**
 * @brief Get the warp ID of the current thread.
 * @return Warp ID.
 */
__device__ __forceinline__ int warpid() { return threadIdx.x >> 5; }

/**
 * @brief Get the lane ID of the current thread within its warp.
 * @return Lane ID.
 */
__device__ __forceinline__ int laneid() { return threadIdx.x & 0x1f; }

#ifdef KITTENS_HOPPER
/**
 * @brief Maximum shared memory available on Hopper architecture.
 */
constexpr int MAX_SHARED_MEMORY = 227000;
#elif KITTENS_A100
/**
 * @brief Maximum shared memory available on A100 architecture.
 */
constexpr int MAX_SHARED_MEMORY = 164000;
#elif KITTENS_4090
/**
 * @brief Maximum shared memory available on 4090 architecture.
 */
constexpr int MAX_SHARED_MEMORY = 101000;
#endif

/* ----------  SHUFFLE UTILS  ---------- */

static constexpr uint32_t MASK_ALL = 0xFFFFFFFF;

/**
 * @brief Perform a shuffle down operation synchronously across a warp.
 * @param mask The bitmask to use for the shuffle operation.
 * @param f The value to be shuffled.
 * @param delta The offset for the shuffle.
 * @return The result of the shuffle operation.
 */
template<typename T>
__device__ static inline float2 packed_shfl_down_sync(uint32_t mask, const T &f, int delta) {
    return __shfl_down_sync(mask, f, delta);
}

/**
 * @brief Specialization of packed_shfl_down_sync for float2 data type.
 */
template<>
__device__ inline float2 packed_shfl_down_sync<float2>(uint32_t mask, const float2 &f, int delta) {
    float2 r;
    r.x = __shfl_down_sync(mask, f.x, delta);
    r.y = __shfl_down_sync(mask, f.y, delta);
    return r;
}

/**
 * @brief Perform a shuffle operation synchronously across a warp.
 * @param mask The bitmask to use for the shuffle operation.
 * @param f The value to be shuffled.
 * @param src The source lane for the shuffle.
 * @return The result of the shuffle operation.
 */
template<typename T>
__device__ static inline float2 packed_shfl_sync(uint32_t mask, const T &f, int src) {
    return __shfl_sync(mask, f, src);
}

/**
 * @brief Specialization of packed_shfl_sync for float2 data type.
 */
template<>
__device__ inline float2 packed_shfl_sync<float2>(uint32_t mask, const float2 &f, int src) {
    float2 r;
    r.x = __shfl_sync(mask, f.x, src);
    r.y = __shfl_sync(mask, f.y, src);
    return r;
}

/* ----------  SHARED MEMORY UTILS  ---------- */

/**
 * @brief Dummy struct to enforce alignment in shared memory.
 */
struct alignas(256) alignment_dummy { int dummy; };

/**
 * @brief Allocator for shared memory with alignment and templated allocation functions.
 */
struct shared_allocator {
    int *ptr;

    private:
        __device__ shared_allocator(int *_ptr): ptr(_ptr) {}

        // Recursive template to generate N-dimensional array type
        template<typename A, size_t... dims>
        struct variadic_array;
        template<typename A, size_t first_dim, size_t... rest_dims>
        struct variadic_array<A, first_dim, rest_dims...> {
            using type = typename variadic_array<A[first_dim], rest_dims...>::type;
        };
        template<typename A>
        struct variadic_array<A> {
            using type = A;
        };
        template<typename A, size_t... dims> using variadic_array_t = typename variadic_array<A, dims...>::type;

    public:
        /**
         * @brief Create a shared memory allocator.
         * @param _ptr Pointer to the start of shared memory.
         * @return A new shared_allocator instance.
         */
        __device__ inline static shared_allocator create_allocator(int *_ptr) {
            shared_allocator sa(_ptr); 
            return sa;
        }

        /**
         * @brief Create a shared memory allocator with thread matrix alignment.
         * @param _ptr Pointer to the start of shared memory.
         * @return A new shared_allocator instance with alignment.
         */
        __device__ inline static shared_allocator create_allocator_tma(int *_ptr) {
            shared_allocator sa(_ptr); 
            uint64_t p = reinterpret_cast<uint64_t>(sa.ptr);
            sa.ptr = (int*)(p + (alignment-(p%alignment)));
            return sa;
        }

        /**
         * @brief Allocate shared memory for a variadic array type.
         * @return Reference to the allocated array.
         */
        template<typename A, size_t... dims> 
        __device__ inline variadic_array_t<A, dims...>& allocate() {
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
            return *p;
        }

        /**
         * @brief Allocate shared memory for a 3D array.
         * @return Reference to the allocated 3D array.
         */
        template<typename A, size_t N, size_t M, size_t L> 
        __device__ inline A (&allocate())[N][M][L] {
            using at = A[N][M][L];
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
            return *p;
        }

        /**
         * @brief Allocate shared memory for a 4D array.
         * @return Reference to the allocated 4D array.
         */
        template<typename A, size_t N, size_t M, size_t L, size_t K> 
        __device__ inline A (&allocate())[N][M][L][K] {
            using at = A[N][M][L][K];
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
            return *p;
        }
    };
}

}
