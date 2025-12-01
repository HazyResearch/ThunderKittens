/**
 * @file
 * @brief General utilities for ThunderKittens.
 */

#pragma once

#include <concepts>
#include <iostream>
#include <memory>
#include <stdint.h>
#include <type_traits>

// For checking CUDA driver API calls
#define CUCHECK(cmd) do {                                     \
    CUresult err = cmd;                                       \
    if (err != CUDA_SUCCESS) {                                \
        const char *errStr;                                   \
        cuGetErrorString(err, &errStr);                       \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, errStr);                      \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// For checking CUDA runtime API calls
#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// Convenience utility
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T> void check(
    T err, char const* const func, char const* const file, int const line
) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

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
constexpr int BASE_TILE_DIM = 16;
template<typename T> constexpr int TILE_COL_DIM = sizeof(T) == 1 ? BASE_TILE_DIM * 2 : BASE_TILE_DIM;
template<typename T> constexpr int TILE_ROW_DIM = BASE_TILE_DIM;
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
__device__ __forceinline__ int warpid() { return threadIdx.x >> 5; } 
/**
 * @brief Get the warpgroup ID of the current thread.
 * @return The warpgroup ID.
 */
__device__ __forceinline__ int warpgroupid() { return threadIdx.x >> 7; } 
/**
 * @brief Get the lane ID of the current thread within its warp.
 * @return The lane ID.
 */
__device__ __forceinline__ int laneid() { return threadIdx.x & 0x1f; }

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
constexpr int MAX_SHARED_MEMORY = 227 * 1024;
#elif defined(KITTENS_AMPERE)
constexpr int MAX_SHARED_MEMORY = 164 * 1024;
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
__device__ static inline T packed_shfl_down_sync(uint32_t mask, const T &f, int delta) {
    return __shfl_down_sync(mask, f, delta);
}
template<>
__device__ inline float2 packed_shfl_down_sync<float2>(uint32_t mask, const float2 &f, int delta) {
    float2 r;
    r.x = __shfl_down_sync(mask, f.x, delta);
    r.y = __shfl_down_sync(mask, f.y, delta);
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
__device__ static inline T packed_shfl_sync(uint32_t mask, const T &f, int src) {
    return __shfl_sync(mask, f, src);
}
template<>
__device__ inline float2 packed_shfl_sync<float2>(uint32_t mask, const float2 &f, int src) {
    float2 r;
    r.x = __shfl_sync(mask, f.x, src);
    r.y = __shfl_sync(mask, f.y, src);
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
#if defined(__CUDACC__)
#define KITTENS_ALIGN_AS(n) __align__(n)
#else
#define KITTENS_ALIGN_AS(n) alignas(n)
#endif

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(128)
#else
#define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(16)
#endif

/**
 * @brief Perform a ceiling division operation.
 * @tparam T The type of the value to be divided.
 * @param a[in] numerator.
 * @param b[in] denominator.
 * @return The result of the ceiling division.
 */
template <typename T>
__host__ __device__ constexpr T cdiv(T a, T b) {
    return (a + b - 1) / b;
}

/**
 * @brief Dummy structure for alignment purposes. Needed for WGMMA and TMA calls.
 */
struct KITTENS_DEFAULT_ALIGN alignment_dummy { int dummy; };
/**
 * @brief Very simple allocator for dynamic shared memory. Advances pointer and tracks alignments.
 * @tparam default_alignment The default alignment this allocator will enforce. If <=0 (default -1) it will not align.
 */
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
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
    __device__ inline void align_ptr() {
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
    __device__ shared_allocator(int *_ptr): ptr(_ptr) {}
    /**
    * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
    * @tparam A The type of the object to allocate.
    * @tparam dims... A list of dimensions for the N-dimensional array.
    * @return Reference to the allocated object.
    */
    template<typename A, size_t... dims> 
    __device__ inline variadic_array_t<A, dims...>& allocate() {
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
    __device__ inline variadic_array_t<A, dims...>& allocate() {
        // static_assert(sizeof(A) % alignment == 0, "Type is not aligned properly for array allocation");
        align_ptr<alignment>();
        using at = variadic_array_t<A, dims...>;
        at*p = reinterpret_cast<at*>(ptr);
        ptr += sizeof(at)/sizeof(int);
        return *p;
    }
};

#if (defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL))
/**
 * @brief A wrapper for an allocator that enforces sufficient alignment to be used for TMA loads and stores.
 */
using tma_allocator = shared_allocator<1024>;
using tma_swizzle_allocator = tma_allocator; // swizzled TMA modes require up to 1024 byte alignments :/

/* Get cluster ID */
__device__ static inline int3 clusterIdx() {
    int3 cluster_idx;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(cluster_idx.x));
    asm volatile("mov.u32 %0, %clusterid.y;\n" : "=r"(cluster_idx.y));
    asm volatile("mov.u32 %0, %clusterid.z;\n" : "=r"(cluster_idx.z));
    return cluster_idx;
}

/* Get CTA ID within a cluster */
__device__ static inline int cluster_ctarank() {
    uint32_t ctarank;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(ctarank));
    return ctarank;
}
#endif

template<int half> 
__device__ static inline bool get_phasebit(uint32_t bitfield, int ring_id) {
    return (bitfield & (1 << (half*16 + ring_id))) != 0;
}

template<int half> 
__device__ static inline void update_phasebit(uint32_t &bitfield, int ring_id) {
    bitfield ^= (1 << (half*16 + ring_id));
}

template<int N> __device__ static inline int ring_advance(int ring, int distance=1) { return (ring + distance) % N; }
template<int N> __device__ static inline int ring_retreat(int ring, int distance=1) { return (ring + 16*N - distance) % N; }

} // namespace kittens
