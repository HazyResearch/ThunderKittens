#pragma once

#include <stdint.h>
#include <type_traits>
#include <concepts>
#include <memory>

namespace kittens {

/* ----------  GENERAL CONSTANTS FOR KITTENS  ---------- */

constexpr int TILE_DIM{16};
constexpr int TILE_SIZE{TILE_DIM*TILE_DIM};
constexpr int WARP_SIZE{32};
__device__ __forceinline__ int warpid() { return threadIdx.x >> 5; } 
__device__ __forceinline__ int laneid() { return threadIdx.x & 0x1f; }

#ifdef KITTENS_HOPPER
constexpr int MAX_SHARED_MEMORY = 227000;
#elif KITTENS_A100
constexpr int MAX_SHARED_MEMORY = 164000;
#elif KITTENS_4090
constexpr int MAX_SHARED_MEMORY = 101000;
#endif

/* ----------  TYPE HELPERS  ---------- */

namespace ducks {

// a type representing an empty default for a template.
struct default_type {};

#define typeof(A) typename std::remove_const<typename std::remove_reference<decltype(A)>::type>::type

}

/* ----------  SHUFFLE UTILS  ---------- */

static constexpr uint32_t MASK_ALL = 0xFFFFFFFF;

template<typename T>
__device__ static inline float2 packed_shfl_down_sync(uint32_t mask, const T &f, int delta) {
    return __shfl_down_sync(mask, f, delta);
}
template<>
__device__ inline float2 packed_shfl_down_sync<float2>(uint32_t mask, const float2 &f, int delta) {
    float2 r;
    r.x = __shfl_down_sync(mask, f.x, delta);
    r.y = __shfl_down_sync(mask, f.y, delta);
    return r;
}
template<typename T>
__device__ static inline float2 packed_shfl_sync(uint32_t mask, const T &f, int src) {
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

struct alignas(256) alignment_dummy { int dummy; };
struct shared_allocator {
    int *ptr;

    private:
        __device__ shared_allocator(int *_ptr): ptr(_ptr) {}

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

    public:
        __device__ inline static shared_allocator create_allocator(int *_ptr) {
            shared_allocator sa(_ptr); 
            return sa;
        }
        template<int alignment=128>
        __device__ inline static shared_allocator create_allocator_tma(int *_ptr) {
            shared_allocator sa(_ptr); 
            uint64_t p = reinterpret_cast<uint64_t>(sa.ptr);
            sa.ptr = (int*)(p + (alignment-(p%alignment)));
            return sa;
        }
        template<typename A, size_t... dims> 
        __device__ inline variadic_array_t<A, dims...>& allocate() {
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
            return *p;
        }
};

}