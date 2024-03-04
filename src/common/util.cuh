#pragma once

#include <stdint.h>
#include <type_traits>
#include <memory>

namespace kittens {


/* ----------  GENERAL CONSTANTS FOR KITTENS  ---------- */

constexpr int TILE_DIM{16};
constexpr int TILE_SIZE{TILE_DIM*TILE_DIM};
constexpr int WARP_SIZE{32};
constexpr int N_WARPS{8}; // default
constexpr int THREADS_PER_BLOCK{N_WARPS*WARP_SIZE};
__device__ __forceinline__ int warp_id() {return threadIdx.x >> 5; } 
__device__ __forceinline__ int laneid() {return threadIdx.x & 0x1f; }

/* ----------  BOOL TYPE UTILS  ---------- */

template<typename T>
concept bool_type = std::is_same_v<T, std::true_type> || std::is_same_v<T, std::false_type>;

// turns out to be a useful alias for swapping layouts
template<bool_type B> struct not_type      { using type = std::true_type;  };
template<> struct not_type<std::true_type> { using type = std::false_type; };

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

struct alignas(128) alignment_dummy { int dummy; };
struct shared_allocator {

    int *ptr;

    // I need to do this to avoid misaligned addresses
    // in TMA
    // __device__ shared_allocator(int * _ptr): ptr(_ptr) {}
    __device__ shared_allocator(int * _ptr): ptr(_ptr) {
        uint64_t p = reinterpret_cast<uint64_t>(ptr);
        ptr = std::assume_aligned<128>((int*)(p + (128-(p%128))));
    }

    template<typename A> 
    __device__ inline A& allocate() {
        A*p = reinterpret_cast<A*>(ptr);
        ptr += sizeof(A)/sizeof(int);
        return *p;
    }
    template<typename A, size_t N> 
    __device__ inline A (&allocate())[N] {
        using at = A[N];
        at*p = reinterpret_cast<at*>(ptr);
        ptr += sizeof(at)/sizeof(int);
        return *p;
    }
    template<typename A, size_t N, size_t M> 
    __device__ inline A (&allocate())[N][M] {
        using at = A[N][M];
        at*p = reinterpret_cast<at*>(ptr);
        ptr += sizeof(at)/sizeof(int);
        return *p;
    }
};

}