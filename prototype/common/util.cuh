#pragma once

#include "../include/kittens.cuh"

namespace kittens {
namespace prototype {

template<int N> __device__ static inline int ring_advance(int ring, int distance=1) { return (ring + distance) % N; }
template<int N> __device__ static inline int ring_retreat(int ring, int distance=1) { return (ring + 16*N - distance) % N; }
template<int half> __device__ static inline bool get_phasebit(uint32_t bitfield, int ring_id) {
    return (bitfield & (1 << (half*16 + ring_id))) != 0;
}
template<int half> __device__ static inline void update_phasebit(uint32_t &bitfield, int ring_id) {
    bitfield ^= (1 << (half*16 + ring_id));
}

__device__ static inline int get_task_iter(int ti) {
    return ti*gridDim.x*gridDim.y*gridDim.z + blockIdx.z*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x;
}

}
}