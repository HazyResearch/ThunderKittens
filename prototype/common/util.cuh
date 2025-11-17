#pragma once

#include "../include/kittens.cuh"

namespace kittens {
namespace prototype {

template<int N> __device__ static inline int ring_advance(int ring, int distance=1) { return (ring + distance) % N; }
template<int N> __device__ static inline int ring_retreat(int ring, int distance=1) { return (ring + 16*N - distance) % N; }
__device__ static inline int get_task_iter(int ti) {
    return ti*gridDim.x*gridDim.y*gridDim.z + blockIdx.z*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x;
}

}
}