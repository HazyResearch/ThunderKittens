#pragma once

#include "../include/kittens.cuh"

namespace kittens {
namespace prototype {

__device__ static inline int get_task_iter(int ti) {
    return ti*gridDim.x*gridDim.y*gridDim.z + blockIdx.z*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x;
}

}
}
