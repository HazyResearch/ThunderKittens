#pragma once

#include "../include/kittens.cuh"

namespace kittens {
namespace prototype {

template<typename T> constexpr int num_threads = T::NUM_CONSUMER_WARPS * 32 + 128;
template<typename T> constexpr int num_warps = T::NUM_CONSUMER_WARPS + 4;
template<typename T> constexpr int num_consumer_warpgroups = T::NUM_CONSUMER_WARPS / 4;

}
}