# pragma once

namespace kittens {
/*
This is meant to be used with a `using block = kittens::block<NUM_WORKERS>;` at the start of every kernel.
*/
template<int N_WARPS>
struct block {
static constexpr int BLOCK_SIZE = N_WARPS * 32; // This alias produces nice parallelism.

#include "memory/memory.cuh"

};
}