/**
 * @file
 * @brief Functions for transferring data directly between tensor memory and register memory.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

__device__ inline static void tm_load_wait() {
   asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ inline static void tm_store_wait() {
   asm volatile("tcgen05.wait::st.sync.aligned;"); 
}

}