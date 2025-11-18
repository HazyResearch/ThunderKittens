/**
 * @file
 * @brief An aggregate header of all device (multi-GPU) operations defined by ThunderKittens
 *
 * WARNING: This API is in an experimental stage.
 */

#pragma once

#include "../../types/types.cuh"

namespace kittens {

template<int _NUM_DEVICES>
struct device {

static_assert(_NUM_DEVICES >= 0 && _NUM_DEVICES <= 72, "Invalid number of devices");
static constexpr int NUM_DEVICES = _NUM_DEVICES;

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)

using barrier_t = pgl<gl<int, -1, -1, -1, -1>, NUM_DEVICES, true>;

__device__ static inline void signal(const barrier_t &barrier, const coord<ducks::default_type> &idx, const int dst_dev_idx, const int val) {
    asm volatile("{red.release.sys.global.add.s32 [%0], %1;}" :: "l"(&barrier[dst_dev_idx][idx]), "r"(val) : "memory");
}

__device__ static inline void signal_all(const barrier_t &barrier, const coord<ducks::default_type> &idx, const int val) {
    asm volatile("{multimem.red.release.sys.global.add.s32 [%0], %1;}" :: "l"(barrier.mc_ptr_at(idx)), "r"(val) : "memory");
}

__device__ static inline void wait(const barrier_t &barrier, const coord<ducks::default_type> &idx, const int dev_idx, const int expected) {
    int val;
    do {
        asm volatile("{ld.relaxed.sys.global.s32 %0, [%1];}" : "=r"(val) : "l"(&barrier[dev_idx][idx]) : "memory");
    } while (val != expected);
}

__device__ static inline void barrier(const barrier_t &barrier, const coord<ducks::default_type> &idx, const int dev_idx) {
    signal_all(barrier, idx, 1);
    wait(barrier, idx, dev_idx, NUM_DEVICES);
    asm volatile("{red.release.sys.global.add.s32 [%0], %1;}" :: "l"(&barrier[dev_idx][idx]), "r"(-NUM_DEVICES) : "memory");
}

#endif

};

} // namespace kittens
