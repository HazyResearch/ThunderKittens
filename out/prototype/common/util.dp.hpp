#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../include/kittens.dp.hpp"

namespace kittens {
namespace prototype {

template<int N> static inline int ring_advance(int ring, int distance=1) { return (ring + distance) % N; }
template<int N> static inline int ring_retreat(int ring, int distance=1) { return (ring + 16*N - distance) % N; }
template<int half> static inline bool get_phasebit(uint32_t bitfield, int ring_id) {
    return (bitfield & (1 << (half*16 + ring_id))) != 0;
}
template<int half> static inline void update_phasebit(uint32_t &bitfield, int ring_id) {
    bitfield ^= (1 << (half*16 + ring_id));
}

static inline int get_task_iter(int ti) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return ti * item_ct1.get_group_range(2) * item_ct1.get_group_range(1) *
               item_ct1.get_group_range(0) +
           item_ct1.get_group(0) * item_ct1.get_group_range(1) *
               item_ct1.get_group_range(2) +
           item_ct1.get_group(1) * item_ct1.get_group_range(2) +
           item_ct1.get_group(2);
}

}
}