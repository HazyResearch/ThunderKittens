#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
namespace warpgroup {

template<int width, int trans_b>
struct wgmma_base {
    __device__ static inline void rt_st(
        rt_fl<1, width, rt_row_layout> &dst,
        const rt_bf<1, 1, rt_row_layout> & a_rt,
        const uint64_t b_st_desc
    );
    __device__ static inline void st_st(
        rt_fl<1, width, rt_row_layout> &dst,
        const uint64_t a_st_desc,
        const uint64_t b_st_desc
    );
};

}
}

#include "4x1.cuh"
#include "4x2.cuh"
#include "4x3.cuh"
#include "4x4.cuh"

// can add bigger ones later
// #include "4x5.cuh"
// #include "4x6.cuh"
// #include "4x7.cuh"
#include "4x8.cuh"