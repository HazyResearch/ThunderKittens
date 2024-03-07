#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

#include "base.cuh"

namespace kittens {
namespace warpgroup {

template<>
struct wgmma_base<2> {
    __device__ static inline void rt_st(
        rt_fl<1, 2, rt_row_layout> &dst,
        const rt_base_bf<rt_row_layout> & a_rt,
        const uint64_t b_st_desc,
        int scale_d = 1
    ) {
        asm volatile (
            "{\n"
            ".reg .pred p;\n" \
            "setp.ne.b32 p, %21, 0;\n" \
            "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 " \
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, " \
            "{%16, %17, %18, %19}, " \
            "%20, " \
            "p, 1, 1, 1;\n" \
            "}\n"
            // a_regs, b_mat descriptor, scale-d, imm-scale-a, imm-scale-b, im-trans-b

        :   "+f"(dst.tiles[0][0].data[0].x), "+f"(dst.tiles[0][0].data[0].y),
            "+f"(dst.tiles[0][0].data[1].x), "+f"(dst.tiles[0][0].data[1].y),
            "+f"(dst.tiles[0][0].data[2].x), "+f"(dst.tiles[0][0].data[2].y),
            "+f"(dst.tiles[0][0].data[3].x), "+f"(dst.tiles[0][0].data[3].y),
            "+f"(dst.tiles[0][1].data[0].x), "+f"(dst.tiles[0][1].data[0].y),
            "+f"(dst.tiles[0][1].data[1].x), "+f"(dst.tiles[0][1].data[1].y),
            "+f"(dst.tiles[0][1].data[2].x), "+f"(dst.tiles[0][1].data[2].y),
            "+f"(dst.tiles[0][1].data[3].x), "+f"(dst.tiles[0][1].data[3].y)

        :   "r"(*(uint32_t*)&a_rt.data[0]), "r"(*(uint32_t*)&a_rt.data[1]),
            "r"(*(uint32_t*)&a_rt.data[2]), "r"(*(uint32_t*)&a_rt.data[3]),
            
            "l"(b_st_desc), "r"(scale_d)
        );
    }
    __device__ static inline void st_st(
        rt_fl<1, 2, rt_row_layout> &dst,
        const uint64_t a_st_desc,
        const uint64_t b_st_desc,
        int scale_d = 1
    ) {
        asm volatile (
            "{\n"
            ".reg .pred p;\n" \
            "setp.ne.b32 p, %18, 0;\n" \
            "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 " \
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, " \
            "%16, " \
            "%17, " \
            "p, 1, 1, 0, 1;\n" \
            "}\n"
            // a_mat descriptor, b_mat descriptor, scale-d, imm-scale-a, imm-scale-b, im-trans-a, imm-trans-b

        :   "+f"(dst.tiles[0][0].data[0].x), "+f"(dst.tiles[0][0].data[0].y),
            "+f"(dst.tiles[0][0].data[1].x), "+f"(dst.tiles[0][0].data[1].y),
            "+f"(dst.tiles[0][0].data[2].x), "+f"(dst.tiles[0][0].data[2].y),
            "+f"(dst.tiles[0][0].data[3].x), "+f"(dst.tiles[0][0].data[3].y),
            "+f"(dst.tiles[0][1].data[0].x), "+f"(dst.tiles[0][1].data[0].y),
            "+f"(dst.tiles[0][1].data[1].x), "+f"(dst.tiles[0][1].data[1].y),
            "+f"(dst.tiles[0][1].data[2].x), "+f"(dst.tiles[0][1].data[2].y),
            "+f"(dst.tiles[0][1].data[3].x), "+f"(dst.tiles[0][1].data[3].y)

        :   "l"(a_st_desc),
            "l"(b_st_desc),
        
            "r"(scale_d)
        );
    }
};
    
}
}