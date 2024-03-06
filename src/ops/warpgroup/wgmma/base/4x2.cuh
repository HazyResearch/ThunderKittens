#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

#include "base.cuh"

namespace kittens {
namespace warpgroup {

template<>
struct wgmma_base<2> {
    /**
     * @brief Perform warp-level matrix multiply-accumulate operation using register-tiled data for A and shared memory for B.
     *
     * @param dst The destination fragment where the result will be accumulated.
     * @param a_rt The source fragment for matrix A in register tiles.
     * @param b_st_desc The descriptor for matrix B in shared memory.
     * @param scale_d The scaling factor for the destination matrix.
     */
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
            "p, 1, 1, 0;\n" \
            "}\n"
            // a_regs, b_mat descriptor, scale-d, imm-scale-a, imm-scale-b, im-trans-a

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

    /**
     * @brief Perform warp-level matrix multiply-accumulate operation using shared memory for both A and B matrices.
     *
     * @param dst The destination fragment where the result will be accumulated.
     * @param a_st_desc The descriptor for matrix A in shared memory.
     * @param b_st_desc The descriptor for matrix B in shared memory.
     * @param scale_d The scaling factor for the destination matrix.
     */
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
            "p, 1, 1, 0, 0;\n" \
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
