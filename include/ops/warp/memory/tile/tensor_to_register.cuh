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

/**
 * @brief Load data from a tensor tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam TM The tensor memory tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source tensor tile.
 */
template<ducks::rt::all RT, ducks::tmem::all TM>
__device__ inline static void load(RT &dst, const TM &src) {
    static_assert(RT::height == TM::height, "register tile and tensor tile must match height");
    static_assert(RT::width == TM::width, "register tile and tensor tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = TM::dtype;
    using U2 = base_types::packing<U>::packed_type;

    if constexpr (sizeof(typename TM::dtype) == 2) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x128b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                        : "=r"(dst.tiles[i][j].data[0].x), "=r"(dst.tiles[i][j].data[0].y),
                          "=r"(dst.tiles[i][j].data[1].x), "=r"(dst.tiles[i][j].data[1].y),
                          "=r"(dst.tiles[i][j].data[2].x), "=r"(dst.tiles[i][j].data[2].y),
                          "=r"(dst.tiles[i][j].data[3].x), "=r"(dst.tiles[i][j].data[3].y)
                        : "r"(src.addr + (i * dst.tile_size_row << 16) + j * dst.tile_size_col/(4/(uint32_t)sizeof(U)))
                    );
                } else {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                        : "=r"(dst.tiles[i][j].data[0].x), "=r"(dst.tiles[i][j].data[0].y),
                          "=r"(dst.tiles[i][j].data[1].x), "=r"(dst.tiles[i][j].data[1].y),
                          "=r"(dst.tiles[i][j].data[2].x), "=r"(dst.tiles[i][j].data[2].y),
                          "=r"(dst.tiles[i][j].data[3].x), "=r"(dst.tiles[i][j].data[3].y)
                        : "r"(src.addr + (i * dst.tile_size_row << 16) + j * dst.tile_size_col/(4/(uint32_t)sizeof(U)))
                    );
                }
            }
        }
    }
    else if constexpr (sizeof(typename TM::dtype) == 4) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                    : "=f"(dst.tiles[i][j].data[0].x), "=f"(dst.tiles[i][j].data[0].y),
                      "=f"(dst.tiles[i][j].data[1].x), "=f"(dst.tiles[i][j].data[1].y),
                      "=f"(dst.tiles[i][j].data[2].x), "=f"(dst.tiles[i][j].data[2].y),
                      "=f"(dst.tiles[i][j].data[3].x), "=f"(dst.tiles[i][j].data[3].y)
                    : "r"(src.addr + (i * dst.tile_size_row << 16) + j * dst.tile_size_col/(4/(uint32_t)sizeof(U)))
                );
            }
        }
    }
}


/**
 * @brief Store data into a tensor tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam TM The tensor memory tile type
 * @param dst[out] The destination tensor tile.
 * @param src[in]  The source register tile.
 */
template<ducks::rt::all RT, ducks::tmem::all TM>
__device__ inline static void store(TM &dst, const RT &src) {
    static_assert(RT::height == TM::height, "register tile and tensor tile must match height");
    static_assert(RT::width == TM::width, "register tile and tensor tile must match width");

    using T2 = RT::dtype;
    using T = base_types::packing<T2>::unpacked_type;
    using U = TM::dtype;
    using U2 = base_types::packing<U>::packed_type;

    if constexpr (sizeof(typename TM::dtype) == 2) {
        #pragma unroll
        for(int i = 0; i < src.height; i++) {
            #pragma unroll
            for(int j = 0; j < src.width; j++) {
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x2.b32 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(dst.addr + (i * src.tile_size_row << 16) + j * src.tile_size_col/(4/(uint32_t)sizeof(U))),
                           "r"(*(uint32_t*)&src.tiles[i][j].data[0]),
                           "r"(*(uint32_t*)&src.tiles[i][j].data[1]),
                           "r"(*(uint32_t*)&src.tiles[i][j].data[2]),
                           "r"(*(uint32_t*)&src.tiles[i][j].data[3])
                    );
                }
            }
        }
    }
    else if constexpr (sizeof(typename TM::dtype) == 4) {
        #pragma unroll
        for(int i = 0; i < src.height; i++) {
            #pragma unroll
            for(int j = 0; j < src.width; j++) {
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x2.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
                    :: "r"(dst.addr + (i * src.tile_size_row << 16) + j * src.tile_size_col/(4/(uint32_t)sizeof(U))),
                       "f"(src.tiles[i][j].data[0].x), "f"(src.tiles[i][j].data[0].y),
                       "f"(src.tiles[i][j].data[1].x), "f"(src.tiles[i][j].data[1].y),
                       "f"(src.tiles[i][j].data[2].x), "f"(src.tiles[i][j].data[2].y),
                       "f"(src.tiles[i][j].data[3].x), "f"(src.tiles[i][j].data[3].y)
                );
            }
        }
    }
}

}