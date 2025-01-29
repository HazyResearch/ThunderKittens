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
__device__ inline static void load_async(RT &dst, const TM &src) {
    static_assert(RT::height == TM::height, "register tile and tensor tile must match height");
    static_assert(RT::width == TM::width, "register tile and tensor tile must match width");

    using T2 = RT::dtype;
    using U  = typename TM::dtype;
    using U2 = base_types::packing<typename TM::dtype>::packed_type;

    if constexpr (sizeof(typename TM::dtype) == 1) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(*(uint32_t*) &dst.tiles[i][j].data[0]),
                          "=r"(*(uint32_t*) &dst.tiles[i][j].data[1]),
                          "=r"(*(uint32_t*) &dst.tiles[i][j].data[2]),
                          "=r"(*(uint32_t*) &dst.tiles[i][j].data[3])
                        : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col)/(4/(uint32_t)sizeof(U)))
                    );
                } else {
                    static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "register layout must be row");
                }
            }
        }
    } else if constexpr (sizeof(typename TM::dtype) == 2) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(*(uint32_t*) &dst.tiles[i][j].data[0]),
                          "=r"(*(uint32_t*) &dst.tiles[i][j].data[1]),
                          "=r"(*(uint32_t*) &dst.tiles[i][j].data[2]),
                          "=r"(*(uint32_t*) &dst.tiles[i][j].data[3])
                        : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col))
                    );
                } else {
                    static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "register layout must be row");
                }
            }
        }
    }
    else if constexpr (sizeof(typename TM::dtype) == 4) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                U2 data[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                    : "=f"(data[0].x), "=f"(data[0].y),
                      "=f"(data[1].x), "=f"(data[1].y),
                      "=f"(data[2].x), "=f"(data[2].y),
                      "=f"(data[3].x), "=f"(data[3].y)
                    : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col)/(4/(uint32_t)sizeof(U)))
                );
                #pragma unroll
                for(int k = 0; k < 4; k++) {
                    dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                }
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
__device__ inline static void store_async(TM &dst, const RT &src) {
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
                        :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
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
                U2 data[4];
                #pragma unroll
                for(int k = 0; k < 4; k++) {
                    data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x2.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
                    :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                       "f"(data[0].x), "f"(data[0].y),
                       "f"(data[1].x), "f"(data[1].y),
                       "f"(data[2].x), "f"(data[2].y),
                       "f"(data[3].x), "f"(data[3].y)
                );
            }
        }
    }
}

__device__ inline static void tm_load_wait() {
   asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ inline static void tm_store_wait() {
   asm volatile("tcgen05.wait::st.sync.aligned;"); 
}

}