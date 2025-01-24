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
    using T = base_types::packing<T2>::unpacked_type;
    using U = TM::dtype;
    using U2 = base_types::packing<U>::packed_type;

    // For 16-bit types (half and bfloat16)
    if constexpr (sizeof(typename TM::dtype) == 2) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                U2 tmp[4];
                // Use tcgen05.ld with appropriate shape based on tile dimensions
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x64b.x1.b32 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
                        : "r"(src.addr + (i * dst.tile_size_row << 16) + j * dst.tile_size_col/(4/sizeof(U)))
                    );
                } else {
                    // Column-major layout requires different data arrangement
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
                        : "r"(src.addr + (i * dst.tile_size_row << 16) + j * dst.tile_size_col/(4/sizeof(U)))
                    );
                }
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
        }
    }
    // For 32-bit types (float)
    else if constexpr (sizeof(typename TM::dtype) == 4) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                U2 tmp[4];
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
                        : "r"(src.addr + (i * dst.tile_size_row << 16) + j * dst.tile_size_col/(4/sizeof(U)))
                    );
                } else {
                    // Handle column-major layout for 32-bit types
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
                        : "r"(src.addr + (i * dst.tile_size_row << 16) + j * dst.tile_size_col/(4/sizeof(U)))
                    );
                }
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
        }
    }
    // For 8-bit types (fp8)
    else if constexpr (sizeof(typename TM::dtype) == 1) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                U2 tmp[4];
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x1.b32 {%0, %1, %2, %3}, [%4], 32;\n"
                        : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
                        : "r"(src.addr + (i * dst.tile_size_row << 16) + j * dst.tile_size_col/(4/sizeof(U)))
                    );
                } else {
                    // Handle column-major layout for 8-bit types
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 {%0, %1, %2, %3}, [%4], 32;\n"
                        : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
                        : "r"(src.addr + (i * dst.tile_size_row << 16) + j * dst.tile_size_col/(4/sizeof(U)))
                    );
                }
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
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
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);

                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x64b.x1.b32 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(dst.addr + (i * src.tile_size_row << 16) + j * src.tile_size_col/(4/sizeof(U))),
                           "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]), "r"(tmp[3])
                    );
                } else {
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x64b.x1.unpack::16b.b32 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(dst.addr + (i * src.tile_size_row << 16) + j * src.tile_size_col/(4/sizeof(U))),
                           "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]), "r"(tmp[3])
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
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);

                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(dst.addr + (i * src.tile_size_row << 16) + j * src.tile_size_col/(4/sizeof(U))),
                       "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]), "r"(tmp[3])
                );
            }
        }
    }
    else if constexpr (sizeof(typename TM::dtype) == 1) {
        #pragma unroll
        for(int i = 0; i < src.height; i++) {
            #pragma unroll
            for(int j = 0; j < src.width; j++) {
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);

                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x1.b32 [%0], 32, {%1, %2, %3, %4};\n"
                        :: "r"(dst.addr + (i * src.tile_size_row << 16) + j * src.tile_size_col/(4/sizeof(U))),
                           "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]), "r"(tmp[3])
                    );
                } else {
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32 [%0], 32, {%1, %2, %3, %4};\n"
                        :: "r"(dst.addr + (i * src.tile_size_row << 16) + j * src.tile_size_col/(4/sizeof(U))),
                           "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]), "r"(tmp[3])
                    );
                }
            }
        }
    }
}

}