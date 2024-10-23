/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

namespace detail {
template<ducks::st::all ST> __device__ inline static int wmma_offset(int tile_row, int tile_col) {
    return tile_row*512 + tile_col*(512*ST::underlying_height);
}
}

// These probably need to be redone to reduce bank conflicts.
// They currently work fine with xor layout but it should be
// possible to reduce their bank conflicts with other layouts too.

/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    // convert to shared state space
    uint64_t shared_addr;
    asm volatile("cvta.to.shared.u64 %0, %1;\n" : "=l"(shared_addr) : "l"((uint64_t)(&src)));

    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if constexpr(std::is_same_v<typename ST::layout, ducks::st_layout::wmma>) {
                U2 tmp[4];
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 2) {
                    asm volatile("wmma.load.a.sync.aligned.row.m16n16k16.shared.bf16 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(*(uint32_t*)&tmp[0]), "=r"(*(uint32_t*)(&tmp[1])), "=r"(*(uint32_t*)&tmp[2]), "=r"(*(uint32_t*)(&tmp[3]))
                                 : "l"(shared_addr+detail::wmma_offset<ST>(i, j)));
                }
                else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 4) {
                    asm volatile("wmma.load.a.sync.aligned.row.m16n16k16.shared.tf32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                                 : "=f"(tmp[0].x), "=f"(tmp[0].y), "=f"(tmp[1].x), "=f"(tmp[1].y), "=f"(tmp[2].x), "=f"(tmp[2].y), "=f"(tmp[3].x), "=f"(tmp[3].y)
                                 : "l"(shared_addr+detail::wmma_offset<ST>(i, j)));
                }
                else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::col> && sizeof(typename ST::dtype) == 2) {
                    asm volatile("wmma.load.b.sync.aligned.row.m16n16k16.shared.bf16 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(*(uint32_t*)&tmp[0]), "=r"(*(uint32_t*)(&tmp[2])), "=r"(*(uint32_t*)&tmp[1]), "=r"(*(uint32_t*)(&tmp[3]))
                                 : "l"(shared_addr+detail::wmma_offset<ST>(i, j)));
                }
                else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::col> && sizeof(typename ST::dtype) == 4) {
                    asm volatile("wmma.load.b.sync.aligned.row.m16n16k16.shared.tf32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                                 : "=f"(tmp[0].x), "=f"(tmp[0].y), "=f"(tmp[2].x), "=f"(tmp[2].y), "=f"(tmp[1].x), "=f"(tmp[1].y), "=f"(tmp[3].x), "=f"(tmp[3].y)
                                 : "l"(shared_addr+detail::wmma_offset<ST>(i, j)));
                }
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
            else
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 2) {
                // handle the row-major layout for 16-bit types
                int row = i*dst.tile_size + (laneid / 4);
                int col = j*dst.tile_size + 2*(laneid % 4);
                U2 tmp[4];
                move<U2>::lds(tmp[0], &src[{row+0, col+0}]);
                move<U2>::lds(tmp[1], &src[{row+8, col+0}]);
                move<U2>::lds(tmp[2], &src[{row+0, col+8}]);
                move<U2>::lds(tmp[3], &src[{row+8, col+8}]);
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 4) {
                // handle the row-major layout for 32-bit types
                int row = i*dst.tile_size + (laneid / 4);
                int col = j*dst.tile_size + 2*(laneid % 4);
                int blit = 4 * ((laneid%4)/2);
                U2 tmp[4];
                static constexpr int swizzle_repeat = ST::swizzle_bytes * 8;
                static constexpr int subtile_cols   = ST::swizzle_bytes / sizeof(U);
                const int outer_idx = col/subtile_cols;
                const uint64_t addr_1 = (uint64_t)(&src.data[outer_idx*ST::rows*subtile_cols + (row+0)*subtile_cols + col%subtile_cols]);
                const uint64_t addr_2 = (uint64_t)(&src.data[outer_idx*ST::rows*subtile_cols + (row+8)*subtile_cols + col%subtile_cols]);
                const int swizzle_1 = blit ^ ((addr_1 % swizzle_repeat) >> 7) << 4;
                const int swizzle_2 = blit ^ ((addr_2 % swizzle_repeat) >> 7) << 4;
                move<U>::lds(tmp[0].x, (U*)((addr_1+ 0)^swizzle_1));
                move<U>::lds(tmp[0].y, (U*)((addr_1+ 4)^swizzle_1));
                move<U>::lds(tmp[2].x, (U*)((addr_1+32)^swizzle_1));
                move<U>::lds(tmp[2].y, (U*)((addr_1+36)^swizzle_1));
                move<U>::lds(tmp[1].x, (U*)((addr_2+ 0)^swizzle_2));
                move<U>::lds(tmp[1].y, (U*)((addr_2+ 4)^swizzle_2));
                move<U>::lds(tmp[3].x, (U*)((addr_2+32)^swizzle_2));
                move<U>::lds(tmp[3].y, (U*)((addr_2+36)^swizzle_2));
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
                if(blit) {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        dst.tiles[i][j].data[k] = T2{dst.tiles[i][j].data[k].y, dst.tiles[i][j].data[k].x};
                    }
                }
            }
            else {
                // handle the column-major layout
                int row = i*dst.tile_size + 2*(laneid % 4);
                int col = j*dst.tile_size + (laneid / 4);
                U2 tmp[4];
                move<U>::lds(tmp[0].x, &src[{row+0, col+0}]);
                move<U>::lds(tmp[0].y, &src[{row+1, col+0}]);
                move<U>::lds(tmp[1].x, &src[{row+0, col+8}]);
                move<U>::lds(tmp[1].y, &src[{row+1, col+8}]);
                move<U>::lds(tmp[2].x, &src[{row+8, col+0}]);
                move<U>::lds(tmp[2].y, &src[{row+9, col+0}]);
                move<U>::lds(tmp[3].x, &src[{row+8, col+8}]);
                move<U>::lds(tmp[3].y, &src[{row+9, col+8}]);
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
        }
    }
}


/**
 * @brief Store data into a shared tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void store(ST &dst, const RT &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    // convert to shared state space
    uint64_t shared_addr;
    asm volatile("cvta.to.shared.u64 %0, %1;\n" : "=l"(shared_addr) : "l"((uint64_t)(&src)));

    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            if constexpr(std::is_same_v<typename ST::layout, ducks::st_layout::wmma>) {
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 2) {
                    asm volatile("wmma.store.d.sync.aligned.row.m16n16k16.shared.f16 [%4], {%0, %1, %2, %3};\n"
                                 :: "r"(*(uint32_t*)&tmp[0]), "r"(*(uint32_t*)(&tmp[1])), "r"(*(uint32_t*)&tmp[2]), "r"(*(uint32_t*)(&tmp[3])), "r"(shared_addr+detail::wmma_offset<ST>(i, j)));
                }
                else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 4) {
                    asm volatile("wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [%8], {%0, %1, %2, %3, %4, %5, %6, %7};\n"
                                 :: "=f"(tmp[0].x), "=f"(tmp[0].y), "=f"(tmp[1].x), "=f"(tmp[1].y), "=f"(tmp[2].x), "=f"(tmp[2].y), "=f"(tmp[3].x), "=f"(tmp[3].y), "r"(shared_addr+detail::wmma_offset<ST>(i, j)));
                }
                else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::col> && sizeof(typename ST::dtype) == 2) {
                    asm volatile("wmma.store.d.sync.aligned.col.m16n16k16.shared.f16 [%4], {%0, %1, %2, %3};\n"
                                 :: "r"(*(uint32_t*)&tmp[0]), "r"(*(uint32_t*)(&tmp[2])), "r"(*(uint32_t*)&tmp[1]), "r"(*(uint32_t*)(&tmp[3])), "r"(shared_addr+detail::wmma_offset<ST>(i, j)));
                }
                else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::col> && sizeof(typename ST::dtype) == 4) {
                    asm volatile("wmma.store.d.sync.aligned.col.m16n16k16.shared.f32 [%8], {%0, %1, %2, %3, %4, %5, %6, %7};\n"
                                 :: "=f"(tmp[0].x), "=f"(tmp[0].y), "=f"(tmp[2].x), "=f"(tmp[2].y), "=f"(tmp[1].x), "=f"(tmp[1].y), "=f"(tmp[3].x), "=f"(tmp[3].y), "r"(shared_addr+detail::wmma_offset<ST>(i, j)));
                }
            }
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 2) {
                // handle the row-major layout
                int row = i*src.tile_size + (laneid / 4);
                int col = j*src.tile_size + 2*(laneid % 4);
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
                move<U2>::sts(&dst[{row+0, col+0}], tmp[0]);
                move<U2>::sts(&dst[{row+8, col+0}], tmp[1]);
                move<U2>::sts(&dst[{row+0, col+8}], tmp[2]);
                move<U2>::sts(&dst[{row+8, col+8}], tmp[3]);
            }
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 4) {
                // handle the row-major layout for 32-bit types
                int row = i*src.tile_size + (laneid / 4);
                int col = j*src.tile_size + 2*(laneid % 4);
                int blit = 4*((laneid%4) / 2);
                T2 reg_tmp[4];
                if(blit) {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        reg_tmp[k] = T2{src.tiles[i][j].data[k].y, src.tiles[i][j].data[k].x};
                    }
                }
                else {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        reg_tmp[k] = src.tiles[i][j].data[k];
                    }
                }
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(reg_tmp[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(reg_tmp[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(reg_tmp[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(reg_tmp[3]);
                static constexpr int swizzle_repeat = ST::swizzle_bytes * 8;
                static constexpr int subtile_cols   = ST::swizzle_bytes / sizeof(U);
                const int outer_idx = col/subtile_cols;
                const uint64_t addr_1 = (uint64_t)(&dst.data[outer_idx*ST::rows*subtile_cols + (row+0)*subtile_cols + col%subtile_cols]);
                const uint64_t addr_2 = (uint64_t)(&dst.data[outer_idx*ST::rows*subtile_cols + (row+8)*subtile_cols + col%subtile_cols]);
                const int swizzle_1 = blit ^ ((addr_1 % swizzle_repeat) >> 7) << 4;
                const int swizzle_2 = blit ^ ((addr_2 % swizzle_repeat) >> 7) << 4;
                move<U>::sts((U*)((addr_1+ 0)^swizzle_1), tmp[0].x);
                move<U>::sts((U*)((addr_1+ 4)^swizzle_1), tmp[0].y);
                move<U>::sts((U*)((addr_1+32)^swizzle_1), tmp[2].x);
                move<U>::sts((U*)((addr_1+36)^swizzle_1), tmp[2].y);
                move<U>::sts((U*)((addr_2+ 0)^swizzle_2), tmp[1].x);
                move<U>::sts((U*)((addr_2+ 4)^swizzle_2), tmp[1].y);
                move<U>::sts((U*)((addr_2+32)^swizzle_2), tmp[3].x);
                move<U>::sts((U*)((addr_2+36)^swizzle_2), tmp[3].y);
            }
            else {
                // handle the column-major layout
                int row = i*src.tile_size + 2*(laneid % 4);
                int col = j*src.tile_size + (laneid / 4);
                U tmp[8];
                tmp[0] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
                tmp[1] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
                tmp[2] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
                tmp[3] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
                tmp[4] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x);
                tmp[5] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y);
                tmp[6] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x);
                tmp[7] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y);
                move<U>::sts(&dst[{row+0, col+0}], tmp[0]);
                move<U>::sts(&dst[{row+1, col+0}], tmp[1]);
                move<U>::sts(&dst[{row+0, col+8}], tmp[2]);
                move<U>::sts(&dst[{row+1, col+8}], tmp[3]);
                move<U>::sts(&dst[{row+8, col+0}], tmp[4]);
                move<U>::sts(&dst[{row+9, col+0}], tmp[5]);
                move<U>::sts(&dst[{row+8, col+8}], tmp[6]);
                move<U>::sts(&dst[{row+9, col+8}], tmp[7]);
            }
        }
    }
}

}