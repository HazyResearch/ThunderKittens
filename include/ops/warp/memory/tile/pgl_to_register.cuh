/**
 * @file
 * @brief Functions for reduction operations called by a single device
 */
#pragma once

#include <cuda.h>
#include "../util/ld_reduce.cuh"
#include "../util/reduce.cuh"


namespace kittens {

enum class ReduceOperation {
    ADD,
    MIN,
    MAX
};

/**
 * @brief Broadcast data from a register tile to all GPUs.
 *
 * @tparam RT The row-major layout tile type.
 * @tparam PGL_OBJ The destination object type.
 * @tparam COORD The coordinate type for indexing.
 * @param p_o[out] The destination object to broadcast to.
 * @param src[in] The source register tile to broadcast from.
 * @param idx[in] The starting coordinate for broadcasting.
 */
template <ducks::rt::row_layout RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void broadcast(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename PGL_OBJ::dtype;

    #ifdef KITTENS_HOPPER
    // static assert that we're not using fp8e4m3 or fp8e5m2
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for load/store");
    #endif

    // TODO: update to take in AXIS parameter instead of hardcoding 2
    auto coord = idx.template unit_coord<2, 3>();
    auto index = ((coord.b * p_o.gl.depth() + coord.d) * p_o.gl.rows() + coord.r) * p_o.gl.cols() + coord.c;
    U *mc_ptr = p_o.mc_ptr + index;

    U *dst_ptr = (U*)&p_o.gl[(idx.template unit_coord<2, 3>())];
    const int row_stride = p_o.gl.template stride<2>();
    using U2 = base_types::packing<U>::packed_type;
    int laneid = kittens::laneid();
    int warphalf = (laneid & 16) > 0;
    int warphalflaneid = laneid % 16;

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row_0to3 = i*src.tile_size_row + (warphalflaneid / 4);
        int row = i*src.tile_size_row + (laneid / 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + warphalf*8 + 2*(laneid % 4);
            U2 transfers[2];
            transfers[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            transfers[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
            transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
            // TODO: figure out why this call leads to issues w/ TMA
            kittens::multimem_reduce<U2>::add((U2*)&mc_ptr[(row_0to3+0)*row_stride + col], &transfers[0]);
            kittens::multimem_reduce<U2>::add((U2*)&mc_ptr[(row_0to3+4)*row_stride + col], &transfers[1]);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + warphalf*8 + 2*(laneid % 4);
            U2 transfers[2];
            transfers[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            transfers[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
            transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
            
            kittens::multimem_reduce<U2>::add((U2*)&mc_ptr[(row_0to3+8)*row_stride + col], &transfers[0]);
            kittens::multimem_reduce<U2>::add((U2*)(&mc_ptr[(row_0to3+12)*row_stride + col]), &transfers[1]);
        }
    }
}

} // namespace kittens