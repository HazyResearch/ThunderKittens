/**
 * @file
 * @brief Functions for reduction operations called by a single device
 */
#pragma once

#include <cuda.h>
#include "../util/reduce.cuh"


namespace kittens {

template <int axis, ReduceOp OP, ducks::rt::row_layout RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void ld_reduce_op(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename PGL_OBJ::dtype;

    static_assert(std::is_same_v<U, kittens::bf16> || std::is_same_v<U, half> || !std::is_same_v<U, float>, 
        "Unsupported type for ld_reduce_op");

    auto coord = idx.template unit_coord<axis, 3>();
    auto index = ((coord.b * p_o.gl.depth() + coord.d) * p_o.gl.rows() + coord.r) * p_o.gl.cols() + coord.c;
    U *mc_ptr = p_o.mc_ptr + index;

    const int row_stride = p_o.gl.template stride<axis>();
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
            U2 dst_buf;
            
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf, (U2*)&mc_ptr[(row_0to3+0)*row_stride + col]);
            move<U2>::stg((U2*)&mc_ptr[(row_0to3+0)*row_stride + col], dst_buf);

            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf, (U2*)&mc_ptr[(row_0to3+4)*row_stride + col]);
            move<U2>::stg((U2*)&mc_ptr[(row_0to3+4)*row_stride + col], dst_buf);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + warphalf*8 + 2*(laneid % 4);
            U2 dst_buf;
            
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf, (U2*)&mc_ptr[(row_0to3+8)*row_stride + col]);
            move<U2>::stg((U2*)&mc_ptr[(row_0to3+8)*row_stride + col], dst_buf);

            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf, (U2*)&mc_ptr[(row_0to3+12)*row_stride + col]);
            move<U2>::stg((U2*)&mc_ptr[(row_0to3+12)*row_stride + col], dst_buf);

        }
    }
}

// TODO: bf16 and f16 can't be used with col_layout? operations with multimem need to be at least 32 bits
template <int axis, ReduceOp OP, ducks::rt::col_layout RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void ld_reduce_op(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename PGL_OBJ::dtype;

    static_assert(std::is_same_v<U, float>, "Unsupported type for col_layout ld_reduce_op");

    auto coord = idx.template unit_coord<axis, 3>();
    auto index = ((coord.b * p_o.gl.depth() + coord.d) * p_o.gl.rows() + coord.r) * p_o.gl.cols() + coord.c;
    U *mc_ptr = p_o.mc_ptr + index;

    const int row_stride = p_o.gl.template stride<axis>();
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = i*src.tile_size_row + 2*(laneid % 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (laneid / 4);
            U dst_buf;
            
            multimem_ld_reduce_op<U, OP>::apply(
                &dst_buf, &mc_ptr[(row+0)*row_stride + (col+0)]);
            move<U>::stg(&mc_ptr[(row+0)*row_stride + (col+0)], dst_buf);
            multimem_ld_reduce_op<U, OP>::apply(
                &dst_buf, &mc_ptr[(row+0)*row_stride + (col+8)]);
            move<U>::stg(&mc_ptr[(row+0)*row_stride + (col+8)], dst_buf);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (laneid / 4);
            U dst_buf; 

            multimem_ld_reduce_op<U, OP>::apply(
                &dst_buf, &mc_ptr[(row+1)*row_stride + (col+0)]);
            move<U>::stg(&mc_ptr[(row+1)*row_stride + (col+0)], dst_buf);
            multimem_ld_reduce_op<U, OP>::apply(
                &dst_buf, &mc_ptr[(row+1)*row_stride + (col+8)]);
            move<U>::stg(&mc_ptr[(row+1)*row_stride + (col+8)], dst_buf);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (laneid / 4);
            U dst_buf;

            multimem_ld_reduce_op<U, OP>::apply(
                &dst_buf, &mc_ptr[(row+8)*row_stride + (col+0)]);
            move<U>::stg(&mc_ptr[(row+8)*row_stride + (col+0)], dst_buf);
            multimem_ld_reduce_op<U, OP>::apply(
                &dst_buf, &mc_ptr[(row+8)*row_stride + (col+8)]);
            move<U>::stg(&mc_ptr[(row+8)*row_stride + (col+8)], dst_buf);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (laneid / 4);
            U dst_buf;

            multimem_ld_reduce_op<U, OP>::apply(
                &dst_buf, &mc_ptr[(row+9)*row_stride + (col+0)]);
            move<U>::stg(&mc_ptr[(row+9)*row_stride + (col+0)], dst_buf);
            multimem_ld_reduce_op<U, OP>::apply(
                &dst_buf, &mc_ptr[(row+9)*row_stride + (col+8)]);
            move<U>::stg(&mc_ptr[(row+9)*row_stride + (col+8)], dst_buf);
        }
    }
}

template <ducks::rt::all RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_add(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::ADD>(p_o, src, idx);
}

template <ducks::rt::all RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_min(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::MIN>(p_o, src, idx);
}

template <ducks::rt::all RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_max(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::MAX>(p_o, src, idx);
}

template <int axis, ReduceOp OP, ducks::rt::row_layout RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void reduce_op(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename PGL_OBJ::dtype;

    static_assert(std::is_same_v<U, kittens::bf16> || std::is_same_v<U, half> || !std::is_same_v<U, float>, 
        "Unsupported type for reduce_op");

    auto coord = idx.template unit_coord<axis, 3>();
    auto index = ((coord.b * p_o.gl.depth() + coord.d) * p_o.gl.rows() + coord.r) * p_o.gl.cols() + coord.c;
    U *mc_ptr = p_o.mc_ptr + index;

    const int row_stride = p_o.gl.template stride<axis>();
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
            
            multimem_reduce_op<U2, OP>::apply(
                (U2*)&mc_ptr[(row_0to3+0)*row_stride + col], &transfers[0]);
            multimem_reduce_op<U2, OP>::apply(
                (U2*)&mc_ptr[(row_0to3+4)*row_stride + col], &transfers[1]);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + warphalf*8 + 2*(laneid % 4);
            U2 transfers[2];
            transfers[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            transfers[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
            transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
            
            multimem_reduce_op<U2, OP>::apply(
                (U2*)&mc_ptr[(row_0to3+8)*row_stride + col], &transfers[0]);
            multimem_reduce_op<U2, OP>::apply(
                (U2*)&mc_ptr[(row_0to3+12)*row_stride + col], &transfers[1]);
        }
    }
}

// TODO: bf16 and f16 can't be used with col_layout? operations with multimem need to be at least 32 bits
template <int axis, ReduceOp OP, ducks::rt::col_layout RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void reduce_op(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename PGL_OBJ::dtype;

    static_assert(std::is_same_v<U, float>, "Unsupported type for col_layout reduce_op");

    auto coord = idx.template unit_coord<axis, 3>();
    auto index = ((coord.b * p_o.gl.depth() + coord.d) * p_o.gl.rows() + coord.r) * p_o.gl.cols() + coord.c;
    U *mc_ptr = p_o.mc_ptr + index;

    const int row_stride = p_o.gl.template stride<axis>();
    int laneid = threadIdx.x % 32;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = i*src.tile_size_row + 2*(laneid % 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (laneid / 4);
            multimem_reduce_op<U, OP>::apply(
                &mc_ptr[(row+0)*row_stride + (col+0)], &base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x));
            multimem_reduce_op<U, OP>::apply(
                &mc_ptr[(row+0)*row_stride + (col+8)], &base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x));
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (laneid / 4);
            multimem_reduce_op<U, OP>::apply(
                &mc_ptr[(row+1)*row_stride + (col+0)], &base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y));
            multimem_reduce_op<U, OP>::apply(
                &mc_ptr[(row+1)*row_stride + (col+8)], &base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y));
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (laneid / 4);
            multimem_reduce_op<U, OP>::apply(
                &mc_ptr[(row+8)*row_stride + (col+0)], &base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x));
            multimem_reduce_op<U, OP>::apply(
                &mc_ptr[(row+8)*row_stride + (col+8)], &base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x));
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (laneid / 4);
            multimem_reduce_op<U, OP>::apply(
                &mc_ptr[(row+9)*row_stride + (col+0)], &base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y));
            multimem_reduce_op<U, OP>::apply(
                &mc_ptr[(row+9)*row_stride + (col+8)], &base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y));
        }
    }
}

template <ducks::rt::all RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void atomic_add(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    reduce_op<2, ReduceOp::ADD>(p_o, src, idx);
}

template <ducks::rt::all RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void atomic_min(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    reduce_op<2, ReduceOp::MIN>(p_o, src, idx);
}

template <ducks::rt::all RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void atomic_max(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    reduce_op<2, ReduceOp::MAX>(p_o, src, idx);
}

} // namespace kittens