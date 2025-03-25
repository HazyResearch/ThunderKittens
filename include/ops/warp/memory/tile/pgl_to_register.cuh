/**
 * @file
 * @brief Functions for reduction operations called by a single device
 */
#pragma once

#include <cuda.h>
#include "../util/reduce.cuh"


namespace kittens {

// template <int axis, bool assume_aligned, ReduceOp OP, ducks::pgl::all PGL, ducks::rt::row_layout RT, ducks::coord::tile COORD=coord<RT>, int N_THREADS=WARP_THREADS>
// __device__ static inline void ld_reduce_op(PGL p_o, const RT &src, const COORD &idx) {
//     using T = typename RT::T;
//     using U = typename PGL::dtype;
//     const int row_stride = p_o.gl.template stride<axis>();

//     constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename RT::T);
//     constexpr int memcpy_per_row = src.cols / elem_per_memcpy;
//     constexpr int dst_num_elem = src.height*src.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T>;
//     constexpr int total_calls = (dst_num_elem + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy);
//     constexpr bool needs_bounds_check = dst_num_elem % (N_THREADS*elem_per_memcpy);

//     coord<> coord = idx.template unit_coord<axis, 3>();
//     auto index = ((coord.b * p_o.gl.depth() + coord.d) * p_o.gl.rows() + coord.r) * p_o.gl.cols() + coord.c;
//     U* mc_ptr = p_o.mc_ptr + index;
//     int laneid = threadIdx.x % N_THREADS;

//     for (int i = 0; i < total_calls; i++) {
//         int load_idx = i * N_THREADS + laneid;
//         int row = load_idx / memcpy_per_row;
//         int col = (load_idx*elem_per_memcpy) % src.cols;


//         if constexpr (needs_bounds_check) {
//             if (row >= src.rows) continue;
//         }

//         if constexpr (assume_aligned) {
//             float4 val;
//             T* ptr = static_cast<T*>(mc_ptr) + row*row_stride + col;
//             multimem_ld_reduce_op<T, OP>::apply_vec(&val, ptr);
//             move<float4>::stg((float4*)ptr, val);
//         }
//         else {
//             if (row + coord.template dim<axis>() < p_o.gl.template shape<axis>()) {
//                 float4 val;
//                 T* ptr = static_cast<T*>(mc_ptr) + row*row_stride + col;
//                 multimem_ld_reduce_op<T, OP>::apply_vec(&val, ptr);
//                 move<float4>::stg((float4*)ptr, val);
//             }
//         }
//     }
// }

// template <ducks::pgl::all PGL, ducks::rt::all RT, ducks::coord::tile COORD=coord<RT>>
// __device__ static inline void all_reduce_add(PGL p_o, const RT &src, const COORD &idx) {
//     ld_reduce_op<2, false, ReduceOp::ADD>(p_o, src, idx);
// }

// template <ducks::pgl::all PGL, ducks::rt::all RT, ducks::coord::tile COORD=coord<RT>>
// __device__ static inline void all_reduce_min(PGL p_o, const RT &src, const COORD &idx) {
//     ld_reduce_op<2, false, ReduceOp::MIN>(p_o, src, idx);
// }

// template <ducks::pgl::all PGL, ducks::rt::all RT, ducks::coord::tile COORD=coord<RT>>
// __device__ static inline void all_reduce_max(PGL p_o, const RT &src, const COORD &idx) {
//     ld_reduce_op<2, false, ReduceOp::MAX>(p_o, src, idx);
// }

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

template <int axis, ReduceOp OP, ducks::pgl::all PGL, ducks::rt::row_layout RT, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void reduce_op(PGL p_o, const RT &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename PGL::dtype;

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

template <ducks::pgl::all PGL, ducks::rt::all RT, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void atomic_add(PGL p_o, const RT &src, const COORD &idx) {
    reduce_op<2, ReduceOp::ADD>(p_o, src, idx);
}

template <ducks::pgl::all PGL, ducks::rt::all RT, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void atomic_min(PGL p_o, const RT &src, const COORD &idx) {
    reduce_op<2, ReduceOp::MIN>(p_o, src, idx);
}

template <ducks::pgl::all PGL, ducks::rt::all RT, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void atomic_max(PGL p_o, const RT &src, const COORD &idx) {
    reduce_op<2, ReduceOp::MAX>(p_o, src, idx);
}

} // namespace kittens