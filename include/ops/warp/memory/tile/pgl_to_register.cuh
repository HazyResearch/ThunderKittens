/**
 * @file
 * @brief Functions for performing operations between PGLs and register tiles.
 */
#pragma once

#include "../util/reduce.cuh"

namespace kittens {

template <int axis, ReduceOp OP, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void ld_reduce_op(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename PGL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    static_assert(std::is_same_v<U, kittens::bf16> || std::is_same_v<U, half> || std::is_same_v<U, float>, 
        "Unsupported type for ld_reduce_op");

    U *src_mc_ptr = src.mc_ptr_at(idx.template unit_coord<axis, 3>(), dev_id);
    const int row_stride = src[dev_id].template stride<axis>();
    int laneid = kittens::laneid();
    int warphalf = (laneid & 16) > 0;
    int warphalflaneid = laneid % 16;
    
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row_0to3 = i*dst.tile_size_row + (warphalflaneid / 4);
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + warphalf*8 + 2*(laneid % 4);
            U2 dst_buf[2];
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[0], (U2*)&src_mc_ptr[(row_0to3+0)*row_stride + col]);
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[1], (U2*)&src_mc_ptr[(row_0to3+4)*row_stride + col]);
            dst_buf[1-warphalf] = packed_shfl_sync(MASK_ALL, dst_buf[1-warphalf], laneid^16);
            dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(dst_buf[0]);
            dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(dst_buf[1]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + warphalf*8 + 2*(laneid % 4);
            U2 dst_buf[2];
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[0], (U2*)&src_mc_ptr[(row_0to3+8)*row_stride + col]);
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[1], (U2*)&src_mc_ptr[(row_0to3+12)*row_stride + col]);
            dst_buf[1-warphalf] = packed_shfl_sync(MASK_ALL, dst_buf[1-warphalf], laneid^16);
            dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(dst_buf[0]);
            dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(dst_buf[1]);
        }
    }
}

/**
 * @brief Add data together across all devices in a PGL and store the result in a register tile.
 * 
 * @tparam RT The register tile layout type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination tile to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template <int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_add(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<axis, ReduceOp::ADD>(dst, src, dev_id, idx);
}

template <ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_add(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::ADD>(dst, src, dev_id, idx);
}

/**
 * @brief Store the minimum value across all devices in a PGL into a register tile.
 * 
 * @tparam RT The register tile layout type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination tile to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template <int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_min(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<axis, ReduceOp::MIN>(dst, src, dev_id, idx);
}

template <ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_min(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::MIN>(dst, src, dev_id, idx);
}

/**
 * @brief Store the maximum value across all devices in a PGL into a register tile.
 * 
 * @tparam RT The register tile layout type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination tile to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template <int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_max(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<axis, ReduceOp::MAX>(dst, src, dev_id, idx);
}

template <ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_max(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::MAX>(dst, src, dev_id, idx);
}

template <int axis, ReduceOp OP, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void reduce_op(const PGL &dst, const RT &src, int dev_id, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename PGL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    
    static_assert(std::is_same_v<U, kittens::bf16> || std::is_same_v<U, half> || std::is_same_v<U, float>, 
        "Unsupported type for reduce_op");

    U *dst_mc_ptr = dst.mc_ptr_at(idx.template unit_coord<axis, 3>(), dev_id);
    const int row_stride = dst[dev_id].template stride<axis>();
    int laneid = kittens::laneid();
    int warphalf = (laneid & 16) > 0;
    int warphalflaneid = laneid % 16;

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row_0to3 = i*src.tile_size_row + (warphalflaneid / 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + warphalf*8 + 2*(laneid % 4);
            U2 transfers[2];
            transfers[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            transfers[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
            transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
            
            multimem_reduce_op<U2, OP>::apply(
                (U2*)&dst_mc_ptr[(row_0to3+0)*row_stride + col], &transfers[0]);
            multimem_reduce_op<U2, OP>::apply(
                (U2*)&dst_mc_ptr[(row_0to3+4)*row_stride + col], &transfers[1]);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + warphalf*8 + 2*(laneid % 4);
            U2 transfers[2];
            transfers[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            transfers[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
            transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
            
            multimem_reduce_op<U2, OP>::apply(
                (U2*)&dst_mc_ptr[(row_0to3+8)*row_stride + col], &transfers[0]);
            multimem_reduce_op<U2, OP>::apply(
                (U2*)&dst_mc_ptr[(row_0to3+12)*row_stride + col], &transfers[1]);
        }
    }
}

/**
 * @brief Add data from a register tile to global memory for all devices in a PGL
 * 
 * @tparam RT The register tile layout type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination tile to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template <int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void atomic_add(const PGL &dst, const RT &src, int dev_id, const COORD &idx) {
    reduce_op<axis, ReduceOp::ADD>(dst, src, dev_id, idx);
}

template <ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void atomic_add(const PGL &dst, const RT &src, int dev_id, const COORD &idx) {
    reduce_op<2, ReduceOp::ADD>(dst, src, dev_id, idx);
}

/**
 * @brief Store data from a register tile to global memory for all devices in a PGL
 * 
 * @tparam RT The register tile layout type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination tile to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template <int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void broadcast(const PGL &dst, const RT &src, int dev_id, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename PGL::dtype;

    #ifdef KITTENS_HOPPER
    // static assert that we're not using fp8e4m3 or fp8e5m2
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for load/store");
    #endif
    
    U *dst_mc_ptr = dst.mc_ptr_at(idx.template unit_coord<axis, 3>(), dev_id);
    const int row_stride = dst[dev_id].template stride<axis>();
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
            *(U2*)(&dst_mc_ptr[(row_0to3+0)*row_stride + col]) = transfers[0];
            *(U2*)(&dst_mc_ptr[(row_0to3+4)*row_stride + col]) = transfers[1];
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + warphalf*8 + 2*(laneid % 4);
            U2 transfers[2];
            transfers[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            transfers[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
            transfers[1-warphalf] = packed_shfl_sync(MASK_ALL, transfers[1-warphalf], laneid^16);
            *(U2*)(&dst_mc_ptr[(row_0to3+ 8)*row_stride + col]) = transfers[0];
            *(U2*)(&dst_mc_ptr[(row_0to3+12)*row_stride + col]) = transfers[1];
        }
    }
}

template <ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void broadcast(const PGL &dst, const RT &src, int dev_id, const COORD &idx) {
    broadcast<2>(dst, src, dev_id, idx);
}

} // namespace kittens