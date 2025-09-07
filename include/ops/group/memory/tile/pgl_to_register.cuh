/**
 * @file
 * @brief Functions for a group to collaboratively perform operations between pgls and register tiles 
 */

template<int axis, ReduceOp OP, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void ld_reduce_op(RT &dst, const PGL &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename PGL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    static_assert(std::is_same_v<U, kittens::bf16> || std::is_same_v<U, half> || std::is_same_v<U, float>, 
        "Unsupported type for ld_reduce_op");

    U *src_mc_ptr = src.mc_ptr_at(idx.template unit_coord<axis, 3>());
    const int row_stride = src.template stride<axis>();
    int warp_laneid = threadIdx.x % WARP_THREADS;
    int local_warpid;
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();
    const int row_offset = dst.rows*local_warpid;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = row_offset + i*dst.tile_size_row + (warp_laneid / 4);
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + 2*(warp_laneid % 4);
            U2 dst_buf[2];
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[0], (U2*)&src_mc_ptr[(row+0)*row_stride + (col+0)]);
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[1], (U2*)&src_mc_ptr[(row+0)*row_stride + (col+8)]);
            dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(dst_buf[0]);
            dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(dst_buf[1]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + 2*(warp_laneid % 4);
            U2 dst_buf[2];
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[0], (U2*)&src_mc_ptr[(row+8)*row_stride + (col+0)]);
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[1], (U2*)&src_mc_ptr[(row+8)*row_stride + (col+8)]);
            dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(dst_buf[0]);
            dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(dst_buf[1]);
        }
    }
}

/**
 * @brief Collaboratively adds data together across all devices and stores the result in a register tile.
 * 
 * @tparam RT The row-major register tile type.
 * @tparam PGL The parallel global layout type.
 * @tparam COORD The coordinate tile type
 * @param dst The destination register tile to store the result.
 * @param src The source PGL to load data across devices from
 */
template<int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_add(RT &dst, const PGL &src, const COORD &idx) {
    ld_reduce_op<axis, ReduceOp::ADD>(dst, src, idx);
}

template<ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_add(RT &dst, const PGL &src, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::ADD>(dst, src, idx);
}

/**
 * @brief Collaboratively store the minimum value across all devices in a PGL into a register tile.
 * 
 * @tparam RT The row-major register tile type.
 * @tparam PGL The parallel global layout type.
 * @tparam COORD The coordinate tile type
 * @param dst The destination register tile to store the result.
 * @param src The source PGL to load data across devices from
 */
template<int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_min(RT &dst, const PGL &src, const COORD &idx) {
    ld_reduce_op<axis, ReduceOp::MIN>(dst, src, idx);
}

template<ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_min(RT &dst, const PGL &src, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::MIN>(dst, src, idx);
}

/**
 * @brief Collaboratively store the maximum value across all devices in a PGL into a register tile.
 * 
 * @tparam RT The row-major register tile type.
 * @tparam PGL The parallel global layout type.
 * @tparam COORD The coordinate tile type
 * @param dst The destination register tile to store the result.
 * @param src The source PGL to load data across devices from
 */
template<int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_max(RT &dst, const PGL &src, const COORD &idx) {
    ld_reduce_op<axis, ReduceOp::MAX>(dst, src, idx);
}

template<ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_max(RT &dst, const PGL &src, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::MAX>(dst, src, idx);
}
