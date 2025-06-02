/**
 * @file
 * @brief Functions for a group to collaboratively perform operations between pgls and register tiles 
 */

template<int axis, ReduceOp OP, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void ld_reduce_op(RT &dst, const PGL &src, int dev_idx, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename PGL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    static_assert(std::is_same_v<U, kittens::bf16> || std::is_same_v<U, half> || std::is_same_v<U, float>, 
        "Unsupported type for ld_reduce_op");

    U *dst_mc_ptr = src.mc_ptr_at(idx.template unit_coord<axis, 3>(), dev_idx);
    const int row_stride = src.template stride<axis>();
    int warp_laneid = threadIdx.x % WARP_THREADS;
    const int row_offset = dst.rows*warpid();
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = row_offset + i*dst.tile_size_row + (warp_laneid / 4);
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + 2*(warp_laneid % 4);
            U2 dst_buf[2];
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[0], (U2*)&dst_mc_ptr[(row+0)*row_stride + (col+0)]);
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[1], (U2*)&dst_mc_ptr[(row+0)*row_stride + (col+8)]);
            dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(dst_buf[0]);
            dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(dst_buf[1]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + 2*(warp_laneid % 4);
            U2 dst_buf[2];
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[0], (U2*)&dst_mc_ptr[(row+8)*row_stride + (col+0)]);
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[1], (U2*)&dst_mc_ptr[(row+8)*row_stride + (col+8)]);
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
__device__ inline static void all_reduce_add(RT &dst, const PGL &src, int dev_idx, const COORD &idx) {
    ld_reduce_op<axis, ReduceOp::ADD>(dst, src, dev_idx, idx);
}

template<ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_add(RT &dst, const PGL &src, int dev_idx, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::ADD>(dst, src, dev_idx, idx);
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
__device__ inline static void all_reduce_min(RT &dst, const PGL &src, int dev_idx, const COORD &idx) {
    ld_reduce_op<axis, ReduceOp::MIN>(dst, src, dev_idx, idx);
}

template<ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_min(RT &dst, const PGL &src, int dev_idx, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::MIN>(dst, src, dev_idx, idx);
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
__device__ inline static void all_reduce_max(RT &dst, const PGL &src, int dev_idx, const COORD &idx) {
    ld_reduce_op<axis, ReduceOp::MAX>(dst, src, dev_idx, idx);
}

template<ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_max(RT &dst, const PGL &src, int dev_idx, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::MAX>(dst, src, dev_idx, idx);
}

template<int axis, ReduceOp OP, ducks::pgl::all PGL, ducks::rt::row_layout RT, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void reduce_op(const PGL &dst, const RT &src, int dev_idx, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename PGL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    static_assert(std::is_same_v<U, kittens::bf16> || std::is_same_v<U, half> || std::is_same_v<U, float>, 
        "Unsupported type for reduce_op");
    
    U *dst_mc_ptr = dst.mc_ptr_at(idx.template unit_coord<axis, 3>(), dev_idx);
    const int row_stride = dst.template stride<axis>();
    int warp_laneid = threadIdx.x % WARP_THREADS;
    const int row_offset = src.rows*warpid();
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = row_offset + i*src.tile_size_row + (warp_laneid / 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + 2*(warp_laneid % 4);
            U2 dst_buf[2];
            dst_buf[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            dst_buf[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
            multimem_reduce_op<U2, OP>::apply(
                (U2*)&dst_mc_ptr[(row+0)*row_stride + (col+0)], &dst_buf[0]);
            multimem_reduce_op<U2, OP>::apply(
                (U2*)&dst_mc_ptr[(row+0)*row_stride + (col+8)], &dst_buf[1]);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + 2*(warp_laneid % 4);
            U2 dst_buf[2];
            dst_buf[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            dst_buf[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
            multimem_reduce_op<U2, OP>::apply(
                (U2*)&dst_mc_ptr[(row+8)*row_stride + (col+0)], &dst_buf[0]);
            multimem_reduce_op<U2, OP>::apply(
                (U2*)&dst_mc_ptr[(row+8)*row_stride + (col+8)], &dst_buf[1]);
        }
    }
}

/**
 * @brief Collaboratively adds data from a register to global memory for all devices in a PGL
 * 
 * @tparam RT The row-major register tile type.
 * @tparam PGL The parallel global layout type.
 * @tparam COORD The coordinate tile type
 * @param dst The destination PGL to store the result.
 * @param src The source RT to load data from
 */
template<int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void atomic_add(const PGL &dst, const RT &src, int dev_idx, const COORD &idx) {
    reduce_op<axis, ReduceOp::ADD>(dst, src, dev_idx, idx);
}

template<ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void atomic_add(const PGL &dst, const RT &src, int dev_idx, const COORD &idx) {
    reduce_op<2, ReduceOp::ADD>(dst, src, dev_idx, idx);
}

/**
 * @brief Collaboratively stores data from a register tile to global memory for all devices in a PGL
 * 
 * @tparam RT The row-major register tile type.
 * @tparam PGL The parallel global layout type.
 * @tparam COORD The coordinate tile type
 * @param dst The destination PGL to store the result.
 * @param src The source RT to load data from
 */
template<int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void broadcast(const PGL &dst, const RT &src, int dev_idx, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename PGL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    #ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for load/store");
    #endif

    U *dst_mc_ptr = dst.mc_ptr_at(idx.template unit_coord<axis, 3>(), dev_idx);
    const int row_stride = dst.template stride<axis>();
    int warp_laneid = threadIdx.x % WARP_THREADS;
    const int row_offset = src.rows*warpid();
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = row_offset + i*src.tile_size_row + (warp_laneid / 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + 2*(warp_laneid % 4);
            *(U2*)(&dst_mc_ptr[(row+0)*row_stride + (col+0)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            *(U2*)(&dst_mc_ptr[(row+0)*row_stride + (col+8)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + 2*(warp_laneid % 4);
            *(U2*)(&dst_mc_ptr[(row+8)*row_stride + (col+0)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            *(U2*)(&dst_mc_ptr[(row+8)*row_stride + (col+8)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
        }
    }
}

template<ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void broadcast(const PGL &dst, const RT &src, int dev_idx, const COORD &idx) {
    broadcast<2>(dst, src, dev_idx, idx);
}