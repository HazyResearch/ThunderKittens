/**
 * @file
 * @brief Group (collaborative warp) ops for pgl operations
 */

template<int axis, ReduceOp OP, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, N_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void ld_reduce_op(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename PGL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    static_assert(std::is_same_v<U, kittens::bf16> || std::is_same_v<U, half> || !std::is_same_v<U, float>, 
        "Unsupported type for ld_reduce_op");
    
    auto coord = idx.template unit_coord<axis, 3>();
    auto index = ((coord.b * src[dev_id].depth() + coord.d) * src[dev_id].rows() + coord.r) * src[dev_id].cols() + coord.c;
    U *mc_ptr = src.mc_vas[dev_id] + index;
    
    const int row_stride = src[dev_id].template stride<axis>();
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
                &dst_buf[0], (U2*)&mc_ptr[(row+0)*row_stride + (col+0)]);
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[1], (U2*)&mc_ptr[(row+0)*row_stride + (col+8)]);
            dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(dst_buf[0]);
            dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(dst_buf[1]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + 2*(warp_laneid % 4);
            U2 dst_buf[2];
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[0], (U2*)&mc_ptr[(row+8)*row_stride + (col+0)]);
            multimem_ld_reduce_op<U2, OP>::apply(
                &dst_buf[1], (U2*)&mc_ptr[(row+8)*row_stride + (col+8)]);
            dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(dst_buf[0]);
            dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(dst_buf[1]);
        }
    }
}

template<int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, N_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_add(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<axis, ReduceOp::ADD>(dst, src, dev_id, idx);
}

template<ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, N_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_add(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::ADD>(dst, src, dev_id, idx);
}

template<int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, N_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_min(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<axis, ReduceOp::MIN>(dst, src, dev_id, idx);
}

template<ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, N_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_min(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::MIN>(dst, src, dev_id, idx);
}

template<int axis, ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, N_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_max(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<axis, ReduceOp::MAX>(dst, src, dev_id, idx);
}

template<ducks::rt::row_layout RT, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<rt<typename RT::T, N_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void all_reduce_max(RT &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<2, ReduceOp::MAX>(dst, src, dev_id, idx);
}