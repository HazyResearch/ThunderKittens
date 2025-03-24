/**
 * @file
 * @brief Group (collaborative warp) ops for pgl operations
 */
template <int axis, bool assume_aligned, ducks::rt::all RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_add(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<axis, assume_aligned, ReduceOp::ADD, RT, PGL_OBJ, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <ducks::rt::all RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_add(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::ADD, RT, PGL_OBJ, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <int axis, bool assume_aligned, ducks::rt::all RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_min(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<axis, assume_aligned, ReduceOp::MIN, RT, PGL_OBJ, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <ducks::rt::all RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_min(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::MIN, RT, PGL_OBJ, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <int axis, bool assume_aligned, ducks::rt::all RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_max(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<axis, assume_aligned, ReduceOp::MAX, RT, PGL_OBJ, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <ducks::rt::all RT, typename PGL_OBJ, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_max(PGL_OBJ p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::MAX, RT, PGL_OBJ, COORD, GROUP_THREADS>(p_o, src, idx);
}


