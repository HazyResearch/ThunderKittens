/**
 * @file
 * @brief Group (collaborative warp) ops for pgl operations
 */

 /*
 NOTES for discussion: 
 - only support ld_reduce_op for group operations as from conceptual view of
 ld_reduce_ops can be split among a group but not reduce_ops
 */
template <int axis, bool assume_aligned, ducks::pgl::all PGL, ducks::rt::all RT, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_add(PGL p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<axis, assume_aligned, ReduceOp::ADD, PGL, RT, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <ducks::pgl::all PGL, ducks::rt::all RT, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_add(PGL p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::ADD, PGL, RT, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <int axis, bool assume_aligned, ducks::pgl::all PGL, ducks::rt::all RT, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_min(PGL p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<axis, assume_aligned, ReduceOp::MIN, PGL, RT, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <ducks::pgl::all PGL, ducks::rt::all RT, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_min(PGL p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::MIN, PGL, RT, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <int axis, bool assume_aligned, ducks::pgl::all PGL, ducks::rt::all RT, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_max(PGL p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<axis, assume_aligned, ReduceOp::MAX, PGL, RT, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <ducks::pgl::all PGL, ducks::rt::all RT, ducks::coord::tile COORD=coord<RT>>
__device__ static inline void all_reduce_max(PGL p_o, const RT &src, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::MAX, PGL, RT, COORD, GROUP_THREADS>(p_o, src, idx);
}


