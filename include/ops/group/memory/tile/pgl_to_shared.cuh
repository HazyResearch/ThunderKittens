/**
 * @file
 * @brief Group (collaborative warp) ops for pgl operations
 */

 /*
 NOTES for discussion: 
 - only support ld_reduce_op for group operations as from conceptual view of
 ld_reduce_ops can be split among a group but not reduce_ops
 */
template <int axis, bool assume_aligned, ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_add(PGL p_o, const ST &src, const COORD &idx) {
    ld_reduce_op<axis, assume_aligned, ReduceOp::ADD, PGL, ST, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_add(PGL p_o, const ST &src, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::ADD, PGL, ST, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <int axis, bool assume_aligned, ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_min(PGL p_o, const ST &src, const COORD &idx) {
    ld_reduce_op<axis, assume_aligned, ReduceOp::MIN, PGL, ST, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_min(PGL p_o, const ST &src, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::MIN, PGL, ST, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <int axis, bool assume_aligned, ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_max(PGL p_o, const ST &src, const COORD &idx) {
    ld_reduce_op<axis, assume_aligned, ReduceOp::MAX, PGL, ST, COORD, GROUP_THREADS>(p_o, src, idx);
}

template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_max(PGL p_o, const ST &src, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::MAX, PGL, ST, COORD, GROUP_THREADS>(p_o, src, idx);
}


