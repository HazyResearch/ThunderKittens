/**
 * @file
 * @brief Group (collaborative warp) ops for pgl operations
 */

template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_add(ST &dst, const PGL &src, int dev_id, const COORD &idx) {
    kittens::ld_reduce_op<axis, assume_aligned, ReduceOp::ADD, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_id, idx);
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_add(ST &dst, const PGL &src, int dev_id, const COORD &idx) {
    kittens::ld_reduce_op<2, false, ReduceOp::ADD, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_id, idx);
}

template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_min(ST &dst, const PGL &src, int dev_id, const COORD &idx) {
    kittens::ld_reduce_op<axis, assume_aligned, ReduceOp::MIN, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_id, idx);
}

template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_min(ST &dst, const PGL &src, int dev_id, const COORD &idx) {
    kittens::ld_reduce_op<2, false, ReduceOp::MIN, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_id, idx);
}

template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_max(ST &dst, const PGL &src, int dev_id, const COORD &idx) {
    kittens::ld_reduce_op<axis, assume_aligned, ReduceOp::MAX, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_id, idx);
}

template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_max(ST &dst, const PGL &src, int dev_id, const COORD &idx) {
    kittens::ld_reduce_op<2, false, ReduceOp::MAX, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_id, idx);
}

template <int axis, bool assume_aligned, ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void broadcast(const PGL &dst, const ST &src, int dev_id, const COORD &idx) {
    kittens::broadcast<axis, assume_aligned, false, PGL, ST, COORD, GROUP_THREADS>(dst, src, dev_id, idx);
}

template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void broadcast(const PGL &dst, const ST &src, int dev_id, const COORD &idx) {
    kittens::broadcast<2, false, PGL, ST, COORD, GROUP_THREADS>(dst, src, dev_id, idx);
}
