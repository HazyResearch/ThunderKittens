template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<rv<typename RV::T, N_WARPS*RV::length, typename RV::layout>>>
__device__ inline static void all_reduce_add(RV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ::kittens::ld_reduce_op<ReduceOp::ADD>(dst, src, dev_id, coord<RV>(idx.b, idx.d, idx.r, idx.c*N_WARPS+warpid()));
}

template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<rv<typename RV::T, N_WARPS*RV::length, typename RV::layout>>>
__device__ inline static void all_reduce_min(RV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ::kittens::ld_reduce_op<ReduceOp::MIN>(dst, src, dev_id, coord<RV>(idx.b, idx.d, idx.r, idx.c*N_WARPS+warpid()));
}

template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<rv<typename RV::T, N_WARPS*RV::length, typename RV::layout>>>
__device__ inline static void all_reduce_max(RV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ::kittens::ld_reduce_op<ReduceOp::MAX>(dst, src, dev_id, coord<RV>(idx.b, idx.d, idx.r, idx.c*N_WARPS+warpid()));
}

template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<rv<typename RV::T, N_WARPS*RV::length, typename RV::layout>>>
__device__ inline static void atomic_add(const PGL &src, const RV &src, int dev_id, const COORD &idx) {
    ::kittens::reduce_op<ReduceOp::ADD>(dst, src, dev_id, coord<RV>(idx.b, idx.d, idx.r, idx.c*N_WARPS+warpid()));
}

template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<rv<typename RV::T, N_WARPS*RV::length, typename RV::layout>>>
__device__ inline static void broadcast(const PGL &src, const RV &src, int dev_id, const COORD &idx) {
    ::kittens::broadcast(dst, src, dev_id, coord<RV>(idx.b, idx.d, idx.r, idx.c*N_WARPS+warpid()));
}