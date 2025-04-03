/**
 * @file
 * @brief Functions for a kittens::group to collaboratively perform operations between pgls and register vectors.
 */

/**
 * @brief Collaboratively adds data together across all devices and stores the result in a register vector.
 *
 * @tparam RV The register vector type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination register vector to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<rv<typename RV::T, N_WARPS*RV::length, typename RV::layout>>>
__device__ inline static void all_reduce_add(RV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ::kittens::ld_reduce_op<ReduceOp::ADD>(dst, src, dev_id, coord<RV>(idx.b, idx.d, idx.r, idx.c*N_WARPS+warpid()));
}

/**
 * @brief Collaboratively store the minimum value across all devices in a PGL into a register vector.
 *
 * @tparam RV The register vector type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination register vector to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<rv<typename RV::T, N_WARPS*RV::length, typename RV::layout>>>
__device__ inline static void all_reduce_min(RV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ::kittens::ld_reduce_op<ReduceOp::MIN>(dst, src, dev_id, coord<RV>(idx.b, idx.d, idx.r, idx.c*N_WARPS+warpid()));
}

/**
 * @brief Collaboratively store the maximum value across all devices in a PGL into a register vector.
 *
 * @tparam RV The register vector type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination register vector to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<rv<typename RV::T, N_WARPS*RV::length, typename RV::layout>>>
__device__ inline static void all_reduce_max(RV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ::kittens::ld_reduce_op<ReduceOp::MAX>(dst, src, dev_id, coord<RV>(idx.b, idx.d, idx.r, idx.c*N_WARPS+warpid()));
}

/**
 * @brief Collaboratively adds data from a register vector to global memory for all devices in a PGL
 *
 * @tparam RV The register vector type.
 * @tparam PGL The parallel global layout type.
 * @tparam COORD The coordinate vector type
 * @param[out] dst The destination PGL to store the result.
 * @param[in] src The source register vector to load data from
 */
template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<rv<typename RV::T, N_WARPS*RV::length, typename RV::layout>>>
__device__ inline static void atomic_add(const PGL &dst, const RV &src, int dev_id, const COORD &idx) {
    ::kittens::reduce_op<ReduceOp::ADD>(dst, src, dev_id, coord<RV>(idx.b, idx.d, idx.r, idx.c*N_WARPS+warpid()));
}

/**
 * @brief Collaboratively stores data from a register vector to global memory for all devices in a PGL
 *
 * @tparam RV The register vector type.
 * @tparam PGL The parallel global layout type.
 * @tparam COORD The coordinate vector type
 * @param[out] dst The destination PGL to store the result.
 * @param[in] src The source register vector to load data from
 */
template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<rv<typename RV::T, N_WARPS*RV::length, typename RV::layout>>>
__device__ inline static void broadcast(const PGL &dst, const RV &src, int dev_id, const COORD &idx) {
    ::kittens::broadcast(dst, src, dev_id, coord<RV>(idx.b, idx.d, idx.r, idx.c*N_WARPS+warpid()));
}