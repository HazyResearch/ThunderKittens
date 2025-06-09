#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/**
 * @file
 * @brief Functions for a group to collaboratively perform operations between
 * pgls and shared tiles
 */

 /**
 * @brief Collaboratively adds data together across all devices and stores the result in a shared tile.
 * 
 * @tparam ST The shared tile type.
 * @tparam PGL The parallel global layout type.
 * @param dst The destination ST to store the result.
 * @param src The source PGL to load data across devices from
 */
template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
static inline void all_reduce_add(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    kittens::ld_reduce_op<axis, assume_aligned, ReduceOp::ADD, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_idx, idx);
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
static inline void all_reduce_add(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    kittens::ld_reduce_op<2, false, ReduceOp::ADD, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_idx, idx);
}

/**
 * @brief Collaboratively store the minimum value across all devices in a PGL to a shared tile.
 * 
 * @tparam ST The shared tile type.
 * @tparam PGL The parallel global layout type.
 * @param dst The destination ST to store the result.
 * @param src The source PGL to load data across devices from
 */
template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
static inline void all_reduce_min(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    kittens::ld_reduce_op<axis, assume_aligned, ReduceOp::MIN, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_idx, idx);
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
static inline void all_reduce_min(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    kittens::ld_reduce_op<2, false, ReduceOp::MIN, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_idx, idx);
}

/**
 * @brief Collaboratively store the maximum value across all devices in a PGL to a shared tile.
 * 
 * @tparam ST The shared tile type.
 * @tparam PGL The parallel global layout type.
 * @param dst The destination ST to store the result.
 * @param src The source PGL to load data across devices from
 */
template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
static inline void all_reduce_max(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    kittens::ld_reduce_op<axis, assume_aligned, ReduceOp::MAX, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_idx, idx);
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
static inline void all_reduce_max(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    kittens::ld_reduce_op<2, false, ReduceOp::MAX, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_idx, idx);
}

/**
 * @brief Collaboratively add data from a shared tile to global memory for all devices in a PGL
 * 
 * @tparam ST The shared tile type.
 * @tparam PGL The parallel global layout type.
 * @param dst The destination PGL to store the result.
 * @param src The source ST to load data from
 */
template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
static inline void atomic_add(const PGL &dst, const ST &src, int dev_idx, const COORD &idx) {
    kittens::reduce_op<axis, assume_aligned, ReduceOp::ADD, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_idx, idx);
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
static inline void atomic_add(const PGL &dst, const ST &src, int dev_idx, const COORD &idx) {
    kittens::reduce_op<2, false, ReduceOp::ADD, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_idx, idx);
}

/**
 * @brief Collaboratively store data from a shared tile to global memory for all devices in a PGL
 * 
 * @tparam ST The shared tile type.
 * @tparam PGL The parallel global layout type.
 * @param dst The destination PGL to store the result.
 * @param src The source ST to load data from
 */
template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
static inline void broadcast(const PGL &dst, const ST &src, int dev_idx, const COORD &idx) {
    kittens::broadcast<axis, assume_aligned, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_idx, idx);
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
static inline void broadcast(const PGL &dst, const ST &src, int dev_idx, const COORD &idx) {
    kittens::broadcast<2, false, ST, PGL, COORD, GROUP_THREADS>(dst, src, dev_idx, idx);
}
