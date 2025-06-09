/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../../../../../common/common.dp.hpp"
#include "../../../../../types/types.dp.hpp"

#include "../global_to_shared.dp.hpp"

namespace kittens {

/**
 * @brief Loads data from global memory into a complex shared memory tile.
 *
 * @tparam CST The complex shared tile type.
 * @tparam CGL The complex global array type.
 * @tparam COORD The coordinate type.
 * @param[out] dst The destination complex shared memory tile.
 * @param[in] src The source complex global memory array.
 * @param[in] idx The coordinate of the tile in the global memory array.
 */
template<int axis, bool assume_aligned, ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
static inline void load(CST &dst, const CGL &src, const COORD &idx) {
    load<axis, assume_aligned, typename CST::component, typename CGL::component>(dst.real, src.real, idx);
    load<axis, assume_aligned, typename CST::component, typename CGL::component>(dst.imag, src.imag, idx);
}
template<ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
static inline void load(CST &dst, const CGL &src, const COORD &idx) {
    load<2, false, CST, CGL, COORD>(dst, src, idx);
}

/**
 * @brief Stores data from a complex shared memory tile into global memory.
 *
 * @tparam CST The complex shared tile type.
 * @tparam CGL The complex global array type.
 * @tparam COORD The coordinate type.
 * @param[out] dst The destination complex global memory array.
 * @param[in] src The source complex shared memory tile.
 * @param[in] idx The coordinate of the tile in the global memory array.
 */
template<int axis, bool assume_aligned, ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
static inline void store(const CGL &dst, CST &src, const COORD &idx) {
    store<axis, assume_aligned, typename CST::component, typename CGL::component>(dst.real, src.real, idx);
    store<axis, assume_aligned, typename CST::component, typename CGL::component>(dst.imag, src.imag, idx);
}
template<ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
static inline void store(const CGL &dst, CST &src, const COORD &idx) {
    store<2, false, CST, CGL, COORD>(dst, src, idx);
}

/**
 * @brief Asynchronously loads data from global memory into a complex shared memory tile.
 *
 * @tparam CST The complex shared tile type.
 * @tparam CGL The complex global array type.
 * @tparam COORD The coordinate type.
 * @param[out] dst The destination complex shared memory tile.
 * @param[in] src The source complex global memory array.
 * @param[in] idx The coordinate of the tile in the global memory array.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is undefined.
 */
template<int axis, bool assume_aligned, ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
static inline void load_async(CST &dst, const CGL &src, const COORD &idx) {
    load_async<axis, assume_aligned, typename CST::component, typename CGL::component>(dst.real, src.real, idx);
    load_async<axis, assume_aligned, typename CST::component, typename CGL::component>(dst.imag, src.imag, idx);
}
template<ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
static inline void load_async(CST &dst, const CGL &src, const COORD &idx) {
    load_async<2, false, CST, CGL, COORD>(dst, src, idx);
}
}