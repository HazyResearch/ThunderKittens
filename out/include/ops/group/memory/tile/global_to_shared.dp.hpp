#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and
 * storing to global memory.
 */

template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
static inline void load(ST &dst, const GL &src, const COORD &idx) {
    kittens::load<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>> // default case
static inline void load(ST &dst, const GL &src, const COORD &idx) {
    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    kittens::store<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>> // default case
static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    kittens::store<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
static inline void load_async(ST &dst, const GL &src, const COORD &idx) {
    kittens::load_async<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>> // default case
static inline void load_async(ST &dst, const GL &src, const COORD &idx) {
    kittens::load_async<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}