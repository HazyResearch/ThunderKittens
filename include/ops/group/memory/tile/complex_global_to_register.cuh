/**
 * @file
 * @brief Functions for a group to collaboratively transfer data directly between global memory and registers and back.
 */

/**
 * @brief Collaboratively loads data from a source array into register tiles.
 *
 * @tparam CRT The complex register tile type.
 * @tparam CGL The complex global layout type.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source global layout to load data from.
 * @param idx[in] The coordinate of the tile to load.
 */
template<int axis, ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void load(CRT &dst, const CGL &src, const COORD &idx) {
    load<axis, CRT::component, CGL::component, COORD>(dst.real, src.real, idx);
    load<axis, CRT::component, CGL::component, COORD>(dst.imag, src.imag, idx);
}
template<ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void load(CRT &dst, const CGL &src, const COORD &idx) {
    load<2, CRT, CGL>(dst, src, idx);
}

/**
 * @brief Collaboratively stores data from register tiles to a destination array in global memory.
 *
 * @tparam CRT The complex register tile type.
 * @tparam CGL The complex global layout type.
 * @param[out] dst The destination global layout to store data into.
 * @param[in] src The source complex register tile to store data from.
 * @param[in] idx The coordinate of the tile to store.
 */
template<int axis, ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void store(CGL &dst, const CRT &src, const COORD &idx) {
    store<axis, typename CRT::component, typename CGL::component>(dst.real, src.real, idx);
    store<axis, typename CRT::component, typename CGL::component>(dst.imag, src.imag, idx);
}
template<ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void store(CGL &dst, const CRT &src, const COORD &idx) {
    store<2, CRT, CGL>(dst, src, idx);
}
