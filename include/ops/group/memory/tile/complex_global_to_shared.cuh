/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and storing to global memory. 
 */

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
__device__ static inline void load(CST &dst, const CGL &src, const COORD &idx) {
    load<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}
template<ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void load(CST &dst, const CGL &src, const COORD &idx) {
    load<2, false, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load<2, false, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
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
__device__ static inline void store(CGL &dst, const CST &src, const COORD &idx) {
    store<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    store<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}
template<ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void store(CGL &dst, const CST &src, const COORD &idx) {
    store<2, false, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    store<2, false, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
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
__device__ static inline void load_async(CST &dst, const CGL &src, const COORD &idx) {
    load_async<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load_async<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}
template<ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void load_async(CST &dst, const CGL &src, const COORD &idx) {
    load_async<2, false, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load_async<2, false, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}
