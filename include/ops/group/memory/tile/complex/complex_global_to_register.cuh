/**
 * @file
 * @brief Functions for a group to collaboratively transfer data directly between global memory and registers and back.
 */

/**
 * @brief Collaboratively loads data from a source array into row-major layout tiles.
 *
 * @tparam RT The row-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<int axis, ducks::crt::row_layout CRT, ducks::cgl::all CGL>
__device__ inline static void load(CRT &dst, const CGL &src, const coord<CRT> &idx) {
    load<axis, CRT::component, CGL::component>(dst.real, src.real, coord<typename CRT::component>(idx));
    load<axis, CRT::component, CGL::component>(dst.imag, src.imag, coord<typename CRT::component>(idx));
}
template<ducks::crt::row_layout CRT, ducks::cgl::all CGL>
__device__ inline static void load(CRT &dst, const CGL &src, const coord<CRT> &idx) {
    load<2, CRT, CGL>(dst, src, idx);
}

/**
 * @brief Collaboratively loads data from a source array into column-major layout tiles.
 *
 * @tparam RT The column-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<int axis, ducks::crt::col_layout CRT, ducks::cgl::all CGL>
__device__ inline static void load(CRT &dst, const CGL &src, const coord<CRT> &idx) {
    load<axis, typename CRT::component, typename CGL::component>(dst.real, src.real, coord<typename CRT::component>(idx));
    load<axis, typename CRT::component, typename CGL::component>(dst.imag, src.imag, coord<typename CRT::component>(idx));
}
template<ducks::crt::col_layout CRT, ducks::cgl::all CGL>
__device__ inline static void load(CRT &dst, const CGL &src, const coord<CRT> &idx) {
    load<2, CRT, CGL>(dst, src, idx);
}


/**
 * @brief Collaboratively stores data from register tiles to a destination array in global memory with a row-major layout.
 *
 * @tparam RT The register tile type with a row-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<ducks::crt::row_layout RT, ducks::cgl::all GL>
__device__ inline static void store(GL &dst, const RT &src, const coord &idx) {
    store(dst.real, src.real, idx);
    store(dst.imag, src.imag, idx);
}

/**
 * @brief Collaboratively stores data from register tiles to a destination array in global memory with a column-major layout.
 *
 * @tparam RT The register tile type with a column-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<int axis, ducks::crt::col_layout CRT, ducks::cgl::all CGL>
__device__ inline static void store(CGL &dst, const CRT &src, const coord<CRT> &idx) {
    store<axis, typename CRT::component, typename CGL::component>(dst.real, src.real, idx);
    store<axis, typename CRT::component, typename CGL::component>(dst.imag, src.imag, idx);
}
template<ducks::crt::col_layout CRT, ducks::cgl::all CGL>
__device__ inline static void store(CGL &dst, const CRT &src, const coord<CRT> &idx) {
    store<2, CRT, CGL>(dst, src, idx);
}
