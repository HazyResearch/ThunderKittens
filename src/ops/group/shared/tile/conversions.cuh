/**
 * @file
 * @brief Group conversions between different shared memory tile types.
 */

/* ----------  COPIES  ---------- */

/**
 * @brief Copies data from one shared memory tile to another, potentially with different data types and layouts.
 *
 * @tparam T The data type of the destination tile.
 * @tparam U The data type of the source tile.
 * @tparam _height The height of the tile.
 * @tparam _width The width of the tile.
 * @tparam L1 The layout of the destination tile.
 * @tparam L2 The layout of the source tile.
 * @param[out] dst The destination tile.
 * @param[in] src The source tile.
 */
template<typename T, typename U, int _height, int _width, ducks::st_layout::all L1, ducks::st_layout::all L2>
__device__ static inline void copy(st<T, _height, _width, L1> &dst, const st<U, _height, _width, L2> &src) {
    if constexpr (std::is_same_v<L1, L2>) { // if same layout can just do a simple copy
        #pragma unroll
        for(int i = laneid(); i < dst.num_elements; i+=GROUP_THREADS) {
            dst[i] = base_types::convertor<T, U>::convert(src[i]);
        }
    }
    else { // otherwise we need to actually do indexing calculations :(
        #pragma unroll
        for(int i = laneid(); i < dst.num_elements; i+=GROUP_THREADS) {
            int row = i/dst.cols;
            int col = i%dst.cols;
            dst[{row, col}] = base_types::convertor<T, U>::convert(src[{row, col}]);
        }
    }
}