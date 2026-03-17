/**
 * @file
 * @brief Group conversions between different shared memory tile types.
 */

/* ----------  COPIES  ---------- */

template<ducks::st::all ST1, ducks::st::all ST2>
__device__ static inline void copy(ST1 &dst, const ST2 &src) {
    static_assert(ST1::rows == ST2::rows && ST1::cols == ST2::cols, "Tiles must have the same rows and cols");
    #pragma unroll
    for(int i = laneid(); i < ST1::num_elements; i+=GROUP_THREADS) {
        int row = i/ST1::cols, col = i%ST1::cols;
        dst[{row, col}] = base_types::convertor<typename ST1::dtype, typename ST2::dtype>::convert(src[{row, col}]);
    }
}