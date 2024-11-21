/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and storing to global memory. 
 */

template<ducks::cst::all CST, ducks::cgl::all CGL>
__device__ static inline void load(CST &dst, const CGL &src, const coord &idx) {
    load(dst.real, src.real, idx);
    load(dst.imag, src.imag, idx);
}
template<ducks::cst::all CST, ducks::cgl::all CGL>
__device__ static inline void store(CGL &dst, const CST &src, const coord &idx) {
    load(dst.real, src.real, idx);
    load(dst.imag, src.imag, idx);
}

template<ducks::cst::all CST, ducks::cgl::all CGL>
__device__ static inline void load_async(CST &dst, const CGL &src, const coord &idx) {
    load_async(dst.real, src.real, idx);
    load_async(dst.imag, src.imag, idx);
}