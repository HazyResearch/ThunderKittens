/**
 * @file
 * @brief Group (collaborative warp) ops for loading tensor tiles into register tiles.
 */

template<ducks::rt::all RT, ducks::tmem::all TM>
__device__ static inline void load(RT &dst, const TM &src) {
    static_assert(N_WARPS==2 || N_WARPS%4==0);
    constexpr int warp_rows = TM::height/N_WARPS;
    static_assert(TM::cols==RT::cols);
    static_assert(warp_rows==RT::rows);
    auto src_subtile = src.template subtile<tmem<typename TM::dtype, warp_rows, TM::cols>>(warp_rows*(warpid()/4+(warpid()%4)*(N_WARPS/4)), 0);
    kittens::load(dst, src_subtile);
}

template<ducks::rt::all RT, ducks::tmem::all TM>
__device__ static inline void store(TM &dst, const RT &src) {
    static_assert(N_WARPS==2 || N_WARPS%4==0);
    constexpr int warp_rows = TM::height/N_WARPS;
    static_assert(TM::cols==RT::cols);
    static_assert(warp_rows==RT::rows);
    auto dst_subtile = dst.template subtile<tmem<typename TM::dtype, warp_rows, TM::cols>>(warp_rows*(warpid()/4+(warpid()%4)*(N_WARPS/4)), 0);
    kittens::store(dst_subtile, src);
}