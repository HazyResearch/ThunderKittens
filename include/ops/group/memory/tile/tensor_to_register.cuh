/**
 * @file
 * @brief Group (collaborative warp) ops for loading tensor tiles into register tiles.
 */

template<ducks::rt::all RT, ducks::tt::all TM>
__device__ static inline void load_async(RT &dst, const TM &src) {
    static_assert(N_WARPS==4 || N_WARPS==8);
    constexpr int warp_rows = TM::rows/N_WARPS;
    static_assert(TM::cols==RT::cols);
    static_assert(warp_rows==RT::rows);
    if constexpr (N_WARPS == 4) {
        auto src_subtile = src.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*warpid(), 0);
        kittens::load_async(dst, src_subtile);
    }
    else {
        auto src_subtile = src.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*(warpid()%4)+16*(warpid()/4), 0);
        kittens::load_async(dst, src_subtile);
    }
}

template<ducks::rt::all RT, ducks::tt::all TM>
__device__ static inline void store_async(TM &dst, const RT &src) {
    static_assert(N_WARPS==4 || N_WARPS==8);
    constexpr int warp_rows = TM::rows/N_WARPS;
    static_assert(TM::cols==RT::cols);
    static_assert(warp_rows==RT::rows);
    if constexpr (N_WARPS == 4) {
        auto dst_subtile = dst.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*warpid(), 0);
        kittens::store_async(dst_subtile, src);
    }
    else {
        auto dst_subtile = dst.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*(warpid()%4)+16*(warpid()/4), 0);
        kittens::store_async(dst_subtile, src);
    }
}