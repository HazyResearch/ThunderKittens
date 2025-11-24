/**
 * @file
 * @brief Single-threaded ops for loading shared memory into tensor tiles
 */

template<kittens::ducks::tt::full TT, kittens::ducks::sv::all SV>
__device__ inline static void load_mxnv_scale_async(TT &dst, const SV &src) {
    static_assert(std::is_same_v<typename TT::T, kittens::fp8e8m0> || std::is_same_v<typename TT::T, kittens::fp8e4m3>, "Scale TT must be fp8e8m0 or fp8e4m3");
    static_assert(std::is_same_v<typename TT::T, typename SV::T>, "Scale TT and SV must have the same type");
    static_assert(TT::rows == kittens::MAX_TENSOR_ROWS, "Scale TT must be full");
    static_assert(TT::cols == 16, "Scale TT must have 16 columns");
    static_assert(SV::length == TT::cols * 32, "Scale SV has incorrect length. Must have sufficient elements to fill in 32x16 scale layout");

    uint64_t sv_desc = kittens::detail::matrix_descriptor_raw(&src.data[0], 128, 128, 0);
    asm volatile("{tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;}" :: "r"(dst.addr), "l"(sv_desc));
}
