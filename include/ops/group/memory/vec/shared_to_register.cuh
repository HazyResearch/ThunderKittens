/**
 * @file
 * @brief Functions for a group to collaboratively transfer data directly between shared memory and registers and back.
 */

/**
 * @brief Collaboratively load data from a shared vector into register vectors split across a warpgroup.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
template<ducks::rv::all RV, ducks::sv::all SV>
__device__ inline static void load(RV &dst, const SV &_src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(_src.length == dst.length*GROUP_WARPS);// confirm size correct
    auto &src = subvec_inplace<dst.length>(_src, warpid()); // pretend it's smaller and do warp-level load

    ::kittens::load(dst, src); // warp-level
}

/**
 * @brief Collaboratively store data into a shared vector from register vectors split across a warpgroup.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination shared vector.
 * @param src[in]  The source register vector.
 */
template<ducks::sv::all SV, ducks::rv::all RV>
__device__ inline static void store(SV &_dst, const RV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(_dst.length == src.length*GROUP_WARPS);// confirm size correct
    auto &dst = subvec_inplace<src.length>(_dst, warpid()); // pretend it's smaller and do warp-level load

    ::kittens::store(dst, src); // warp-level
}