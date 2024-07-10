/**
 * @file
 * @brief Functions for a warpgroup to collaboratively transfer  data directly between global memory and registers and back.
 */

/**
 * @brief Collaboratively loads data into register vectors from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<ducks::rv::all RV, typename U>
__device__ inline static void load(RV &dst, const U *_src) {
    using T2 = RV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;
    
    const U *src = &_src[warpid() * dst.outer_dim * kittens::TILE_DIM]; // pretend smaller, do single warp load.
    
    // Call warp level store
    ::kittens::load(dst, src);
}
/**
 * @brief Collaboratively stores data from register vectors to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */
template<ducks::rv::all RV, typename U>
__device__ inline static void store(U *_dst, const RV &src) {
    using T2 = RV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;
    
    U *dst = &_dst[warpid() * src.outer_dim * kittens::TILE_DIM]; // pretend smaller, do single warp store.

    // Call warp level store
    ::kittens::store(dst, src);
}