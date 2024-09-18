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
template<ducks::rv::all RV, ducks::gl::all GL>
__device__ inline static void load(RV &dst, const GL &src, const index &idx) {
    // Call warp level load
    ::kittens::load(dst, src, {idx.b, idx.d, idx.r, idx.c*N_WARPS+warpid()});
}
/**
 * @brief Collaboratively stores data from register vectors to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL>
__device__ inline static void store(GL &dst, const RV &src, const index &idx) {
    // Call warp level store
    ::kittens::store(dst, src, {idx.b, idx.d, idx.r, idx.c*N_WARPS+warpid()});
}