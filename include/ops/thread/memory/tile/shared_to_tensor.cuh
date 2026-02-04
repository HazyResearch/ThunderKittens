/**
 * @file
 * @brief Single-threaded ops for loading shared memory into tensor tiles
 */

#include "../../mma/tcgen05.cuh"

template<int ncta=1, kittens::ducks::tt::full TT, kittens::ducks::st::all ST>
__device__ inline static void load_mxnv_scale_async(TT &dst, const ST &src) {
    static_assert(ncta == 1 || ncta == 2, "ncta must be 1 or 2");
    static_assert(std::is_same_v<typename TT::T, kittens::fp8e8m0> || std::is_same_v<typename TT::T, kittens::fp8e4m3>, "Scale TT must be fp8e8m0 or fp8e4m3");
    static_assert(std::is_same_v<typename TT::T, typename ST::T>, "Scale TT and ST must have the same type");
    static_assert(TT::rows == 128 && TT::cols == 16, "Tensor memory scale tile must always be 128x16");
    static_assert(ST::rows == 32 && ST::cols == 16, "Shared memory scale tile must always be 32x16");
    static_assert(!ST::swizzle, "Shared memory scale tile must not be TMA-swizzled");

    uint64_t st_desc = kittens::detail::matrix_descriptor_raw(&src.data[0], 128, 128, 0);

    if constexpr (ncta == 1) {
        asm volatile("{tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}" :: "r"(dst.addr), "l"(st_desc));
    } else {
        asm volatile("{tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;}" :: "r"(dst.addr), "l"(st_desc));
    }
}
template<int ncta=1, kittens::ducks::tt::full TT, kittens::ducks::st::all ST>
__device__ inline static void load_mxnv_scale_async(TT &dst, const ST &src, kittens::semaphore &sem) {
    load_mxnv_scale_async<ncta>(dst, src);
    kittens::detail::tcgen05::commit<ncta>(sem);
}
template<kittens::ducks::tt::full TT, kittens::ducks::st::all ST>
__device__ inline static void load_mxnv_scale_async2(TT &dst, const ST &src) {
    load_mxnv_scale_async<2>(dst, src);
}
template<kittens::ducks::tt::full TT, kittens::ducks::st::all ST>
__device__ inline static void load_mxnv_scale_async2(TT &dst, const ST &src, kittens::semaphore &sem) {
    load_mxnv_scale_async<2>(dst, src, sem);
}
