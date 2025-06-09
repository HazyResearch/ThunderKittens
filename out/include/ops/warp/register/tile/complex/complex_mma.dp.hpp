/**
 * @file
 * @brief Matrix multiply-accumulate operations for complex register tiles.
 */

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../../../../../common/common.dp.hpp"
#include "../../../../../types/types.dp.hpp"

#include "../mma.dp.hpp"

namespace kittens {


/**
 * @brief Matrix multiply-accumulate operation for complex tiles
 *
 * This function calls mma_AB with hf arguments
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_cmplx_hf<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_cmplx_hf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_cmplx_hf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_cmplx_hf<N, M, row_layout> accumulator matrix.
 */
template<int N, int K, int M>
static inline void mma_AB(crt_hf<N, M, ducks::rt_layout::row> &d,
                               const crt_hf<N, K, ducks::rt_layout::row> &a,
                               const crt_hf<K, M, ducks::rt_layout::col> &b,
                               const crt_hf<N, M, ducks::rt_layout::row> &c) {
    
    // Copy data from input accumulate register into output
    copy(d.real, c.real);
    copy(d.imag, c.imag);

    // Negative on B matrix so we can use single accum register
    rt_hf<N, K, ducks::rt_layout::row> tmp;
    // Hex value for -1 in float16
    constexpr sycl::half factor = std::bit_cast<sycl::half>(uint16_t(0xFB80));
    mul(tmp, a.imag, factor);
    mma_AB(d.real, a.real, b.real, d.real);
    mma_AB(d.real, tmp, b.imag, d.real);

    mma_AB(d.imag, a.real, b.imag, d.imag);
    mma_AB(d.imag, a.imag, b.real, d.imag);
}
/**
 * @brief Matrix multiply-accumulate operation for complex tiles
 *
 * This function calls mma_AB with bf16 arguments
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_cmplx_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_cmplx_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_cmplx_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_cmplx_fl<N, M, row_layout> accumulator matrix.
 */

template<int N, int K, int M>
static inline void mma_AB(crt_fl<N, M, ducks::rt_layout::row> &d,
                               const crt_bf<N, K, ducks::rt_layout::row> &a,
                               const crt_bf<K, M, ducks::rt_layout::col> &b,
                               const crt_fl<N, M, ducks::rt_layout::row> &c) {
    
    // Copy data from input accumulate register into output
    copy(d.real, c.real);
    copy(d.imag, c.imag);

    // Negative on B matrix so we can use single accum register
    kittens::rt_bf<N, K, ducks::rt_layout::row> tmp;
    // Hex value for -1 in bf16
    constexpr bf16 factor =
        std::bit_cast<sycl::ext::oneapi::bfloat16>(uint16_t(0xBF80));
    mul(tmp, a.imag, factor);
    mma_AB(d.real, a.real, b.real, d.real);
    mma_AB(d.real, tmp, b.imag, d.real);

    mma_AB(d.imag, a.real, b.imag, d.imag);
    mma_AB(d.imag, a.imag, b.real, d.imag);
}


}