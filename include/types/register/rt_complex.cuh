/**
 * @file
 * @brief Abstraction for a complex register tile composed of real and imaginary tiles
 */
 
#pragma once

#include "rt.cuh"

namespace kittens {

namespace ducks {
namespace rt {
/**
 * @brief A dummy type used to identify complex register tiles.
 * 
 * For a type to quack like an rt_cmplx, it should define its identifier as ducks::rt::cmplx_identifier.
 * If a type quacks like ducks::rt::cmplx_identifier, it will be treated as an rt_cmplx by compiler checks.
 */
struct cmplx_identifier {};
} // namespace rt
} // namespace ducks

/**
 * @brief Complex tile structure
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _height The height of the tile in terms of the number of subtiles.
 * @tparam _width The width of the tile in terms of the number of subtiles.
 * @tparam _layout The layout of the internal register tiles, either row-major or column-major.
 *
 * This structure is designed to abstract complex number operations internally to the real and imaginary
 * register tiles, respectively
 * 
 * In general, you probably want a row-major tile, unless you specifically want to call mma
 */
template<typename _T, int _height, int _width, ducks::rt_layout::all _layout=ducks::rt_layout::row>
struct rt_cmplx {
    using identifier = ducks::rt::cmplx_identifier;
    using layout = _layout; ///< Layout of the matrix tile, ensures compatibility with the rt concepts
    using dtype = rt<_T, _height, _width, layout>; /// Data type of each internal tile.

    // Real/imag tiles have same internal layout and size
    dtype real;
    dtype imag;
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rt {

/**
* @brief Concept for register tiles that are complex.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a register tile.
* - T has a complex tile identifier.
*/
template <typename T> concept complex = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, cmplx_identifier> && ducks::rt::all<typename T::dtype>;

}
}

template<int _height, int _width, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_cmplx_fl = rt_cmplx<float, _height, _width, layout>;
template<int _height, int _width, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_cmplx_bf = rt_cmplx<bf16, _height, _width, layout>;
template<int _height, int _width, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_cmplx_hf = rt_cmplx<half, _height, _width, layout>;



}