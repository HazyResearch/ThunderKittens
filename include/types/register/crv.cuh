/**
 * @file
 * @brief Register vectors for computations on axes.
 */

#pragma once

#include "../../common/common.cuh"
#include "rv.cuh"

namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
/**
 * @namespace crv
 *
 * @brief The namespace where concepts and abstract types for complex register vectors live.
 */
namespace crv {
/**
 * @brief A dummy type used to identify complex register vectors.
 *
 * For a type to quack like a crv, it should define its identifier as ducks::crv::identifier.
 * If a type quacks like ducks::crv::identifier, it will be treated as a crv by compiler checks.
 */
struct identifier {};
/**
* @brief Concept for all complex register vectors.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as crv::identifier.
*/
template<typename T>
concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::crv::identifier.

template<typename T> concept naive_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::naive>;
template<typename T> concept align_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::align>;
template<typename T> concept ortho_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::ortho>;
template<typename T> concept tile_layout  = align_layout<T> || ortho_layout<T>; // vector layouts for interacting with tiles.
}
}
/**
 * @brief Register vector structure.
 *
 * @tparam _T The packed data type used for the vector elements.
 * @tparam _length The length of the vector, in units of TILE_DIM (16).
 * @tparam _layout This controls the layout of the vector in terms of which axis it maps on the register tile layout.
 *
 * Register vectors are used to accumulate and map values across tiles. You can do computation
 * on them directly if you want, but they're not designed to be maximally efficient vectors
 * as they have substantial duplication and strange layouts to help them work efficiently with
 * the register layouts used by the tensor cores. ThunderKittens wants you working with tiles
 * where possible!
 */

template<typename _T, size_t _length, ducks::rv_layout::all _layout=ducks::rv_layout::naive>
struct crv {
    using identifier = ducks::crv::identifier;
    using component  = rv<_T, _length, _layout>; /// Data type of each internal tile.
    using layout     = component::layout; ///< Layout of the matrix tile, ensures compatibility with the rv concepts
    
    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype; ///< Data type of the elements in the tile.

    static constexpr int length     = component::length;
    static constexpr int tiles      = component::tiles;

    // Real/imag tiles have same internal layout and size
    component real;
    component imag;
};


template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using crv_fl = crv<float, _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using crv_bf = crv<bf16,  _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using crv_hf = crv<half,  _l, layout>;

} // namespace kittens