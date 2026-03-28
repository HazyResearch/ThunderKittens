/**
 * @file
 * @brief Abstraction for a complex shared vector composed of real and imaginary vectors
 */
 
#pragma once

#include "sv.cuh"

namespace kittens {

namespace ducks {
namespace csv {
/**
 * @brief A dummy type used to identify complex shared vectors.
 *
 * For a type to quack like a csv, it should define its identifier as ducks::csv::identifier.
 * If a type quacks like ducks::csv::identifier, it will be treated as a csv by compiler checks.
 */
struct identifier {};
/**
* @brief Concept for shared vectors that are complex.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a shared vector.
* - T has a complex shared vector identifier.
*/
template <typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier> && ducks::sv::all<typename T::component>;

} // namespace csv
} // namespace ducks

/**
 * @brief Complex shared vector structure
 *
 * @tparam _T The data type used for the vector elements.
 * @tparam _length The length of the vector.
 *
 * This structure is designed to abstract complex number operations internally to the real and imaginary
 * shared vectors, respectively
 * 
 *
 */
template<typename _T, int _length>
struct csv {
    using identifier = ducks::csv::identifier;
    using component  = sv<_T, _length>; /// Data type of each internal tile.
    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype; ///< Data type of the elements in the tile.

    static constexpr int length     = component::length;
    static constexpr int tiles      = component::tiles;

    // todo: fill in the rest for convenience, but they're all accessible via component so it's not urgent.

    // Real/imag tiles have same internal layout and size
    component real;
    component imag;
};


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<int _length> using csv_bf = csv<bf16,  _length>;
template<int _length> using csv_hf = csv<half,  _length>;
template<int _length> using csv_fl = csv<float, _length>;

}