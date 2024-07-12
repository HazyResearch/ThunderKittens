/**
 * @file
 * @brief Register vectors for computations on axes.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"
#include "rt_layout.cuh"

namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
/**
 * @namespace rt
 * 
 * @brief The namespace where concepts and abstract types for register vectors live.
 */
namespace rv {
/**
 * @brief A dummy type used to identify register vectors.
 * 
 * For a type to quack like an rv, it should define its identifier as ducks::rv::identifier.
 * If a type quacks like ducks::rv::identifier, it will be treated as an rv by compiler checks.
 */
struct identifier {};
}
}
/**
 * @brief Register vector structure.
 *
 * @tparam _T The packed data type used for the vector elements.
 * @tparam _outer_dim The size of the tile, in units of TILE_DIM (16).
 * @tparam _inner_dim This controls the layout of the tile in terms of which axis it maps on the register tile layout.
 *
 * Register vectors are used to accumulate and map values across tiles. You can do computation
 * on them directly if you want, but they're not designed to be maximally efficient vectors
 * as they have substantial duplication and strange layouts to help them work efficiently with
 * the register layouts used by the tensor cores. ThunderKittens wants you working with tiles
 * where possible!
 */
template<typename T, size_t _outer_dim, size_t _inner_dim=1>
struct rv {
    using identifier = ducks::rv::identifier; ///< Type identifier for the rv structure.
    using dtype = kittens::base_types::packing<T>::packed_type; ///< Data type of the vector elements.

    static constexpr int outer_dim = _outer_dim; ///< Length in subtiles.
    static constexpr int inner_dim = _inner_dim; ///< Internal layout within a subtile. Either 1 or 2.

    dtype data[outer_dim][inner_dim]; ///< The actual register vector data.

    __device__ inline       dtype* operator[](size_t idx)       { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    __device__ inline const dtype* operator[](size_t idx) const { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    __device__ inline       dtype& operator[](int2 outin)       { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
    __device__ inline const dtype& operator[](int2 outin) const { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rv {
/**
* @brief Concept for all register vectors.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as rv::identifier.
*/
template<typename T>
concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rv::identifier.

} // namespace rv
} // namespace ducks



// No rv_fl, rv_bf, rv_hf, etc, because we rv's are tricky enough that it's better to encourage initialization as row_vec<rt_bf<4,4>> / similar.



} // namespace kittens