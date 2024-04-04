/**
 * @file
 * @brief The ThunderKittens shared vector struct.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"

namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

namespace ducks {
/**
 * @namespace sv
 * 
 * @brief The namespace where concepts and abstract types for shared vectors live.
 */
namespace sv {
/**
 * @brief A dummy type used to identify shared vectors.
 * 
 * For a type to quack like an sv, it should define its identifier as ducks::sv::identifier.
 * If a type quacks like ducks::sv::identifier, it will be treated as an sv by compiler checks.
 */
struct identifier {};
}
}

/**
 * @brief Shared vector structure.
 *
 * @tparam _T The packed data type used for the vector elements.
 * @tparam _tiles The size of the tile, in units of TILE_DIM (16).
 *
 * Shared vectors are used to accumulate and map values across shared tiles.
 * Unlike every other structure present in ThunderKittens, these have a simple
 * uniform layout which is just an array in memory. EZ!
 */
template<typename _T, size_t _tiles>
struct sv {
    using identifier = ducks::sv::identifier;
    using dtype = _T;

    static constexpr int tiles  = _tiles; ///< Length in subtiles.
    static constexpr int length = tiles * 16; ///< Length in elements.

    dtype data[length]; ///< The actual shared vector data.

    __device__ inline       dtype& operator[](size_t idx)       { return data[idx]; }
    __device__ inline const dtype& operator[](size_t idx) const { return data[idx]; }

    template<size_t sub_tiles> using subvec = sv<dtype, sub_tiles>; ///< A subvector which allows warpgroups and blocks to work cooperatively.
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace sv {
/**
* @brief Concept for all shared vectors.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as sv::identifier.
*/
template<typename T>
concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::sv::identifier

} // namespace sv
} // namespace ducks


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// vector types
using sv_bf_1 = sv<bf16 ,1>;
using sv_bf_2 = sv<bf16 ,2>;
using sv_bf_4 = sv<bf16 ,4>;
using sv_bf_8 = sv<bf16 ,8>;
using sv_fl_1 = sv<float,1>;
using sv_fl_2 = sv<float,2>;
using sv_fl_4 = sv<float,4>;
using sv_fl_8 = sv<float,8>;

} // namespace kittens