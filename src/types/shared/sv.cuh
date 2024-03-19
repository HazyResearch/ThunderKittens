#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"

namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
namespace sv {
struct identifier {};
}
}

template<typename _T, size_t _tiles>
struct sv {
    using identifier = ducks::sv::identifier;
    using dtype = _T;

    static constexpr int tiles  = _tiles;
    static constexpr int length = tiles * 16;

    dtype data[length];

    __device__ inline       dtype& operator[](size_t idx)       { return data[idx]; }
    __device__ inline const dtype& operator[](size_t idx) const { return data[idx]; }
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace sv {

template<typename T>
concept all = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is abstract_vector

} // namespace rv
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