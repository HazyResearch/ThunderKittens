#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"
#include "rt_layout.cuh"

namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
namespace rv {
struct identifier {};
}
}

template<typename _T, size_t _outer_dim, size_t _inner_dim=1>
struct rv {
    using identifier = ducks::rv::identifier;
    using dtype = _T;

    static constexpr int outer_dim = _outer_dim;
    static constexpr int inner_dim = _inner_dim;

    dtype data[outer_dim][inner_dim];

    __device__ inline       dtype* operator[](size_t idx)       { return &data[idx][0]; }
    __device__ inline const dtype* operator[](size_t idx) const { return &data[idx][0]; }
    __device__ inline       dtype& operator[](int2 outin)       { return data[outin.x][outin.y]; }
    __device__ inline const dtype& operator[](int2 outin) const { return data[outin.x][outin.y]; }
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rv {

template<typename T>
concept all = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is abstract_vector

} // namespace rv
} // namespace ducks

} // namespace kittens