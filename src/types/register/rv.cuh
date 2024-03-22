#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"
#include "rt_layout.cuh"

/**
 * @brief Namespace for all kitten-related data structures and operations.
 */
namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

// helper struct for type inference
/**
 * @brief Namespace for duck-related data structures within the kittens namespace.
 */
namespace ducks {
/**
 * @brief Namespace for rv-related data structures within the ducks namespace.
 */
namespace rv {
/**
 * @brief Identifier struct for rv namespace, used for type inference.
 */
struct identifier {};
}
}

/**
 * @brief Represents a register vector with customizable data type and dimensions.
 * 
 * @tparam _T Data type of the elements in the vector.
 * @tparam _outer_dim Number of elements in the outer dimension.
 * @tparam _inner_dim Number of elements in the inner dimension (default is 1).
 */
template<typename _T, size_t _outer_dim, size_t _inner_dim=1>
struct rv {
    using identifier = ducks::rv::identifier; ///< Identifier for the rv struct.
    using dtype = _T; ///< Data type of the elements.

    static constexpr int outer_dim = _outer_dim; ///< Outer dimension size.
    static constexpr int inner_dim = _inner_dim; ///< Inner dimension size.

    dtype data[outer_dim][inner_dim]; ///< Storage for the vector data.

    /**
     * @brief Access operator to get a pointer to the row at the given index.
     * 
     * @param idx Index of the row.
     * @return Pointer to the row.
     */
    __device__ inline       dtype* operator[](size_t idx)       { return &data[idx][0]; }

    /**
     * @brief Access operator to get a const pointer to the row at the given index.
     * 
     * @param idx Index of the row.
     * @return Const pointer to the row.
     */
    __device__ inline const dtype* operator[](size_t idx) const { return &data[idx][0]; }

    /**
     * @brief Access operator to get a reference to the element at the given 2D index.
     * 
     * @param outin 2D index of the element.
     * @return Reference to the element.
     */
    __device__ inline       dtype& operator[](int2 outin)       { return data[outin.x][outin.y]; }

    /**
     * @brief Access operator to get a const reference to the element at the given 2D index.
     * 
     * @param outin 2D index of the element.
     * @return Const reference to the element.
     */
    __device__ inline const dtype& operator[](int2 outin) const { return data[outin.x][outin.y]; }
};

/* ----------  TEMPLATE SPECIALIZATIONS  ---------- */

/**
 * @brief Specialization of rv for float data type with 1x1 dimension.
 * @details Utilizes 1 register.
 */
template<> struct rv<float, 1, 1> {
    // ... (implementation details)
};

/**
 * @brief Specialization of rv for float data type with 2x2 dimension.
 * @details Utilizes 4 registers.
 */
template<> struct rv<float, 2, 2> {
    // ... (implementation details)
};

// ... (additional specializations)

/* ----------  CONCEPTS  ---------- */

/**
 * @brief Namespace for duck-related concepts within the kittens namespace.
 */
namespace ducks {
/**
 * @brief Namespace for rv-related concepts within the ducks namespace.
 */
namespace rv {

/**
 * @brief Concept to check if a type belongs to the rv namespace.
 * 
 * @tparam T Type to check.
 */
template<typename T>
concept all = requires {
    typename T::identifier; // Checks if T::vector_identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is abstract_vector

} // namespace rv
} // namespace ducks

} // namespace kittens
