#pragma once

#include <concepts>

/**
 * @brief Namespace for kittens project
 */
namespace kittens {
    /**
     * @brief Namespace for ducks module within kittens project
     */
    namespace ducks {
        /**
         * @brief Namespace for register tile layout configurations
         */
        namespace rt_layout {

            /**
             * @struct row
             * @brief Represents a row layout configuration for matrices
             */
            struct row { static constexpr bool is_row=true;  }; // for most matrices

            /**
             * @struct col
             * @brief Represents a column layout configuration, specifically for the B-matrix of MMA operations
             */
            struct col { static constexpr bool is_row=false; }; // for the B-matrix of MMA ops.

            /**
             * @concept all
             * @brief Concept to check if a type is either row or column layout
             * @tparam T Type to check against row or col layout
             */
            template<typename T>
            concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;

            /**
             * @struct transpose
             * @brief Represents a transposed layout configuration
             * @tparam L Layout type that is either row or col
             */
            template<all L> struct transpose      { using type = col; };

            /**
             * @struct transpose<col>
             * @brief Specialization of transpose for column layout
             */
            template<>      struct transpose<col> { using type = row; };

        } // namespace rt_layout
    } // namespace ducks
} // namespace kittens
