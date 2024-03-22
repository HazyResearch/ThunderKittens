#pragma once

#include "register/register.cuh"
#include "shared/shared.cuh"

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

/**
 * @file types.cuh
 * @brief Wrappers for vector types to enable cleaner syntax in kernel code.
 */

/**
 * @brief Wrapper to enable syntax for creating a row vector from a tile type.
 * 
 * This wrapper allows the following syntax:
 * kittens::row_vec<decltype(some_tile)> tile_accumulator;
 * 
 * @tparam T The tile type from which to create a row vector.
 */
template<typename T> using row_vec = T::row_vec;

/**
 * @brief Wrapper to enable syntax for creating a column vector from a tile type.
 * 
 * This wrapper allows the following syntax:
 * kittens::col_vec<decltype(some_tile)> tile_accumulator;
 * 
 * @tparam T The tile type from which to create a column vector.
 */
template<typename T> using col_vec = T::col_vec;

// ^ this code lives here because it applies to both sv and rv types
