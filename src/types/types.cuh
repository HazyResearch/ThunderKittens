#pragma once

#include "register/register.cuh"
#include "shared/shared.cuh"

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// The point of these two wrappers is to enable the following syntax:
// kittens::col_vec<decltype(some_tile)> tile_accumulator;
template<typename T> using row_vec = T::row_vec;
template<typename T> using col_vec = T::col_vec;

// ^ this code lives here because it applies to both sv and rv types
