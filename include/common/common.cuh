/**
 * @file
 * @brief A collection of common resources on which ThunderKittens depends.
 */
 

#pragma once

#include "base_types.cuh"
#include "base_ops.cuh"
#include "util.cuh"

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#include "multimem.cuh"
#endif
