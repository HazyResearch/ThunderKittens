#pragma once

/**
 * @file warp.cuh
 * @brief Wrapper for warp-level operations.
 *
 * This header includes all the necessary components for warp-level operations,
 * which are the default operation scope within this context. It aggregates
 * register, shared, and memory operation headers.
 */

// no namespace wrapper needed here
// as warp is the default op scope!

#include "register/register.cuh"
#include "shared/shared.cuh"
#include "memory/memory.cuh"
