#pragma once

// it's sometimes helpful for warpgroup functions to be able to access warp functions.
#include "../warp/warp.cuh"

// All of these files should include the "warpgroup" namespace.

#ifdef KITTENS_HOPPER
#include "wgmma/wgmma.cuh"
#include "shared/shared.cuh"
#endif
#include "memory/memory.cuh"