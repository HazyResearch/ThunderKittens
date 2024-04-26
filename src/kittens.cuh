/**
 * @file
 * @brief The master header file of ThunderKittens. This file includes everything you need!
 */

#pragma once

#include "common/common.cuh"
#include "types/types.cuh"
#include "ops/ops.cuh"



/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// Lifting a fwe really commonly used parts of the hierarchy
// up to the main namespace to make user code more concise.

namespace kittens {

using row_l = ducks::rt_layout::row;
using col_l = ducks::rt_layout::col;

using naive_l            = ducks::st_layout::naive;
using swizzle_l          = ducks::st_layout::swizzle;
using wgmma_swizzle_l    = ducks::st_layout::wgmma_swizzle;
using wgmma_interleave_l = ducks::st_layout::wgmma_interleave;

using warpgroup = group<4>; // special scope commonly used by SM_90 and later.

}