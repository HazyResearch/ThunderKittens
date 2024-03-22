/**
 * @file ops.cuh
 * @brief Header file that includes all operation-related headers.
 *
 * This file acts as an aggregate header to include all the operation
 * categories like warp, warpgroup, and block operations. It simplifies
 * the inclusion process for files that require access to multiple
 * operation-related functionalities.
 */

#pragma once

#include "warp/warp.cuh"
#include "warpgroup/warpgroup.cuh"
#include "block/block.cuh"
