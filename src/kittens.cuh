/**
 * @file kittens.cuh
 * @brief Central header including common modules for the ThunderKittens project.
 *
 * This file serves as a central inclusion point for common headers used throughout
 * the ThunderKittens project. It includes headers for common utilities, type definitions,
 * and operation modules, ensuring that all necessary components are available for files
 * that include it.
 */

#pragma once

#include "common/common.cuh"  // Common utilities and helper functions.
#include "types/types.cuh"    // Type definitions and abstractions for the project.
#include "ops/ops.cuh"        // Operation modules including warp, warpgroup, and block operations.
