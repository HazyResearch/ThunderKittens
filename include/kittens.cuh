/**
 * @file
 * @brief The master header file of ThunderKittens. This file includes everything you need!
 */

#pragma once

// Standard library includes
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#ifdef KITTENS_NO_HOST // useful flag for JIT compilation
namespace std { using namespace cuda::std; }
using uint = uint32_t;
struct alignas(64) CUtensorMap { char __opaque[128]; };
#endif

// CUDA type headers
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#include <cuda_fp8.h>
#endif
#if defined(KITTENS_BLACKWELL)
#include <cuda_fp4.h>
#endif

// Host-only standard library includes
#ifndef KITTENS_NO_HOST
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>
#endif

// ThunderKittens headers
#include "common/common.cuh"
#include "types/types.cuh"
#include "ops/ops.cuh"
