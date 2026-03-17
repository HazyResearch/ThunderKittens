/**
 * @file
 * @brief An aggregate header file for all the system-wide types defined by ThunderKittens.
 */

#pragma once

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#ifndef KITTENS_NO_HOST
#include "ipc.cuh"
#include "vmm.cuh"
#endif
#include "pgl.cuh"
#endif
