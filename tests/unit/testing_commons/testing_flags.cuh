/* ----------  TEST INCLUSION MACROS  ---------- */

/* -----  DEPTH 0 MACROS  ----- */

// Catchall macros
#ifdef TEST_ALL
#define TEST_ALL_THREAD
#define TEST_ALL_GROUP
#endif

/* -----  DEPTH 1 MACROS  ----- */

// Warp macros
#ifdef TEST_ALL_THREAD
#define TEST_ALL_THREAD_MEMORY
#endif

// Group macros
#ifdef TEST_ALL_GROUP
#define TEST_ALL_GROUP_MEMORY
#define TEST_ALL_GROUP_SHARED
#define TEST_ALL_GROUP_REG
#define TEST_ALL_GROUP_MMA // leaf
#endif

/* -----  DEPTH 2 MACROS  ----- */

// Warp macros

#ifdef TEST_ALL_THREAD_MEMORY
#define TEST_ALL_THREAD_MEMORY_TILE
#define TEST_ALL_THREAD_MEMORY_VEC
#endif

// Group macros

#ifdef TEST_ALL_GROUP_MEMORY
#define TEST_ALL_GROUP_MEMORY_TILE
#define TEST_ALL_GROUP_MEMORY_VEC
#endif

#ifdef TEST_ALL_GROUP_SHARED
#define TEST_ALL_GROUP_SHARED_TILE
#define TEST_ALL_GROUP_SHARED_VEC
#endif

#ifdef TEST_ALL_GROUP_REG
#define TEST_ALL_GROUP_REG_TILE
#define TEST_ALL_GROUP_REG_VEC
#endif

#ifdef TEST_ALL_GROUP_MMA
#define TEST_ALL_GROUP_MMA_WARP
#if defined(KITTENS_HOPPER) && !defined(KITTENS_BLACKWELL)
#define TEST_ALL_GROUP_MMA_WARPGROUP
#endif
#ifdef KITTENS_BLACKWELL
#define TEST_ALL_GROUP_MMA_TENSOR
#endif
#endif

/* -----  DEPTH 3 MACROS  ----- */

// Warp macros

#ifdef TEST_ALL_THREAD_MEMORY_TILE
#define TEST_THREAD_MEMORY_TILE_GLOBAL_TO_REGISTER
#define TEST_THREAD_MEMORY_TILE_GLOBAL_TO_SHARED
#define TEST_THREAD_MEMORY_TILE_SHARED_TO_REGISTER
#ifdef KITTENS_HOPPER // only compile on H100
#define TEST_THREAD_MEMORY_TILE_TMA
#define TEST_THREAD_MEMORY_TILE_TMA_MULTICAST
#define TEST_THREAD_MEMORY_TILE_DSMEM
#endif
#endif

#ifdef TEST_ALL_THREAD_MEMORY_VEC
#define TEST_THREAD_MEMORY_VEC_GLOBAL_TO_REGISTER
#define TEST_THREAD_MEMORY_VEC_GLOBAL_TO_SHARED
#define TEST_THREAD_MEMORY_VEC_SHARED_TO_REGISTER
#ifdef KITTENS_HOPPER // only compile on H100
#define TEST_THREAD_MEMORY_VEC_TMA
#define TEST_THREAD_MEMORY_VEC_TMA_MULTICAST
#define TEST_THREAD_MEMORY_VEC_DSMEM
#endif
#endif

// Group macros

#ifdef TEST_ALL_GROUP_REG_TILE
#define TEST_GROUP_REG_TILE_REDUCTIONS
#define TEST_GROUP_REG_TILE_MAPS
#define TEST_GROUP_REG_TILE_MMA
#define TEST_GROUP_REG_TILE_CONVERSIONS
#define TEST_ALL_GROUP_REG_TILE_COMPLEX
#endif

#ifdef TEST_ALL_GROUP_REG_VEC
#define TEST_GROUP_REG_VEC_REDUCTIONS
#define TEST_GROUP_REG_VEC_MAPS
#define TEST_GROUP_REG_VEC_CONVERSIONS
#endif

#ifdef TEST_ALL_GROUP_MEMORY_TILE
#define TEST_GROUP_MEMORY_TILE_GLOBAL_TO_REGISTER
#define TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED
#define TEST_GROUP_MEMORY_TILE_SHARED_TO_REGISTER
#endif

#ifdef TEST_ALL_GROUP_MEMORY_VEC
#define TEST_GROUP_MEMORY_VEC_GLOBAL_TO_REGISTER
#define TEST_GROUP_MEMORY_VEC_GLOBAL_TO_SHARED
#define TEST_GROUP_MEMORY_VEC_SHARED_TO_REGISTER
#endif

#ifdef TEST_ALL_GROUP_SHARED_TILE
#define TEST_GROUP_SHARED_TILE_REDUCTIONS
#define TEST_GROUP_SHARED_TILE_MAPS
#define TEST_GROUP_SHARED_TILE_CONVERSIONS
#endif

#ifdef TEST_ALL_GROUP_SHARED_VEC
#define TEST_GROUP_SHARED_VEC_MAPS
#define TEST_GROUP_SHARED_VEC_CONVERSIONS
#endif

#ifdef TEST_ALL_GROUP_MMA_WARP
#define TEST_GROUP_MMA_WARP_MMA
#define TEST_ALL_GROUP_MMA_WARP_COMPLEX
#endif

#ifdef TEST_ALL_GROUP_MMA_WARPGROUP
// #define TEST_GROUP_WGMMA_MMA_FP32_FP32 TODO
#define TEST_GROUP_MMA_WARPGROUP_FP32_BF16
#define TEST_GROUP_MMA_WARPGROUP_FP32_FP16
#define TEST_GROUP_MMA_WARPGROUP_FP16_FP16
#define TEST_GROUP_MMA_WARPGROUP_FP32_FP8
#define TEST_GROUP_MMA_WARPGROUP_FP16_FP8
#define TEST_ALL_GROUP_MMA_WARPGROUP_COMPLEX
#endif

/* -----  DEPTH 5 MACROS  ----- */

#ifdef TEST_ALL_GROUP_MMA_WARPGROUP_COMPLEX
#define TEST_GROUP_COMPLEX_MMA_WARPGROUP_FP32_BF16
#define TEST_GROUP_COMPLEX_MMA_WARPGROUP_FP32_FP16
#define TEST_GROUP_COMPLEX_MMA_WARPGROUP_FP16_FP16
#endif

#ifdef TEST_ALL_GROUP_MMA_WARP_COMPLEX
#define TEST_GROUP_MMA_WARP_COMPLEX_MMA
#endif

#ifdef TEST_ALL_GROUP_REG_TILE_COMPLEX
#define TEST_GROUP_REG_TILE_COMPLEX_MUL
#endif

// Now we need to go back up the tree and make sure all dependent flags are defined.

/* -----  DEPTH 4 MACROS  ----- */

#if defined(TEST_GROUP_MMA_WARPGROUP_FP32_BF16) || defined(TEST_GROUP_MMA_WARPGROUP_FP32_FP16) || \
    defined(TEST_GROUP_MMA_WARPGROUP_FP16_FP16)
#define TEST_GROUP_MMA_WARPGROUP_COMPLEX
#endif

#if defined(TEST_GROUP_MMA_WARP_COMPLEX_MMA)
#define TEST_GROUP_MMA_WARP_COMPLEX
#endif

#if defined(TEST_GROUP_REG_TILE_COMPLEX_MUL)
#define TEST_GROUP_REG_TILE_COMPLEX
#endif

/* -----  DEPTH 3 MACROS  ----- */

// Warp macros

#if defined(TEST_THREAD_MEMORY_TILE_GLOBAL_TO_REGISTER) || defined(TEST_THREAD_MEMORY_TILE_GLOBAL_TO_SHARED) || \
    defined(TEST_THREAD_MEMORY_TILE_SHARED_TO_REGISTER) || defined(TEST_THREAD_MEMORY_TILE_TMA) || \
    defined(TEST_THREAD_MEMORY_TILE_DSMEM)              || defined(TEST_THREAD_MEMORY_TILE_TMA_MULTICAST)
#define TEST_THREAD_MEMORY_TILE
#endif

#if defined(TEST_THREAD_MEMORY_VEC_GLOBAL_TO_REGISTER) || defined(TEST_THREAD_MEMORY_VEC_GLOBAL_TO_SHARED) || \
    defined(TEST_THREAD_MEMORY_VEC_SHARED_TO_REGISTER) || defined(TEST_THREAD_MEMORY_VEC_TMA) || \
    defined(TEST_THREAD_MEMORY_VEC_DSMEM)              || defined(TEST_THREAD_MEMORY_VEC_TMA_MULTICAST)
#define TEST_THREAD_MEMORY_VEC
#endif

// Group macros

#if defined(TEST_GROUP_REG_TILE_REDUCTIONS) || defined(TEST_GROUP_REG_TILE_MAPS) || \
    defined(TEST_GROUP_REG_TILE_MMA) || defined(TEST_GROUP_REG_TILE_CONVERSIONS) || defined(TEST_GROUP_REG_VEC_COMPLEX)
#define TEST_GROUP_REG_TILE
#endif

#if defined(TEST_GROUP_REG_VEC_REDUCTIONS) || defined(TEST_GROUP_REG_VEC_MAPS) || \
    defined(TEST_GROUP_REG_VEC_CONVERSIONS)
#define TEST_GROUP_REG_VEC
#endif

#if defined(TEST_GROUP_MMA_WARP_MMA) || defined(TEST_GROUP_MMA_WARP_COMPLEX)
#define TEST_GROUP_MMA_WARP
#endif

#if defined(TEST_GROUP_MMA_WARPGROUP_FP16_FP16) || defined(TEST_GROUP_MMA_WARPGROUP_FP32_FP16) || \
    defined(TEST_GROUP_MMA_WARPGROUP_FP32_BF16) || defined(TEST_GROUP_MMA_WARPGROUP_FP16_FP8)  || \
    defined(TEST_GROUP_MMA_WARPGROUP_FP32_FP8)  || defined(TEST_GROUP_MMA_WARPGROUP_COMPLEX)
#define TEST_GROUP_MMA_WARPGROUP
#endif

#if defined(TEST_GROUP_MEMORY_TILE_GLOBAL_TO_REGISTER) || defined(TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED) || \
    defined(TEST_GROUP_MEMORY_TILE_SHARED_TO_REGISTER)
#define TEST_GROUP_MEMORY_TILE
#endif

#if defined(TEST_GROUP_MEMORY_VEC_GLOBAL_TO_REGISTER) || defined(TEST_GROUP_MEMORY_VEC_GLOBAL_TO_SHARED) || \
    defined(TEST_GROUP_MEMORY_VEC_SHARED_TO_REGISTER)
#define TEST_GROUP_MEMORY_VEC
#endif

#if defined(TEST_GROUP_SHARED_TILE_CONVERSIONS) || defined(TEST_GROUP_SHARED_TILE_MAPS) || \
    defined(TEST_GROUP_SHARED_TILE_REDUCTIONS)
#define TEST_GROUP_SHARED_TILE
#endif

#if defined(TEST_GROUP_SHARED_VEC_CONVERSIONS) || defined(TEST_GROUP_SHARED_VEC_MAPS)
#define TEST_GROUP_SHARED_VEC
#endif

/* -----  DEPTH 2 MACROS  ----- */

// Warp macros

#if defined(TEST_THREAD_MEMORY_TILE) || defined(TEST_THREAD_MEMORY_VEC)
#define TEST_THREAD_MEMORY
#endif

// Group macros

#if defined(TEST_GROUP_REG_TILE) || defined(TEST_GROUP_REG_VEC)
#define TEST_GROUP_REG
#endif

#if defined(TEST_GROUP_MEMORY_TILE) || defined(TEST_GROUP_MEMORY_VEC)
#define TEST_GROUP_MEMORY
#endif

#if defined(TEST_GROUP_SHARED_TILE) || defined(TEST_GROUP_SHARED_VEC)
#define TEST_GROUP_SHARED
#endif

#if defined(TEST_GROUP_MMA_WARP) || defined(TEST_GROUP_MMA_WARPGROUP) || defined(TEST_GROUP_MMA_TENSOR)
#define TEST_GROUP_MMA
#endif

/* -----  DEPTH 1 MACROS  ----- */

#if defined(TEST_THREAD_MEMORY)
#define TEST_THREAD
#endif

#if defined(TEST_GROUP_MEMORY) || defined(TEST_GROUP_SHARED) || defined(TEST_GROUP_MMA) || defined(TEST_GROUP_REG)
#define TEST_GROUP
#endif

/* ----------  TEST INTENSITY MACROS  ---------- */

// Intensity 1 is a cursory glance
#define TEST_INTENSITY_1 (1)
// Intensity 2 is to actually check
#define TEST_INTENSITY_2 (2)
// Intensity 3 is a thorough check
#define TEST_INTENSITY_3 (3)
// Intensity 4 is for debugging small chunks of code.
#define TEST_INTENSITY_4 (4)

#ifndef TEST_INTENSITY
// low-mid intensity by default
#define TEST_INTENSITY (2)
#endif

#define INTENSITY_1 (TEST_INTENSITY == TEST_INTENSITY_1)
#define INTENSITY_2 (TEST_INTENSITY == TEST_INTENSITY_2)
#define INTENSITY_3 (TEST_INTENSITY == TEST_INTENSITY_3)
#define INTENSITY_4 (TEST_INTENSITY == TEST_INTENSITY_4)