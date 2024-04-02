/* ----------  TEST INCLUSION MACROS  ---------- */

/* -----  DEPTH 0 MACROS  ----- */

// Catchall macros
#ifdef TEST_ALL
#define TEST_ALL_WARP
#define TEST_ALL_WARPGROUP
#define TEST_ALL_BLOCK
#endif

/* -----  DEPTH 1 MACROS  ----- */

// Warp macros
#ifdef TEST_ALL_WARP
#define TEST_ALL_WARP_MEMORY
#define TEST_ALL_WARP_REGISTER
#define TEST_ALL_WARP_SHARED
#endif

// Warpgroup macros
#ifdef TEST_ALL_WARPGROUP
#define TEST_ALL_WARPGROUP_MEMORY
#ifdef KITTENS_HOPPER  // only compile on H100
#define TEST_ALL_WARPGROUP_WGMMA // leaf
#endif
#endif

// Block macros
#ifdef TEST_ALL_BLOCK
#define TEST_ALL_BLOCK_MEMORY
#endif

/* -----  DEPTH 2 MACROS  ----- */

// Warp macros

#ifdef TEST_ALL_WARP_MEMORY
#define TEST_WARP_MEMORY_GLOBAL_TO_REGISTER
#define TEST_WARP_MEMORY_GLOBAL_TO_SHARED
#define TEST_WARP_MEMORY_SHARED_TO_REGISTER
#ifdef KITTENS_HOPPER // only compile on H100
#define TEST_WARP_MEMORY_TMA
#endif
#endif

#ifdef TEST_ALL_WARP_REGISTER
#define TEST_WARP_REGISTER_REDUCTIONS
#define TEST_WARP_REGISTER_MAPS
#define TEST_WARP_REGISTER_MMA
#define TEST_WARP_REGISTER_CONVERSIONS
#define TEST_WARP_REGISTER_VEC
#endif

#ifdef TEST_ALL_WARP_SHARED
#define TEST_WARP_SHARED_CONVERSIONS
#define TEST_WARP_SHARED_VEC
#endif

// Warpgroup macros

#ifdef TEST_ALL_WARPGROUP_MEMORY
#define TEST_WARPGROUP_MEMORY_GLOBAL_TO_REGISTER
#define TEST_WARPGROUP_MEMORY_GLOBAL_TO_SHARED
#define TEST_WARPGROUP_MEMORY_SHARED_TO_REGISTER
#endif

#ifdef TEST_ALL_WARPGROUP_WGMMA
#define TEST_WARPGROUP_WGMMA_MMA
#endif

#ifdef TEST_ALL_WARPGROUP_SHARED
#define TEST_WARPGROUP_SHARED_MAPS
#endif

// Block macros

#ifdef TEST_ALL_BLOCK_MEMORY
#define TEST_BLOCK_MEMORY_GLOBAL_TO_REGISTER
#define TEST_BLOCK_MEMORY_GLOBAL_TO_SHARED
#define TEST_BLOCK_MEMORY_SHARED_TO_REGISTER
#ifdef KITTENS_HOPPER // only compile on H100
#define TEST_BLOCK_MEMORY_DSMEM
#endif
#endif

// Now we need to go back up the tree and make sure all dependent flags are defined.

/* -----  DEPTH 2 MACROS  ----- */

// Warp macros

#if defined(TEST_WARP_MEMORY_GLOBAL_TO_REGISTER) || defined(TEST_WARP_MEMORY_GLOBAL_TO_SHARED) || \
    defined(TEST_WARP_MEMORY_SHARED_TO_REGISTER) || defined(TEST_WARP_MEMORY_TMA)
#define TEST_WARP_MEMORY
#endif

#if defined(TEST_WARP_REGISTER_REDUCTIONS) || defined(TEST_WARP_REGISTER_MAPS) || \
    defined(TEST_WARP_REGISTER_MMA) || defined(TEST_WARP_REGISTER_CONVERSIONS) || \
    defined(TEST_WARP_REGISTER_VEC)
#define TEST_WARP_REGISTER
#endif

#if defined(TEST_WARP_SHARED_CONVERSIONS) || defined(TEST_WARP_SHARED_VEC)
#define TEST_WARP_SHARED
#endif

// Warpgroup macros
#if defined(TEST_WARPGROUP_MEMORY_GLOBAL_TO_REGISTER) || defined(TEST_WARPGROUP_MEMORY_GLOBAL_TO_SHARED) || \
    defined(TEST_WARPGROUP_MEMORY_SHARED_TO_REGISTER)
#define TEST_WARPGROUP_MEMORY
#endif
#ifdef TEST_WARPGROUP_WGMMA_MMA
#define TEST_WARPGROUP_WGMMA
#endif
#ifdef TEST_WARPGROUP_SHARED_MAPS
#define TEST_WARPGROUP_SHARED
#endif

// Block macros
#if defined(TEST_BLOCK_MEMORY_GLOBAL_TO_REGISTER) || defined(TEST_BLOCK_MEMORY_GLOBAL_TO_SHARED) || \
    defined(TEST_BLOCK_MEMORY_SHARED_TO_REGISTER) || defined(TEST_BLOCK_MEMORY_DSMEM)
#define TEST_BLOCK_MEMORY
#endif

/* -----  DEPTH 1 MACROS  ----- */

#if defined(TEST_WARP_MEMORY) || defined(TEST_WARP_REGISTER) || defined(TEST_WARP_SHARED)
#define TEST_WARP
#endif

#if defined(TEST_WARPGROUP_MEMORY) || defined(TEST_WARPGROUP_WGMMA) || defined(TEST_WARPGROUP_SHARED)
#define TEST_WARPGROUP
#endif

#if defined(TEST_BLOCK_MEMORY)
#define TEST_BLOCK
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