#include "memory.cuh"

#ifdef TEST_WARP_MEMORY

void warp::memory::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/warp/memory tests! --------------------\n" << std::endl;
#ifdef TEST_WARP_MEMORY_TILE
    warp::memory::tile::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_VEC
    warp::memory::vec::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_UTIL
    warp::memory::util::tests(results);
#endif
}

#endif
