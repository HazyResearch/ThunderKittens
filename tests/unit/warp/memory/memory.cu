#include "memory.cuh"

#ifdef TEST_WARP_MEMORY

void warp::memory::tests(test_data &results) {
    std::cout << " -------------------- Starting ops/warp/memory tests! --------------------\n" << std::endl;
#ifdef TEST_WARP_MEMORY_TILE
    warp::memory::tile::tests(results);
#else
    std::cout << "INFO: Skipping ops/warp/memory/tile tests!\n" << std::endl;
#endif
#ifdef TEST_WARP_MEMORY_VEC
    warp::memory::vec::tests(results);
#else
    std::cout << "INFO: Skipping ops/warp/memory/vec tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif