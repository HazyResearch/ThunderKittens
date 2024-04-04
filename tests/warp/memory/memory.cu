#include "memory.cuh"

#ifdef TEST_WARP_MEMORY

void warp::memory::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/warp/memory tests! --------------------\n" << std::endl;
#ifdef TEST_WARP_MEMORY_GLOBAL_TO_REGISTER
    warp::memory::global_to_register::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_GLOBAL_TO_SHARED
    warp::memory::global_to_shared::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_SHARED_TO_REGISTER
    warp::memory::shared_to_register::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_TMA
    warp::memory::tma::tests(results);
#endif
}

#endif