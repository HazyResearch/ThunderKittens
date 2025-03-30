#include "vec.cuh"

#ifdef TEST_WARP_MEMORY_VEC

void warp::memory::vec::tests(test_data &results) {
    std::cout << " --------------- Starting ops/warp/memory/vec tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_MEMORY_VEC_TMA
    warp::memory::vec::tma::tests(results);
#else
    std::cout << "INFO: Skipping ops/warp/memory/vec/tma tests!\n" << std::endl;
#endif
#ifdef TEST_WARP_MEMORY_VEC_TMA_MULTICAST
    warp::memory::vec::tma_multicast::tests(results);
#else
    std::cout << "INFO: Skipping ops/warp/memory/vec/tma_multicast tests!\n" << std::endl;
#endif
#ifdef TEST_WARP_MEMORY_VEC_DSMEM
    warp::memory::vec::dsmem::tests(results);
#else
    std::cout << "INFO: Skipping ops/warp/memory/vec/dsmem tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif