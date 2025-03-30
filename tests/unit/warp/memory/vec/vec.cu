#include "vec.cuh"

#ifdef TEST_WARP_MEMORY_VEC

void warp::memory::vec::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/memory/vec tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_MEMORY_VEC_TMA
    warp::memory::vec::tma::tests(results);
#else
    std::cout << "Skipping ops/warp/memory/vec/tma tests!" << std::endl;
#endif
#ifdef TEST_WARP_MEMORY_VEC_TMA_MULTICAST
    warp::memory::vec::tma_multicast::tests(results);
#else
    std::cout << "Skipping ops/warp/memory/vec/tma_multicast tests!" << std::endl;
#endif
#ifdef TEST_WARP_MEMORY_VEC_DSMEM
    warp::memory::vec::dsmem::tests(results);
#else
    std::cout << "Skipping ops/warp/memory/vec/dsmem tests!" << std::endl;
#endif
}

#endif