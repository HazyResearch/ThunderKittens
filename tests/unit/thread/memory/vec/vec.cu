#include "vec.cuh"

#ifdef TEST_THREAD_MEMORY_VEC

void thread::memory::vec::tests(test_data &results) {
    std::cout << " --------------- Starting ops/thread/memory/vec tests! ---------------\n" << std::endl;
#ifdef TEST_THREAD_MEMORY_VEC_TMA
    thread::memory::vec::tma::tests(results);
#else
    std::cout << "INFO: Skipping ops/thread/memory/vec/tma tests!\n" << std::endl;
#endif
#ifdef TEST_THREAD_MEMORY_VEC_TMA_MULTICAST
    thread::memory::vec::tma_multicast::tests(results);
#else
    std::cout << "INFO: Skipping ops/thread/memory/vec/tma_multicast tests!\n" << std::endl;
#endif
#ifdef TEST_THREAD_MEMORY_VEC_DSMEM
    thread::memory::vec::dsmem::tests(results);
#else
    std::cout << "INFO: Skipping ops/thread/memory/vec/dsmem tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif