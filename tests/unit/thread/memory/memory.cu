#include "memory.cuh"

#ifdef TEST_THREAD_MEMORY

void thread::memory::tests(test_data &results) {
    std::cout << " -------------------- Starting ops/thread/memory tests! --------------------\n" << std::endl;
#ifdef TEST_THREAD_MEMORY_TILE
    thread::memory::tile::tests(results);
#else
    std::cout << "INFO: Skipping ops/thread/memory/tile tests!\n" << std::endl;
#endif
#ifdef TEST_THREAD_MEMORY_VEC
    thread::memory::vec::tests(results);
#else
    std::cout << "INFO: Skipping ops/thread/memory/vec tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif