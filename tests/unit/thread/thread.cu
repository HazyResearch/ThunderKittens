#include "thread.cuh"

#ifdef TEST_THREAD

using namespace thread;

void thread::tests(test_data &results) {
    std::cout << " ------------------------------     Starting ops/warp tests!     ------------------------------\n"  << std::endl;
#ifdef TEST_THREAD_MEMORY
    memory::tests(results);
#else
    std::cout << "INFO: Skipping ops/thread/memory tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif