#include "warp.cuh"

#ifdef TEST_WARP

using namespace warp;

void warp::tests(test_data &results) {
    std::cout << " ------------------------------     Starting ops/warp tests!     ------------------------------\n"  << std::endl;
#ifdef TEST_WARP_MEMORY
    memory::tests(results);
#else
    std::cout << "INFO: Skipping ops/warp/memory tests!\n" << std::endl;
#endif
#ifdef TEST_WARP_REGISTER
    reg::tests(results); // register is a reserved word, hence reg
#else
    std::cout << "INFO: Skipping ops/warp/register tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif