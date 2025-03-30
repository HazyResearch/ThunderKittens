#include "warp.cuh"

#ifdef TEST_WARP

using namespace warp;

void warp::tests(test_data &results) {
    std::cout << "\n ------------------------------     Starting ops/warp tests!     ------------------------------\n"  << std::endl;
#ifdef TEST_WARP_MEMORY
    memory::tests(results);
#else
    std::cout << "Skipping ops/warp/memory tests!" << std::endl;
#endif
#ifdef TEST_WARP_REGISTER
    reg::tests(results); // register is a reserved word, hence reg
#else
    std::cout << "Skipping ops/warp/register tests!" << std::endl;
#endif
}

#endif