#include "warpgroup.cuh"

#ifdef TEST_WARPGROUP

using namespace warpgroup;

void warpgroup::tests(test_data &results) {
    std::cout << "\n ------------------------------     Starting ops/warpgroup tests!     ------------------------------\n" << std::endl;
#ifdef TEST_WARPGROUP_MEMORY
    memory::tests(results);
#endif
#ifdef TEST_WARPGROUP_REGISTER
    reg::tests(results); // register is a reserved word, hence reg
#endif
#ifdef TEST_WARPGROUP_SHARED
    shared::tests(results);
#endif
}

#endif