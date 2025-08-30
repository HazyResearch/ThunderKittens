#include "group.cuh"

#ifdef TEST_GROUP

void group::tests(test_data &results) {
    std::cout << " ------------------------------     Starting ops/group tests!     ------------------------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY
    group::memory::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/memory tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_REG
    group::reg::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/register tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_SHARED
    group::shared::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/shared tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_MMA
    group::mma::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/mma tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif