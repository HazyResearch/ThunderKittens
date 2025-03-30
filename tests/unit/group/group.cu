#include "group.cuh"

#ifdef TEST_GROUP

void group::tests(test_data &results) {
    std::cout << "\n ------------------------------     Starting ops/group tests!     ------------------------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY
    group::memory::tests(results);
#else
    std::cout << "Skipping ops/group/memory tests!" << std::endl;
#endif
#ifdef TEST_GROUP_SHARED
    group::shared::tests(results);
#else
    std::cout << "Skipping ops/group/shared tests!" << std::endl;
#endif
#ifdef TEST_GROUP_MMA
    group::mma::tests(results);
#else
    std::cout << "Skipping ops/group/mma tests!" << std::endl;
#endif
}

#endif