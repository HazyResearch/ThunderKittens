#include "group.cuh"

#ifdef TEST_GROUP

using namespace group;

void group::tests(test_data &results) {
    std::cout << "\n ------------------------------     Starting ops/group tests!     ------------------------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY
    memory::tests(results);
#endif
}

#endif