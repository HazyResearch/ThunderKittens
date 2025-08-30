#include "register.cuh"

#ifdef TEST_GROUP_REG

void group::reg::tests(test_data &results) {
    std::cout << " -------------------- Starting ops/group/register tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_REG_TILE
    group::reg::tile::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/register/tile tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_REG_VEC
    group::reg::vec::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/register/vec tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif