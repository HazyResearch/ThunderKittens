#include "vec.cuh"

#ifdef TEST_GROUP_REG_VEC

void group::reg::vec::tests(test_data &results) {
    std::cout << " --------------- Starting ops/group/register/vec tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_REG_VEC_CONVERSIONS
    group::reg::vec::conversions::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/register/vec/conversions tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_REG_VEC_MAPS
    group::reg::vec::maps::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/register/vec/maps tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_REG_VEC_REDUCTIONS
    group::reg::vec::reductions::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/register/vec/reductions tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif