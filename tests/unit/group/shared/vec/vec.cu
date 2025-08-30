#include "vec.cuh"

#ifdef TEST_GROUP_SHARED_VEC

void group::shared::vec::tests(test_data &results) {
    std::cout << " --------------- Starting ops/group/shared/vec tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_SHARED_VEC_CONVERSIONS
    group::shared::vec::conversions::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/shared/vec/conversions tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_SHARED_VEC_MAPS
    group::shared::vec::maps::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/shared/vec/maps tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif