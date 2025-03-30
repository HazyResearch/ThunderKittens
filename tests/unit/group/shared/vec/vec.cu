#include "vec.cuh"

#ifdef TEST_GROUP_SHARED_VEC

void group::shared::vec::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/group/shared/vec tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_SHARED_VEC_CONVERSIONS
    group::shared::vec::conversions::tests(results);
#else
    std::cout << "Skipping ops/group/shared/vec/conversions tests!" << std::endl;
#endif
#ifdef TEST_GROUP_SHARED_VEC_MAPS
    group::shared::vec::maps::tests(results);
#else
    std::cout << "Skipping ops/group/shared/vec/maps tests!" << std::endl;
#endif
}

#endif