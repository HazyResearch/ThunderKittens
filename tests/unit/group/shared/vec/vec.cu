#include "vec.cuh"

#ifdef TEST_GROUP_SHARED_VEC

void group::shared::vec::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/group/shared/vec tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_SHARED_VEC_CONVERSIONS
    group::shared::vec::conversions::tests(results);
#endif
#ifdef TEST_GROUP_SHARED_VEC_MAPS
    group::shared::vec::maps::tests(results);
#endif
}

#endif