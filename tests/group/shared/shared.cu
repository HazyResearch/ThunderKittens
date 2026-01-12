#include "shared.cuh"

#ifdef TEST_GROUP_SHARED

void group::shared::tests(test_data &results) {
    std::cout << " -------------------- Starting ops/group/shared tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_SHARED_TILE
    group::shared::tile::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/shared/tile tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_SHARED_VEC
    group::shared::vec::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/shared/vec tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif