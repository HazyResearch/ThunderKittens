#include "shared.cuh"

#ifdef TEST_GROUP_SHARED

void group::shared::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/group/shared tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_SHARED_TILE
    group::shared::tile::tests(results);
#endif
#ifdef TEST_GROUP_SHARED_VEC
    group::shared::vec::tests(results);
#endif
}

#endif