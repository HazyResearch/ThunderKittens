#include "shared.cuh"

#ifdef TEST_WARPGROUP_SHARED

void warpgroup::shared::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/warpgroup/shared tests! --------------------\n" << std::endl;
#ifdef TEST_WARPGROUP_SHARED_MAPS
    warpgroup::shared::maps::tests(results);
#endif
}

#endif