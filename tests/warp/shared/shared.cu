#include "shared.cuh"

#ifdef TEST_WARP_SHARED

void warp::shared::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/warp/shared tests! --------------------\n" << std::endl;
#ifdef TEST_WARP_SHARED_CONVERSIONS
    warp::shared::conversions::tests(results);
#endif
#ifdef TEST_WARP_SHARED_VEC
    // warp::shared::vec::tests(results);
#endif
}

#endif