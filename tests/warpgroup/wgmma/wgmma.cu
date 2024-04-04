#include "wgmma.cuh"

#ifdef TEST_WARPGROUP_WGMMA

void warpgroup::wgmma::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/warpgroup/wgmma tests! --------------------\n" << std::endl;
#ifdef TEST_WARPGROUP_WGMMA_MMA
    warpgroup::wgmma::mma::tests(results);
#endif
}

#endif