#include "wgmma.cuh"

#ifdef TEST_GROUP_WGMMA

void group::wgmma::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/group/wgmma tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_WGMMA_MMA
    group::wgmma::mma::tests(results);
#endif
}

#endif