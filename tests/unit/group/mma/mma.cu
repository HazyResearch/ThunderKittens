#include "mma.cuh"

#ifdef TEST_GROUP_MMA

void group::mma::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/group/mma tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_MMA_WARP
    group::mma::warp::tests(results);
#else
    std::cout << "Skipping ops/group/mma/warp tests!" << std::endl;
#endif
#ifdef TEST_GROUP_MMA_WARPGROUP
    group::mma::warpgroup::tests(results);
#else
    std::cout << "Skipping ops/group/mma/warpgroup tests!" << std::endl;
#endif
#ifdef TEST_GROUP_MMA_TENSOR
    group::mma::tensor::tests(results);
#else
    std::cout << "Skipping ops/group/mma/tensor tests!" << std::endl;
#endif
}

#endif