#include "complex.cuh"

#ifdef TEST_GROUP_MMA_WARP_COMPLEX

void group::mma::warp::complex::tests(test_data &results) {
    std::cout << " -------------------- Starting ops/group/mma/warp/complex tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_MMA_WARP_COMPLEX_MMA
    group::mma::warp::complex::mma::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/mma/warp/complex/mma tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif