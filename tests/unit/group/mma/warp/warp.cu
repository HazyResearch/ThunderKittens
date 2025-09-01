#include "warp.cuh"

#ifdef TEST_GROUP_MMA_WARP

// I have concluded it is better to split up the different MMA's into different files
// substantially because otherwise compilation times will get out of hand.

void group::mma::warp::tests(test_data &results) {
    std::cout << " -------------------- Starting ops/group/mma/warp tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_MMA_WARP_MMA
    group::mma::warp::mma::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/mma/warp/mma tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_MMA_WARP_COMPLEX
    group::mma::warp::complex::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/mma/warp/complex tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif