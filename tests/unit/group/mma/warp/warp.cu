#include "warp.cuh"

#ifdef TEST_GROUP_MMA_WARP

// I have concluded it is better to split up the different MMA's into different files
// substantially because otherwise compilation times will get out of hand.

void group::mma::warp:tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/group/mma/warp tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_MMA_WARP_MMA
    group::mma::warp::mma::tests(results);
#else
    std::cout << "Skipping ops/group/mma/warp/mma tests!" << std::endl;
#endif
#ifdef TEST_GROUP_MMA_WARP_COMPLEX
    group::mma::warp::complex::tests(results);
#else
    std::cout << "Skipping ops/group/mma/warp/complex tests!" << std::endl;
#endif
}

#endif