#include "register.cuh"

#ifdef TEST_WARP_REGISTER

void warp::reg::tests(test_data &results) {
    std::cout << "\n ---------- Starting ops/warp/register tests! ----------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_MAPS
    warp::reg::maps::tests(results);
#endif
#ifdef TEST_WARP_REGISTER_REDUCTIONS
    warp::reg::reductions::tests(results);
#endif
#ifdef TEST_WARP_REGISTER_MMA
    warp::reg::mma::tests(results);
#endif
#ifdef TEST_WARP_REGISTER_CONVERSIONS
    warp::reg::conversions::tests(results);
#endif
#ifdef TEST_WARP_REGISTER_VEC
    // warp::reg::vec::tests(results);
#endif
}

#endif