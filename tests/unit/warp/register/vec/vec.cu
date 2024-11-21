#include "vec.cuh"

#ifdef TEST_WARP_REGISTER_VEC

void warp::reg::vec::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/register/vec tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_VEC_CONVERSIONS
    warp::reg::vec::conversions::tests(results);
#endif
#ifdef TEST_WARP_REGISTER_VEC_MAPS
    warp::reg::vec::maps::tests(results);
#endif
#ifdef TEST_WARP_REGISTER_VEC_REDUCTIONS
    warp::reg::vec::reductions::tests(results);
#endif
}

#endif