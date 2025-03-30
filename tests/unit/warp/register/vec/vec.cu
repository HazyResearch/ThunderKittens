#include "vec.cuh"

#ifdef TEST_WARP_REGISTER_VEC

void warp::reg::vec::tests(test_data &results) {
    std::cout << " --------------- Starting ops/warp/register/vec tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_VEC_CONVERSIONS
    warp::reg::vec::conversions::tests(results);
#else
    std::cout << "INFO: Skipping ops/warp/register/vec/conversions tests!\n" << std::endl;
#endif
#ifdef TEST_WARP_REGISTER_VEC_MAPS
    warp::reg::vec::maps::tests(results);
#else
    std::cout << "INFO: Skipping ops/warp/register/vec/maps tests!\n" << std::endl;
#endif
#ifdef TEST_WARP_REGISTER_VEC_REDUCTIONS
    warp::reg::vec::reductions::tests(results);
#else
    std::cout << "INFO: Skipping ops/warp/register/vec/reductions tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif