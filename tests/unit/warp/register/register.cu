#include "register.cuh"

#ifdef TEST_WARP_REGISTER

void warp::reg::tests(test_data &results) {
    std::cout << " -------------------- Starting ops/warp/register tests! --------------------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_TILE
    warp::reg::tile::tests(results);
#else
    std::cout << "INFO: Skipping ops/warp/register/tile tests!\n" << std::endl;
#endif
#ifdef TEST_WARP_REGISTER_VEC
    warp::reg::vec::tests(results);
#else
    std::cout << "INFO: Skipping ops/warp/register/vec tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif