#include "register.cuh"

#ifdef TEST_WARP_REGISTER

void warp::reg::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/warp/register tests! --------------------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_TILE
    warp::reg::tile::tests(results);
#else
    std::cout << "Skipping ops/warp/register/tile tests!" << std::endl;
#endif
#ifdef TEST_WARP_REGISTER_VEC
    warp::reg::vec::tests(results);
#else
    std::cout << "Skipping ops/warp/register/vec tests!" << std::endl;
#endif
}

#endif