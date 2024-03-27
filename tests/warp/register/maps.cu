#include "maps.cuh"

#ifdef TEST_WARP_REGISTER_MAPS

void warp::reg::maps::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_size_2d_warp<warp::reg::maps::test_exp, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<warp::reg::maps::test_exp, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
}

#endif