#include "maps.cuh"

#ifdef TEST_WARP_REGISTER_MAPS

void warp::reg::maps::tests(test_data &results) {
    std::cout << " ----- Starting ops/warp/register/maps tests! -----" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_size_2d_warp<warp::reg::maps::test_exp, SIZE, SIZE, ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<warp::reg::maps::test_exp, SIZE, SIZE, ducks::rt_layout::col>::run(results);
}

#endif