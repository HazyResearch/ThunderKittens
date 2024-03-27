#include "conversions.cuh"

#ifdef TEST_WARP_REGISTER_CONVERSIONS

void warp::reg::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_size_2d_warp<warp::reg::conversions::swap_layout, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<warp::reg::conversions::swap_layout, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<warp::reg::conversions::transpose, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<warp::reg::conversions::transpose, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<warp::reg::conversions::type_convert, SIZE, SIZE, float2, kittens::bf16_2>::run(results);
    sweep_size_2d_warp<warp::reg::conversions::type_convert, SIZE, SIZE, kittens::bf16_2, float2>::run(results);
    sweep_size_2d_warp<warp::reg::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    sweep_size_2d_warp<warp::reg::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    sweep_size_2d_warp<warp::reg::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    sweep_size_2d_warp<warp::reg::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
}

#endif