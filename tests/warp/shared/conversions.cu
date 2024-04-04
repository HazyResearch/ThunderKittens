#include "conversions.cuh"

#ifdef TEST_WARP_SHARED_CONVERSIONS

void warp::shared::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_st_layout_size_2d_warp<warp::shared::conversions::swap_layout, SIZE, SIZE, kittens::ducks::st_layout::naive>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::swap_layout, SIZE, SIZE, kittens::ducks::st_layout::tma_swizzle>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::swap_layout, SIZE, SIZE, kittens::ducks::st_layout::xor_swizzle>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::swap_layout, SIZE, SIZE, kittens::ducks::st_layout::wgmma_row_0b>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::swap_layout, SIZE, SIZE, kittens::ducks::st_layout::wgmma_row_32b>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::swap_layout, SIZE, SIZE, kittens::ducks::st_layout::wgmma_col_t_0b>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::swap_layout, SIZE, SIZE, kittens::ducks::st_layout::wgmma_col_t_32b>::run(results);

    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 1>, std::integral_constant<int, 1>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 1>, std::integral_constant<int, 2>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 1>, std::integral_constant<int, 3>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 1>, std::integral_constant<int, 4>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 2>, std::integral_constant<int, 1>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 2>, std::integral_constant<int, 2>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 2>, std::integral_constant<int, 3>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 2>, std::integral_constant<int, 4>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 1>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 2>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 3>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 4>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 1>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 2>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 3>>::run(results);
    sweep_st_layout_size_2d_warp<warp::shared::conversions::subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 4>>::run(results);
}

#endif