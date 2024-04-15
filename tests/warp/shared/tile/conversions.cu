#include "conversions.cuh"

#ifdef TEST_WARP_SHARED_TILE_CONVERSIONS

void warp::shared::tile::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_st_layout_size_2d_warp<test_swap_layout, SIZE, SIZE, kittens::ducks::st_layout::naive>::run(results);
    sweep_st_layout_size_2d_warp<test_swap_layout, SIZE, SIZE, kittens::ducks::st_layout::xor_swizzle>::run(results);
    sweep_st_layout_size_2d_warp<test_swap_layout, SIZE, SIZE, kittens::ducks::st_layout::wgmma_row_0b>::run(results);
    sweep_st_layout_size_2d_warp<test_swap_layout, SIZE, SIZE, kittens::ducks::st_layout::wgmma_row_32b>::run(results);
    sweep_st_layout_size_2d_warp<test_swap_layout, SIZE, SIZE, kittens::ducks::st_layout::wgmma_col_t_0b>::run(results);
    sweep_st_layout_size_2d_warp<test_swap_layout, SIZE, SIZE, kittens::ducks::st_layout::wgmma_col_t_32b>::run(results);

    warp::shared::tile::conversions::detail::internal_tests_2(results); // run internal tests compiled separately.
    warp::shared::tile::conversions::detail::internal_tests_3(results); // run internal tests compiled separately.
}

#endif