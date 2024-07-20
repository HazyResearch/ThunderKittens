#include "conversions.cuh"

#ifdef TEST_WARP_SHARED_TILE_CONVERSIONS

void warp::shared::tile::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_gmem_type_2d_warp<test_swap_layout, SIZE, SIZE>::run(results);
    sweep_gmem_type_2d_warp<test_swap_layout, SIZE, SIZE>::run(results);
    sweep_gmem_type_2d_warp<test_swap_layout, SIZE, SIZE>::run(results);
    sweep_gmem_type_2d_warp<test_swap_layout, SIZE, SIZE>::run(results);

    warp::shared::tile::conversions::detail::internal_tests_2(results); // run internal tests compiled separately.
    warp::shared::tile::conversions::detail::internal_tests_3(results); // run internal tests compiled separately.
}

#endif