/* 
 * This file exists because there are a *huge* number of shared conversions to check.
 * Splitting it up improves the parallelism and substantially reduces the compile time
 * of the overall test suite.
 */
#include "conversions.cuh"

#ifdef TEST_WARP_SHARED_TILE_CONVERSIONS

void warp::shared::tile::conversions::detail::internal_tests_3(test_data &results) {
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 1>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 2>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 3>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 4>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 1>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 2>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 3>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 4>>::run(results);
}

#endif