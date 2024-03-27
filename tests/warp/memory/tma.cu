#include "tma.cuh"

#ifdef TEST_WARP_MEMORY_TMA

void warp::memory::tma::tests(test_data &results) {
    std::cout << "\n ----- Starting tma tests! -----\n" << std::endl;
    tma_dim_test<ducks::st_layout::naive, false>(results);
    tma_dim_test<ducks::st_layout::naive, true>(results);

    tma_dim_test<ducks::st_layout::tma_swizzle, false, true>(results);
    tma_dim_test<ducks::st_layout::tma_swizzle, true, true>(results);

    tma_dim_test<ducks::st_layout::wgmma_row_0b, false>(results);
    tma_dim_test<ducks::st_layout::wgmma_row_0b, true>(results);
    // not supported with 32B swizzling modes
    // tma_dim_test<ducks::st_layout::wgmma_row_32b, false>(results);
    // tma_dim_test<ducks::st_layout::wgmma_row_32b, true>(results);

    tma_dim_test<ducks::st_layout::wgmma_col_t_0b, false>(results);
    tma_dim_test<ducks::st_layout::wgmma_col_t_0b, true>(results);
    // not supported with 32B swizzling modes
    // tma_dim_test<wgmma_col_t_32b, false>(results);
    // tma_dim_test<wgmma_col_t_32b, true>(results);
}

#endif