#include "mma.cuh"

#ifdef TEST_WARPGROUP_WGMMA_MMA

using namespace warpgroup::wgmma::mma;
using namespace kittens::ducks::st_layout;
// If 1 and 3 work, the others likely will too.
using I1_t = std::integral_constant<int, 1>;
using I3_t = std::integral_constant<int, 3>;
void warpgroup::wgmma::mma::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warpgroup/wgmma/mma tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 1 :
                         INTENSITY_2 ? 2 : 
                         INTENSITY_3 ? 4 :
                         INTENSITY_4 ? 8 : -1;
    mma_sweep_width_warpgroup<warpgroup::wgmma::mma::mma, SIZE, I1_t, wgmma_row_0b , wgmma_col_t_0b >::run(results);
    // mma_sweep_width_warpgroup<warpgroup::wgmma::mma::mma, SIZE, I1_t, wgmma_row_0b , wgmma_col_t_32b>::run(results);
    mma_sweep_width_warpgroup<warpgroup::wgmma::mma::mma, SIZE, I1_t, wgmma_row_32b, wgmma_col_t_0b >::run(results);
    // mma_sweep_width_warpgroup<warpgroup::wgmma::mma::mma, SIZE, I1_t, wgmma_row_32b, wgmma_col_t_32b>::run(results);
    mma_sweep_width_warpgroup<warpgroup::wgmma::mma::mma, SIZE, I3_t, wgmma_row_0b , wgmma_col_t_0b >::run(results);
    // mma_sweep_width_warpgroup<warpgroup::wgmma::mma::mma, SIZE, I3_t, wgmma_row_0b , wgmma_col_t_32b>::run(results);
    mma_sweep_width_warpgroup<warpgroup::wgmma::mma::mma, SIZE, I3_t, wgmma_row_32b, wgmma_col_t_0b >::run(results);
    // mma_sweep_width_warpgroup<warpgroup::wgmma::mma::mma, SIZE, I3_t, wgmma_row_32b, wgmma_col_t_32b>::run(results);
    mma_sweep_width_warpgroup<warpgroup::wgmma::mma::dot, SIZE, I1_t, wgmma_row_0b , wgmma_row_0b >::run(results);
    mma_sweep_width_warpgroup<warpgroup::wgmma::mma::dot, SIZE, I1_t, wgmma_row_0b , wgmma_row_32b>::run(results);
    mma_sweep_width_warpgroup<warpgroup::wgmma::mma::dot, SIZE, I1_t, wgmma_row_32b, wgmma_row_0b >::run(results);
    mma_sweep_width_warpgroup<warpgroup::wgmma::mma::dot, SIZE, I1_t, wgmma_row_32b, wgmma_row_32b>::run(results);
    mma_sweep_width_warpgroup<warpgroup::wgmma::mma::dot, SIZE, I3_t, wgmma_row_0b , wgmma_row_0b >::run(results);
    mma_sweep_width_warpgroup<warpgroup::wgmma::mma::dot, SIZE, I3_t, wgmma_row_0b , wgmma_row_32b>::run(results);
    mma_sweep_width_warpgroup<warpgroup::wgmma::mma::dot, SIZE, I3_t, wgmma_row_32b, wgmma_row_0b >::run(results);
    mma_sweep_width_warpgroup<warpgroup::wgmma::mma::dot, SIZE, I3_t, wgmma_row_32b, wgmma_row_32b>::run(results);
}

#endif
