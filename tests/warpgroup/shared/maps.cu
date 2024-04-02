#include "maps.cuh"

#ifdef TEST_WARPGROUP_SHARED_MAPS

void warpgroup::shared::maps::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warpgroup/shared/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_st_layout_size_2d_warpgroup<warpgroup::shared::maps::test_exp, SIZE, SIZE>::run(results);
}

#endif