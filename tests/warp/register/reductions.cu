#include "reductions.cuh"

#ifdef TEST_WARP_REGISTER_REDUCTIONS

void warp::reg::reductions::tests(test_data &results) {
    std::cout << " ----- Starting ops/warp/register/reductions tests! -----" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_size_2d_warp<normalize_row, SIZE, SIZE, ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<normalize_row, SIZE, SIZE, ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<normalize_col, SIZE, SIZE, ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<normalize_col, SIZE, SIZE, ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<broadcast_row, SIZE, SIZE, ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<broadcast_row, SIZE, SIZE, ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<broadcast_col, SIZE, SIZE, ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<broadcast_col, SIZE, SIZE, ducks::rt_layout::col>::run(results);
}

#endif