#include "global_to_shared.cuh"

#ifdef TEST_WARP_MEMORY_GLOBAL_TO_SHARED

void warp::memory::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_st_layout_size_2d_warp<warp::memory::global_to_shared::load_store, SIZE, SIZE>::run(results);
    sweep_st_layout_size_2d_warp<warp::memory::global_to_shared::load_store_async, SIZE, SIZE>::run(results);

    sweep_size_1d_warp<warp::memory::global_to_shared::vec_load_store, SIZE>::run(results);
}

#endif