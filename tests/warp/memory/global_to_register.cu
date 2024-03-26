#include "global_to_register.cuh"

#ifdef TEST_WARP_MEMORY_GLOBAL_TO_REGISTER

void warp::memory::global_to_register::tests(test_data &results) {
    std::cout << " ----- Starting ops/warp/memory/global_to_register tests! -----" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_2d_warp<warp::memory::global_to_register::load_store, SIZE, SIZE, ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<warp::memory::global_to_register::load_store, SIZE, SIZE, ducks::rt_layout::col>::run(results);

    sweep_size_1d_warp<warp::memory::global_to_register::vec_load_store, SIZE, ducks::rt_layout::row>::run(results);
    sweep_size_1d_warp<warp::memory::global_to_register::vec_load_store, SIZE, ducks::rt_layout::col>::run(results);
}

#endif