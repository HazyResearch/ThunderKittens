#include "global_to_register.cuh"

#ifdef TEST_WARPGROUP_MEMORY_GLOBAL_TO_REGISTER

void warpgroup::memory::global_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warpgroup/memory/global_to_register tests! -----\n" << std::endl;
    constexpr int H_SIZE = INTENSITY_1 ? 4  :
                           INTENSITY_2 ? 8  : 
                           INTENSITY_3 ? 12  :
                           INTENSITY_4 ? 16 : -1;
    constexpr int W_SIZE = INTENSITY_1 ? 2  :
                           INTENSITY_2 ? 4  : 
                           INTENSITY_3 ? 8  :
                           INTENSITY_4 ? 16 : -1;
                         
    sweep_size_2d_warpgroup<warpgroup::memory::global_to_register::load_store, H_SIZE, W_SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warpgroup<warpgroup::memory::global_to_register::load_store, H_SIZE, W_SIZE, kittens::ducks::rt_layout::col>::run(results);

    sweep_size_1d_warpgroup<warpgroup::memory::global_to_register::vec_load_store, H_SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d_warpgroup<warpgroup::memory::global_to_register::vec_load_store, H_SIZE, kittens::ducks::rt_layout::col>::run(results);
}

#endif