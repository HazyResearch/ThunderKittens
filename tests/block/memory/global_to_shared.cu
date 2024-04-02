#include "global_to_shared.cuh"

#ifdef TEST_BLOCK_MEMORY_GLOBAL_TO_SHARED

void block::memory::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/block/memory/global_to_shared tests! -----\n" << std::endl;
    constexpr int H_SIZE = INTENSITY_1 ? 4  :
                           INTENSITY_2 ? 8  : 
                           INTENSITY_3 ? 12  :
                           INTENSITY_4 ? 16 : -1;
    constexpr int W_SIZE = INTENSITY_1 ? 2  :
                           INTENSITY_2 ? 4  : 
                           INTENSITY_3 ? 8  :
                           INTENSITY_4 ? 16 : -1;
                           
    sweep_st_layout_size_2d_block<block::memory::global_to_shared::load_store, H_SIZE, W_SIZE>::run(results);
    sweep_st_layout_size_2d_block<block::memory::global_to_shared::load_store_async, H_SIZE, W_SIZE>::run(results);

    sweep_size_1d_block<block::memory::global_to_shared::vec_load_store, H_SIZE>::run(results);
}

#endif