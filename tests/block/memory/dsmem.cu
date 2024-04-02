#include "dsmem.cuh"

#ifdef TEST_BLOCK_MEMORY_DSMEM

void block::memory::dsmem::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/block/memory/dsmem tests! -----\n" << std::endl;
    constexpr int H_SIZE = INTENSITY_1 ? 4  :
                           INTENSITY_2 ? 8  : 
                           INTENSITY_3 ? 12  :
                           INTENSITY_4 ? 16 : -1;
    constexpr int W_SIZE = INTENSITY_1 ? 2  :
                           INTENSITY_2 ? 4  : 
                           INTENSITY_3 ? 8  :
                           INTENSITY_4 ? 16 : -1;
                           
    sweep_st_layout_dsmem_block<block::memory::dsmem::nextneighbor, H_SIZE, W_SIZE>::run(results);
}

#endif