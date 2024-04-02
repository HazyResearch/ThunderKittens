#include "memory.cuh"

#ifdef TEST_BLOCK_MEMORY

void block::memory::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/block/memory tests! --------------------\n" << std::endl;
#ifdef TEST_BLOCK_MEMORY_GLOBAL_TO_SHARED
    block::memory::global_to_shared::tests(results);
#endif
#ifdef TEST_BLOCK_MEMORY_DSMEM
    block::memory::dsmem::tests(results);
#endif
}

#endif