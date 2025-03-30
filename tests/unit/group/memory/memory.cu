#include "memory.cuh"

#ifdef TEST_GROUP_MEMORY

void group::memory::tests(test_data &results) {
    std::cout << " -------------------- Starting ops/group/memory tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY_TILE
    group::memory::tile::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/memory/tile tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_MEMORY_VEC
    group::memory::vec::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/memory/vec tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif