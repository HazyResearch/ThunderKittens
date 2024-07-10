#include "memory.cuh"

#ifdef TEST_GROUP_MEMORY

void group::memory::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/group/memory tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY_TILE
    group::memory::tile::tests(results);
#endif
#ifdef TEST_GROUP_MEMORY_VEC
    group::memory::vec::tests(results);
#endif
}

#endif