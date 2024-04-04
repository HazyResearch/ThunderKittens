#include "memory.cuh"

#ifdef TEST_WARPGROUP_MEMORY

void warpgroup::memory::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/warpgroup/memory tests! --------------------\n" << std::endl;
#ifdef TEST_WARPGROUP_MEMORY_GLOBAL_TO_REGISTER
    warpgroup::memory::global_to_register::tests(results);
#endif
#ifdef TEST_WARPGROUP_MEMORY_GLOBAL_TO_SHARED
    warpgroup::memory::global_to_shared::tests(results);
#endif
#ifdef TEST_WARPGROUP_MEMORY_SHARED_TO_REGISTER
    warpgroup::memory::shared_to_register::tests(results);
#endif
}

#endif