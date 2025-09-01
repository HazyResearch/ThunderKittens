#include "vec.cuh"

#ifdef TEST_GROUP_MEMORY_VEC

void group::memory::vec::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/group/memory/vec tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY_VEC_PGL_TO_REGISTER
    group::memory::vec::pgl_to_register::tests(results);
#endif
#ifdef TEST_GROUP_MEMORY_VEC_PGL_TO_SHARED
    group::memory::vec::pgl_to_shared::tests(results);
#endif
#ifdef TEST_GROUP_MEMORY_VEC_GLOBAL_TO_REGISTER
    group::memory::vec::global_to_register::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/memory/vec/global_to_register tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_MEMORY_VEC_GLOBAL_TO_SHARED
    group::memory::vec::global_to_shared::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/memory/vec/global_to_shared tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_MEMORY_VEC_SHARED_TO_REGISTER
    group::memory::vec::shared_to_register::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/memory/vec/shared_to_register tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif