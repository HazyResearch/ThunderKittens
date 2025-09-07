#include "tile.cuh"

#ifdef TEST_GROUP_MEMORY_TILE

void group::memory::tile::tests(test_data &results) {
    std::cout << " --------------- Starting ops/group/memory/tile tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_REGISTER
    group::memory::tile::global_to_register::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/memory/tile/global_to_register tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED
    group::memory::tile::global_to_shared::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/memory/tile/global_to_shared tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_MEMORY_TILE_PGL_TO_REGISTER
    group::memory::tile::pgl_to_register::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/memory/tile/pgl_to_register tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_MEMORY_TILE_SHARED_TO_REGISTER
    group::memory::tile::shared_to_register::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/memory/tile/shared_to_register tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif