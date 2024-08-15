#include "tile.cuh"

#ifdef TEST_GROUP_MEMORY_TILE

void group::memory::tile::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/group/memory/tile tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_REGISTER
    group::memory::tile::global_to_register::tests(results);
#endif
#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED
    group::memory::tile::global_to_shared::tests(results);
#endif
#ifdef TEST_GROUP_MEMORY_TILE_SHARED_TO_REGISTER
    group::memory::tile::shared_to_register::tests(results);
#endif
#ifdef TEST_GROUP_MEMORY_TILE_TMA
    group::memory::tile::tma::tests(results);
#endif
#ifdef TEST_GROUP_MEMORY_TILE_DSMEM
    group::memory::tile::dsmem::tests(results);
#endif
}

#endif