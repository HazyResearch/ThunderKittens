#include "tile.cuh"

#ifdef TEST_GROUP_SHARED_TILE

void group::shared::tile::tests(test_data &results) {
    std::cout << " --------------- Starting ops/group/shared/tile tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_SHARED_TILE_CONVERSIONS
    group::shared::tile::conversions::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/shared/tile/conversions tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_SHARED_TILE_MAPS
    group::shared::tile::maps::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/shared/tile/maps tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_SHARED_TILE_REDUCTIONS
    group::shared::tile::reductions::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/shared/tile/reductions tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif