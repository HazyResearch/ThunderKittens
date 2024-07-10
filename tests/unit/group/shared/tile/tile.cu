#include "tile.cuh"

#ifdef TEST_GROUP_SHARED_TILE

void group::shared::tile::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/group/shared/tile tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_SHARED_TILE_CONVERSIONS
    group::shared::tile::conversions::tests(results);
#endif
#ifdef TEST_GROUP_SHARED_TILE_MAPS
    group::shared::tile::maps::tests(results);
#endif
#ifdef TEST_GROUP_SHARED_TILE_REDUCTIONS
    group::shared::tile::reductions::tests(results);
#endif
}

#endif