#include "tile.cuh"

#ifdef TEST_GROUP_REG_TILE

void group::reg::tile::tests(test_data &results) {
    std::cout << " --------------- Starting ops/group/register/tile tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_REG_TILE_CONVERSIONS
    group::reg::tile::conversions::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/register/tile/conversions tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_REG_TILE_MAPS
    group::reg::tile::maps::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/register/tile/maps tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_REG_TILE_REDUCTIONS
    group::reg::tile::reductions::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/register/tile/reductions tests!\n" << std::endl;
#endif



#ifdef TEST_GROUP_REG_TILE_COMPLEX
    std::cout << " --------------- Starting ops/group/register/tile/complex tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_REG_TILE_COMPLEX_MUL
    group::reg::tile::complex::mul::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/register/tile/complex/mul tests!\n" << std::endl;
#endif
#endif

    std::cout << std::endl;
}

#endif