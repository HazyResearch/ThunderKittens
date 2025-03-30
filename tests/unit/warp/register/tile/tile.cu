#include "tile.cuh"

#ifdef TEST_WARP_REGISTER_TILE

void warp::reg::tile::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/register/tile tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_TILE_CONVERSIONS
    warp::reg::tile::conversions::tests(results);
#else
    std::cout << "Skipping ops/warp/register/tile/conversions tests!" << std::endl;
#endif
#ifdef TEST_WARP_REGISTER_TILE_MAPS
    warp::reg::tile::maps::tests(results);
#else
    std::cout << "Skipping ops/warp/register/tile/maps tests!" << std::endl;
#endif
#ifdef TEST_WARP_REGISTER_TILE_REDUCTIONS
    warp::reg::tile::reductions::tests(results);
#else
    std::cout << "Skipping ops/warp/register/tile/reductions tests!" << std::endl;
#endif



#ifdef TEST_WARP_REGISTER_TILE_COMPLEX
    std::cout << "\n --------------- Starting ops/warp/register/tile/complex tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_TILE_CONVERSIONS_COMPLEX
    warp::reg::tile::complex::conversions::tests(results);
#else
    std::cout << "Skipping ops/warp/register/tile/complex/conversions tests!" << std::endl;
#endif
#ifdef TEST_WARP_REGISTER_TILE_MAPS_COMPLEX
    warp::reg::tile::complex::maps::tests(results);
#else
    std::cout << "Skipping ops/warp/register/tile/complex/maps tests!" << std::endl;
#endif
#ifdef TEST_WARP_REGISTER_TILE_MUL_COMPLEX
    warp::reg::tile::complex::mul::tests(results);
#else
    std::cout << "Skipping ops/warp/register/tile/complex/mul tests!" << std::endl;
#endif
#endif

}

#endif