#include "tile.cuh"

#ifdef TEST_WARP_REGISTER_TILE

void warp::reg::tile::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/register/tile tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_TILE_CONVERSIONS
    warp::reg::tile::conversions::tests(results);
#endif
#ifdef TEST_WARP_REGISTER_TILE_MAPS
    warp::reg::tile::maps::tests(results);
#endif
#ifdef TEST_WARP_REGISTER_TILE_REDUCTIONS
    warp::reg::tile::reductions::tests(results);
#endif
#ifdef TEST_WARP_REGISTER_TILE_MMA
    warp::reg::tile::mma::tests(results);
#endif



#ifdef TEST_WARP_REGISTER_TILE_COMPLEX
    std::cout << "\n --------------- Starting ops/warp/register/tile/complex tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_TILE_CONVERSIONS_COMPLEX
    warp::reg::tile::conversions::tests(results);
#endif
// #ifdef TEST_WARP_REGISTER_TILE_MAPS_COMPLEX
//     warp::reg::tile::maps::tests(results);
// #endif
// #ifdef TEST_WARP_REGISTER_TILE_MMA_COMPLEX
//     warp::reg::tile::mma::tests(results);
// #endif
// #ifdef TEST_WARP_REGISTER_TILE_MUL_COMPLEX
//     warp::reg::tile::mul::tests(results);
// #endif
#endif

}

#endif