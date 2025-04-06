#include "tile.cuh"

#ifdef TEST_WARP_MEMORY_TILE

void warp::memory::tile::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/memory/tile tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER
    warp::memory::tile::global_to_register::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED
    warp::memory::tile::global_to_shared::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_PGL_TO_REGISTER
    warp::memory::tile::pgl_to_register::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_PGL_TO_SHARED
    warp::memory::tile::pgl_to_shared::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER
    warp::memory::tile::shared_to_register::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_TMA
    warp::memory::tile::tma::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_TMA_MULTICAST
    warp::memory::tile::tma_multicast::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_TMA_PGL
    warp::memory::tile::tma_pgl::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_DSMEM
    warp::memory::tile::dsmem::tests(results);
#endif
}

#endif