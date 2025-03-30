#include "tile.cuh"

#ifdef TEST_WARP_MEMORY_TILE

void warp::memory::tile::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/memory/tile tests! ---------------\n" << std::endl;
#endif
#ifdef TEST_WARP_MEMORY_TILE_TMA
    warp::memory::tile::tma::tests(results);
#else
    std::cout << "Skipping ops/warp/memory/tile/tma tests!" << std::endl;
#endif
#ifdef TEST_WARP_MEMORY_TILE_TMA_MULTICAST
    warp::memory::tile::tma_multicast::tests(results);
#else
    std::cout << "Skipping ops/warp/memory/tile/tma_multicast tests!" << std::endl;
#endif
#ifdef TEST_WARP_MEMORY_TILE_DSMEM
    warp::memory::tile::dsmem::tests(results);
#else
    std::cout << "Skipping ops/warp/memory/tile/dsmem tests!" << std::endl;
#endif
}

#endif