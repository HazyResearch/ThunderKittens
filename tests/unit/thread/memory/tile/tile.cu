#include "tile.cuh"

#ifdef TEST_THREAD_MEMORY_TILE

void thread::memory::tile::tests(test_data &results) {
    std::cout << " --------------- Starting ops/thread/memory/tile tests! ---------------\n" << std::endl;
#ifdef TEST_THREAD_MEMORY_TILE_TMA
    thread::memory::tile::tma::tests(results);
#else
    std::cout << "INFO: Skipping ops/thread/memory/tile/tma tests!\n" << std::endl;
#endif
#ifdef TEST_THREAD_MEMORY_TILE_TMA_MULTICAST
    thread::memory::tile::tma_multicast::tests(results);
#else
    std::cout << "INFO: Skipping ops/thread/memory/tile/tma_multicast tests!\n" << std::endl;
#endif
#ifdef TEST_THREAD_MEMORY_TILE_TMA_PGL
    thread::memory::tile::tma_pgl::tests(results);
#else
    std::cout << "INFO: Skipping ops/thread/memory/tile/tma_pgl tests!\n" << std::endl;
#endif
#ifdef TEST_THREAD_MEMORY_TILE_DSMEM
    thread::memory::tile::dsmem::tests(results);
#else
    std::cout << "INFO: Skipping ops/thread/memory/tile/dsmem tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif