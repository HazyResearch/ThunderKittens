#include "vec.cuh"

#ifdef TEST_WARP_MEMORY_VEC

void warp::memory::vec::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/memory/vec tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_MEMORY_VEC_PGL_TO_REGISTER
    warp::memory::vec::pgl_to_register::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_VEC_PGL_TO_SHARED
    warp::memory::vec::pgl_to_shared::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_VEC_GLOBAL_TO_REGISTER
    warp::memory::vec::global_to_register::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_VEC_GLOBAL_TO_SHARED
    warp::memory::vec::global_to_shared::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_VEC_SHARED_TO_REGISTER
    warp::memory::vec::shared_to_register::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_VEC_TMA
    warp::memory::vec::tma::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_VEC_TMA_MULTICAST
    warp::memory::vec::tma_multicast::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_VEC_DSMEM
    warp::memory::vec::dsmem::tests(results);
#endif
}

#endif