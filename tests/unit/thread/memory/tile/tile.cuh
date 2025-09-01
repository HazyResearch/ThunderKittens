#include "testing_flags.cuh"

#ifdef TEST_THREAD_MEMORY_TILE

#include "testing_commons.cuh"

#include "tma.cuh"
#include "tma_multicast.cuh"
#include "tma_pgl.cuh"
#include "dsmem.cuh"

namespace thread {
namespace memory {
namespace tile {

void tests(test_data &results);

}
}
}

#endif