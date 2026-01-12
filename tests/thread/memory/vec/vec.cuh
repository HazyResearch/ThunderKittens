#include "testing_flags.cuh"

#ifdef TEST_THREAD_MEMORY_VEC

#include "testing_commons.cuh"

#include "tma.cuh"
#include "tma_multicast.cuh"
#include "tma_pgl.cuh"
#include "dsmem.cuh"

namespace thread {
namespace memory {
namespace vec {

void tests(test_data &results);

}
}
}

#endif