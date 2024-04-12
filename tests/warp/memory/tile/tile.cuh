#include "testing_flags.cuh"

#ifdef TEST_WARP_MEMORY_TILE

#include "testing_commons.cuh"

#include "global_to_register.cuh"
#include "global_to_shared.cuh"
#include "shared_to_register.cuh"
#include "tma.cuh"

namespace warp {
namespace memory {
namespace tile {

void tests(test_data &results);

}
}
}

#endif