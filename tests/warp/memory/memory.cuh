#include "testing_flags.cuh"

#ifdef TEST_WARP_MEMORY

#include "testing_commons.cuh"

#include "tile/tile.cuh"
#include "vec/vec.cuh"

namespace warp {
namespace memory {

void tests(test_data &results);

}
}

#endif