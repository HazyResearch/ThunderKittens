#include "testing_flags.cuh"

#ifdef TEST_WARP_SHARED

#include "testing_commons.cuh"

#include "tile/tile.cuh"
#include "vec/vec.cuh"

namespace warp {
namespace shared {

void tests(test_data &results);

}
}

#endif