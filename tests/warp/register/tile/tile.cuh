#include "testing_flags.cuh"

#ifdef TEST_WARP_REGISTER_TILE

#include "testing_commons.cuh"

#include "maps.cuh"
#include "reductions.cuh"
#include "mma.cuh"
#include "conversions.cuh"

namespace warp {
namespace reg {
namespace tile {

void tests(test_data &results);

}
}
}

#endif