#include "testing_flags.cuh"

#ifdef TEST_WARP_SHARED_TILE

#include "testing_commons.cuh"

#include "conversions.cuh"
#include "maps.cuh"
#include "reductions.cuh"

namespace warp {
namespace shared {
namespace tile {

void tests(test_data &results);

}
}
}

#endif