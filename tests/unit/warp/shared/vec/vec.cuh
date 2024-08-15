#include "testing_flags.cuh"

#ifdef TEST_WARP_SHARED_VEC

#include "testing_commons.cuh"

#include "maps.cuh"
#include "reductions.cuh"
#include "conversions.cuh"

namespace warp {
namespace shared {
namespace vec {

void tests(test_data &results);

}
}
}

#endif