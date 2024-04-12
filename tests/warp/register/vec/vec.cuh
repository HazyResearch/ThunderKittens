#include "testing_flags.cuh"

#ifdef TEST_WARP_REGISTER_VEC

#include "testing_commons.cuh"

#include "maps.cuh"
#include "reductions.cuh"
#include "conversions.cuh"

namespace warp {
namespace reg {
namespace vec {

void tests(test_data &results);

}
}
}

#endif