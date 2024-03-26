#include "testing_flags.cuh"

#ifdef TEST_WARP_REGISTER

#include "testing_commons.cuh"

#include "maps.cuh"
#include "reductions.cuh"
#include "mma.cuh"
#include "conversions.cuh"
#include "vec.cuh"

namespace warp {
namespace reg { // register is a reserved word

void tests(test_data &results);

}
}

#endif