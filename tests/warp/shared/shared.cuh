#include "testing_flags.cuh"

#ifdef TEST_WARP_SHARED

#include "testing_commons.cuh"

#include "conversions.cuh"
#include "vec.cuh"

namespace warp {
namespace shared { // register is a reserved word

void tests(test_data &results);

}
}

#endif