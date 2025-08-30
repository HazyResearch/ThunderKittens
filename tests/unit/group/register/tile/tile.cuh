#include "testing_flags.cuh"

#ifdef TEST_GROUP_REG_TILE

#include "testing_commons.cuh"

#include "maps.cuh"
#include "reductions.cuh"
#include "conversions.cuh"

#include "complex/complex_mul.cuh"

namespace group {
namespace reg {
namespace tile {

void tests(test_data &results);

}
}
}

#endif