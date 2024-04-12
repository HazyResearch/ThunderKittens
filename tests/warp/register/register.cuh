#include "testing_flags.cuh"

#ifdef TEST_WARP_REGISTER

#include "testing_commons.cuh"

#include "tile/tile.cuh"
#include "vec/vec.cuh"

namespace warp {
namespace reg { // register is a reserved word

void tests(test_data &results);

}
}

#endif