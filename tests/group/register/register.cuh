#include "testing_flags.cuh"

#ifdef TEST_GROUP_REG

#include "testing_commons.cuh"

#include "tile/tile.cuh"
#include "vec/vec.cuh"

namespace group {
namespace reg { // register is a reserved word

void tests(test_data &results);

}
}

#endif