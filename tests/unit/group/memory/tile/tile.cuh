#include "testing_flags.cuh"

#ifdef TEST_GROUP_MEMORY_TILE

#include "testing_commons.cuh"

#include "global_to_register.cuh"
#include "global_to_shared.cuh"
#include "shared_to_register.cuh"

namespace group {
namespace memory {
namespace tile {

void tests(test_data &results);

}
}
}

#endif