#include "testing_flags.cuh"

#ifdef TEST_WARPGROUP_MEMORY

#include "testing_commons.cuh"

#include "global_to_register.cuh"
#include "global_to_shared.cuh"
#include "shared_to_register.cuh"

namespace warpgroup {
namespace memory {

void tests(test_data &results);

}
}

#endif