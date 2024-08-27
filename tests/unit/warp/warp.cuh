#include "testing_flags.cuh"

#ifdef TEST_WARP

#include "testing_commons.cuh"

#include "memory/memory.cuh"
#include "register/register.cuh"
#include "shared/shared.cuh"

namespace warp {

void tests(test_data &results);

}

#endif