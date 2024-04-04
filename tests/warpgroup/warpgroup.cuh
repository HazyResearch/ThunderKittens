#include "testing_flags.cuh"

#ifdef TEST_WARPGROUP

#include "testing_commons.cuh"

#include "memory/memory.cuh"
#include "wgmma/wgmma.cuh"
#include "shared/shared.cuh"

namespace warpgroup {

void tests(test_data &results);

}

#endif