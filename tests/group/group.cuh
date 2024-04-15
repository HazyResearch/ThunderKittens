#include "testing_flags.cuh"

#ifdef TEST_GROUP

#include "testing_commons.cuh"

#include "memory/memory.cuh"
#include "shared/shared.cuh"
#include "wgmma/wgmma.cuh"

namespace group {

void tests(test_data &results);

}

#endif