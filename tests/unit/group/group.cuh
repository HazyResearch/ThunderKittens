#include "testing_flags.cuh"

#ifdef TEST_GROUP

#include "testing_commons.cuh"

#include "memory/memory.cuh"
#include "register/register.cuh"
#include "shared/shared.cuh"
#include "mma/mma.cuh"

namespace group {

void tests(test_data &results);

}

#endif