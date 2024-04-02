#include "testing_flags.cuh"

#ifdef TEST_BLOCK_MEMORY

#include "testing_commons.cuh"

#include "dsmem.cuh"
#include "global_to_shared.cuh"

namespace block {
namespace memory {

void tests(test_data &results);

}
}

#endif