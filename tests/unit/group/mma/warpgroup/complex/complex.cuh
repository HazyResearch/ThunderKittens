#include "testing_flags.cuh"

#ifdef TEST_GROUP_MMA_WARPGROUP

#include "testing_commons.cuh"

#include "fp32_bf16.cuh"
#include "fp32_fp16.cuh"
#include "fp16_fp16.cuh"

namespace group {
namespace mma {
namespace warpgroup {
namespace complex {
void tests(test_data &results);
}
}
}
}

#endif