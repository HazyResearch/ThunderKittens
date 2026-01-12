#include "testing_flags.cuh"

#ifdef TEST_GROUP_MMA_WARPGROUP

#include "testing_commons.cuh"

#include "fp32_fp8.cuh"
#include "fp16_fp8.cuh"
#include "fp32_bf16.cuh"
#include "fp32_fp16.cuh"
#include "fp16_fp16.cuh"
// #include "mma_fp32_fp32.cuh" TODO
#include "complex/complex.cuh"

namespace group {
namespace mma {
namespace warpgroup{

void tests(test_data &results);

}
}
}

#endif