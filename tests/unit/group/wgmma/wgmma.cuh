#include "testing_flags.cuh"

#ifdef TEST_GROUP_WGMMA

#include "testing_commons.cuh"

#include "mma_fp32_bf16.cuh"
#include "mma_fp32_fp16.cuh"
#include "mma_fp16_fp16.cuh"
// #include "mma_fp32_fp32.cuh" TODO

namespace group {
namespace wgmma {

void tests(test_data &results);

}
}

#endif