#include "testing_flags.cuh"

#ifdef TEST_GROUP_MMA

#include "testing_commons.cuh"

#include "warp/warp.cuh"
#include "warpgroup/warpgroup.cuh"
// #include "tensor/tensor.cuh"

namespace group {
namespace mma {

void tests(test_data &results);

}
}

#endif