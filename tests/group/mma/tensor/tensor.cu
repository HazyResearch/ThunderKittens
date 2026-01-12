#include "tensor.cuh"

#ifdef TEST_GROUP_MMA_TENSOR

void group::mma::tensor:tests(test_data &results) {
    std::cout << " -------------------- Starting ops/group/mma/tensor tests! --------------------\n" << std::endl;
// #ifdef TEST_GROUP_MMA_TENSOR_MMA
//     group::mma::tensor:mma::tests(results);
// #else
    std::cout << "INFO: Skipping ops/group/mma/tensor/mma tests!\n" << std::endl;
// #endif
    std::cout << std::endl;
}

#endif