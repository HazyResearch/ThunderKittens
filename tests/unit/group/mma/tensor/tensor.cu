#include "tensor.cuh"

#ifdef TEST_GROUP_MMA_TENSOR

void group::mma::tensor:tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/group/mma/tensor tests! --------------------\n" << std::endl;
// #ifdef TEST_GROUP_MMA_TENSOR_MMA
//     group::mma::tensor:mma::tests(results);
// #else
    std::cout << "Skipping ops/group/mma/tensor/mma tests!" << std::endl;
// #endif
}

#endif