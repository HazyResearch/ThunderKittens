#include "complex.cuh"

#ifdef TEST_GROUP_MMA_WARPGROUP_COMPLEX

// I have concluded it is better to split up the different MMA's into different files
// substantially because otherwise compilation times will get out of hand.

void group::mma::warpgroup::complex::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/group/mma/warpgroup/complex tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_MMA_WARPGROUP_COMPLEX_FP32_BF16
    group::mma::warpgroup::complex::fp32_bf16::tests(results);
#else
    std::cout << "Skipping ops/group/mma/warpgroup/complex/fp32_bf16 tests!" << std::endl;
#endif
#ifdef TEST_GROUP_MMA_WARPGROUP_COMPLEX_FP32_FP16
    group::mma::warpgroup::complex::fp32_fp16::tests(results);
#else
    std::cout << "Skipping ops/group/mma/warpgroup/complex/fp32_fp16 tests!" << std::endl;
#endif
#ifdef TEST_GROUP_MMA_WARPGROUP_COMPLEX_FP16_FP16
    group::mma::warpgroup::complex::fp16_fp16::tests(results);
#else
    std::cout << "Skipping ops/group/mma/warpgroup/complex/fp16_fp16 tests!" << std::endl;
#endif
}

#endif