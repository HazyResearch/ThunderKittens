#include "warpgroup.cuh"

#ifdef TEST_GROUP_MMA_WARPGROUP

// I have concluded it is better to split up the different MMA's into different files
// substantially because otherwise compilation times will get out of hand.

void group::mma::warpgroup::tests(test_data &results) {
    std::cout << " -------------------- Starting ops/group/mma/warpgroup tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_MMA_WARPGROUP_MMA_FP32_BF16
    group::mma::warpgroup::mma_fp32_bf16::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/mma/warpgroup/mma_fp32_bf16 tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_MMA_WARPGROUP_MMA_FP32_FP8
    group::mma::warpgroup::mma_fp32_fp8::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/mma/warpgroup/mma_fp32_fp8 tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_MMA_WARPGROUP_MMA_FP16_FP8
    group::mma::warpgroup::mma_fp16_fp8::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/mma/warpgroup/mma_fp16_fp8 tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_MMA_WARPGROUP_MMA_FP32_FP16
    group::mma::warpgroup::mma_fp32_fp16::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/mma/warpgroup/mma_fp32_fp16 tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP_MMA_WARPGROUP_MMA_FP16_FP16
    group::mma::warpgroup::mma_fp16_fp16::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/mma/warpgroup/mma_fp16_fp16 tests!\n" << std::endl;
#endif
// #ifdef TEST_GROUP_MMA_WARPGROUP_MMA_FP32_FP32 -- TODO
//     group::mma::warpgroup::mma_fp32_fp32::tests(results);
// #endif
#ifdef TEST_GROUP_MMA_WARPGROUP_COMPLEX
    group::mma::warpgroup::complex::tests(results);
#else
    std::cout << "INFO: Skipping ops/group/mma/warpgroup/complex tests!\n" << std::endl;
#endif
    std::cout << std::endl;
}

#endif