#include "complex_wgmma.cuh"

#ifdef TEST_GROUP_WGMMA

// I have concluded it is better to split up the different MMA's into different files
// substantially because otherwise compilation times will get out of hand.

void group::wgmma::complex::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/group/wgmma/complex tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_COMPLEX_WGMMA_MMA_FP32_BF16
    group::wgmma::complex::complex_mma_fp32_bf16::tests(results);
#endif
#ifdef TEST_GROUP_COMPLEX_WGMMA_MMA_FP32_FP16
    group::wgmma::complex::complex_mma_fp32_fp16::tests(results);
#endif
#ifdef TEST_GROUP_COMPLEX_WGMMA_MMA_FP16_FP16
    group::wgmma::complex::complex_mma_fp16_fp16::tests(results);
#endif
}

#endif