#include "mma.cuh"

#ifdef TEST_WARP_REGISTER_MMA

void warp::reg::mma::tests(test_data &results) {
    std::cout << " ----- Starting ops/warp/register/mma tests! -----" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    mma_sweep_size_warp<warp::reg::mma::mma, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    mma_sweep_size_warp<warp::reg::mma::mma, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    mma_sweep_size_warp<warp::reg::mma::mma, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    mma_sweep_size_warp<warp::reg::mma::mma, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
    mma_sweep_size_warp<warp::reg::mma::dot, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    mma_sweep_size_warp<warp::reg::mma::dot, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    mma_sweep_size_warp<warp::reg::mma::dot, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    mma_sweep_size_warp<warp::reg::mma::dot, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
}

#endif