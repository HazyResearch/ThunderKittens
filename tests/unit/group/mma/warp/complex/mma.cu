#include "mma.cuh"

#ifdef TEST_GROUP_MMA_WARP_COMPLEX_MMA

// Need the wrapper so we can do the implicit const conversion for the inputs
template<typename Ker, typename T, int H, int W, int NW, typename CGL1, typename CGL2, typename CGL3, typename... args>
static __global__ void global_cmplx_mma_wrapper_2d(const __grid_constant__ CGL1 a, const __grid_constant__ CGL2 b, const __grid_constant__ CGL3 c) {
    Ker::template device_func<H, W, NW, CGL1, CGL2, CGL3, args...>(a, b, c);
} 

struct test_cmplx_mma_AB {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "warp_cmplx_mma_AB";
    template<int H, int W, int NW, typename _K> __host__ static void host_func(
        const std::vector<float> &re_i_ref, const std::vector<float> &im_i_ref,
        std::vector<float> &re_o_ref, std::vector<float> &im_o_ref) {
        constexpr int K = _K::value;

        // ac
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[i*16*K + k]*re_i_ref[(256*H*K) + k*16*W + j];
                }
                re_o_ref[i*16*W + j] = sum;
            }
        }

        // bd
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[i*16*K + k]*im_i_ref[(256*H*K) + k*16*W + j];
                }
                re_o_ref[i*16*W + j] -= sum;
            }
        }

        // ad
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[i*16*K + k]*im_i_ref[(256*H*K) + k*16*W + j];
                }
                im_o_ref[i*16*W + j] = sum;
            }
        }

        // bc
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[i*16*K + k]*re_i_ref[(256*H*K) + k*16*W + j];
                }
                im_o_ref[i*16*W + j] += sum;
            }
        }
    }
    template<int H, int W, int NW, typename CGL1, typename CGL2, typename CGL3, typename _K> __device__ static void device_func(const CGL1 &A, const CGL2 &B, const CGL3 &C) {
        constexpr int K = _K::value;
        kittens::crt_bf<16*H, 16*K> a;
        kittens::crt_bf<16*K, 16*W, kittens::ducks::rt_layout::col> b;
        kittens::crt_fl<16*H, 16*W> c;
        kittens::warp::load(a, A, {});
        kittens::warp::load(b, B, {});
        kittens::zero(c);
        kittens::mma_AB(c, a, b, c);
        kittens::warp::store(C, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*K::value>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*K::value, 16*W>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*W>;
};

// Due to the strange sizes instantiated, we need a custom base wrapper here
template<typename test, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct cmplx_mma_wrapper_2d {
    static void run(test_data& results) {
        using namespace kittens;
        constexpr int K = _K::value;
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,_K,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, _K, args...>::value) {
            // initialize
            kittens::bf16 *d_re_i, *d_im_i;
            kittens::bf16 *d_re_o, *d_im_o;
            std::vector<float> re_i_ref((H+W)*K*256);
            std::vector<float> im_i_ref((H+W)*K*256);
            std::vector<float> re_o_ref(H*W*256);
            std::vector<float> im_o_ref(H*W*256);
            initialize(&d_re_i, &d_re_o, re_i_ref, re_o_ref);
            initialize(&d_im_i, &d_im_o, im_i_ref, im_o_ref);
            // create globals
            using GTL_A = test::template make_a_layout<H, W, _K>; using CGTL_A = kittens::cgl<GTL_A>; // complex-valued globals
            using GTL_B = test::template make_b_layout<H, W, _K>; using CGTL_B = kittens::cgl<GTL_B>;
            using GTL_C = test::template make_c_layout<H, W, _K>; using CGTL_C = kittens::cgl<GTL_C>;
            GTL_A a_input_real (d_re_i,           nullptr, nullptr, nullptr, nullptr);
            GTL_B b_input_real (d_re_i + H*K*256, nullptr, nullptr, nullptr, nullptr);
            GTL_C c_output_real(d_re_o,           nullptr, nullptr, nullptr, nullptr);
            GTL_A a_input_imag (d_im_i,           nullptr, nullptr, nullptr, nullptr);
            GTL_B b_input_imag (d_im_i + H*K*256, nullptr, nullptr, nullptr, nullptr);
            GTL_C c_output_imag(d_im_o,           nullptr, nullptr, nullptr, nullptr);
            CGTL_A a_input(a_input_real, a_input_imag);
            CGTL_B b_input(b_input_real, b_input_imag);
            CGTL_C c_output(c_output_real, c_output_imag);
            // run kernel
            cudaFuncSetAttribute(
                global_cmplx_mma_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, CGTL_A, CGTL_B, CGTL_C, _K, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            // Can't use global_wrapper_2d b/c it only accepts 2 params and we need 4 for complex-valued function
            global_cmplx_mma_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, CGTL_A, CGTL_B, CGTL_C, _K, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(a_input, b_input, c_output);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, _K, args...>(re_i_ref, im_i_ref, re_o_ref, im_o_ref);
            // check and cleanup
            test_result re_result = validate(d_re_i, d_re_o, re_i_ref, re_o_ref, this_result.label + "_real", W*16, 0.02); // mma's sometimes produce small errors. this appears to be hardware.
            test_result im_result = validate(d_im_i, d_im_o, im_i_ref, im_o_ref, this_result.label + "_imag", W*16, 0.02);
            if (re_result == test_result::PASSED && im_result == test_result::PASSED) {
                // TODO change back
                this_result.result = test_result::PASSED;
            } else {
                this_result.result = test_result::FAILED;
            }
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using cmplx_mma_sweep_size = loop_h<cmplx_mma_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using cmplx_mma_sweep_size_warp = cmplx_mma_sweep_size<test, MAX_H, MAX_W, 1, args...>;


void group::mma::warp::complex::mma::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/mma/warp/complex/mma tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    cmplx_mma_sweep_size_warp<test_cmplx_mma_AB, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    cmplx_mma_sweep_size_warp<test_cmplx_mma_AB, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    cmplx_mma_sweep_size_warp<test_cmplx_mma_AB, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    cmplx_mma_sweep_size_warp<test_cmplx_mma_AB, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
    std::cout << std::endl;
}

#endif