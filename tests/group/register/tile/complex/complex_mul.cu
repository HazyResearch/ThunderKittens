#include "complex_mul.cuh"

#ifdef TEST_GROUP_REG_TILE_COMPLEX_MUL

struct test_mul {
    // element wise mult
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && H*W<=64>; // this is warp-level

    static inline const std::string test_identifier = "reg_complex_mul";
    template<int H, int W, int NW, typename CGL> __host__ static void host_func(
        const std::vector<float> &re_i_ref, 
        const std::vector<float> &im_i_ref,
        std::vector<float> &re_o_ref,
        std::vector<float> &im_o_ref
    ) {
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                // real parts for each element
                re_o_ref[i*16*W + j] += (
                    re_i_ref[i*16*W + j]*re_i_ref[i*16*W + j] - 
                    im_i_ref[i*16*W + j]*im_i_ref[i*16*W + j]
                );

                // imaginary parts for each element
                im_o_ref[i*16*W + j] += (
                    re_i_ref[i*16*W + j]*im_i_ref[i*16*W + j] + 
                    im_i_ref[i*16*W + j]*re_i_ref[i*16*W + j]
                );
            }
        }
    }
    template<int H, int W, int NW, typename CGL> __device__ static void device_func(const CGL &input, const CGL &output) {
        kittens::crt_bf<16*H, 16*W> a;
        kittens::crt_bf<16*H, 16*W> c;
        kittens::warp::load(a, input, {});
        kittens::warp::mul(c, a, a);
        kittens::warp::store(output, c, {});
    }
};

// Due to the strange sizes instantiated, we need a custom base wrapper here
template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct cmplx_mul_wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        printf("Running cmplx_mul_wrapper_2d\n"); 
        using namespace kittens;
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {

            // initialize
            kittens::bf16 *d_re_i, *d_im_i;
            kittens::bf16 *d_re_o, *d_im_o;
            std::vector<float> re_i_ref(H*W*256);
            std::vector<float> im_i_ref(H*W*256);
            std::vector<float> re_o_ref(H*W*256);
            std::vector<float> im_o_ref(H*W*256);
            initialize(&d_re_i, &d_re_o, re_i_ref, re_o_ref);
            initialize(&d_im_i, &d_im_o, im_i_ref, im_o_ref);
            using GL = typename kittens::gl<dtype, 1, 1, H*16, W*16>;
            using CGL = typename kittens::cgl<GL>;
            GL input_real(d_re_i, nullptr, nullptr, nullptr, nullptr);
            GL input_imag(d_im_i, nullptr, nullptr, nullptr, nullptr);
            GL output_real(d_re_o, nullptr, nullptr, nullptr, nullptr);
            GL output_imag(d_im_o, nullptr, nullptr, nullptr, nullptr);
            CGL input{input_real, input_imag};
            CGL output{output_real, output_imag};

            // run kernel
            cudaFuncSetAttribute(
                global_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, CGL, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY-1024
            );
            global_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, CGL, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY-1024>>>(
                input, output
            );

            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, CGL, args...>(
                re_i_ref, im_i_ref, 
                re_o_ref, im_o_ref
            );

            // check and cleanup
            test_result re_result = validate(
                d_re_i, d_re_o, re_i_ref, re_o_ref, this_result.label + "_real", W*16, 0.02
            ); 
            test_result im_result = validate(
                d_im_i, d_im_o, im_i_ref, im_o_ref, this_result.label + "_imag", W*16, 0.02
            );

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

template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using cmplx_mul_sweep_size = loop_h<cmplx_mul_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using cmplx_mul_sweep_size_warp = cmplx_mul_sweep_size<test, MAX_H, MAX_W, 1, args...>;

void group::reg::tile::complex::mul::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/register/tile/complex/mul tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  : 
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  : 
                         INTENSITY_4 ? 16 : -1;
    cmplx_mul_sweep_size_warp<test_mul, SIZE, SIZE>::run(results);
    std::cout << std::endl;
}

#endif


