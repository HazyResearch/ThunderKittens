#include "complex_mul.cuh"

#ifdef TEST_WARP_REGISTER_TILE_MUL_COMPLEX

// Need the wrapper so we can do the implicit const conversion for the inputs
template<typename Ker, typename T, int H, int W, int NW, typename... args>
static __global__ void global_cmplx_wrapper_2d(
    const T *re_input, const T *im_input, 
    T *re_output, T *im_output
) {
    Ker::template device_func<H, W, NW, args...>(
        re_input, im_input, 
        re_output, im_output
    );
}

struct test_mul {
    // element wise mult
    // template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level

    static inline const std::string test_identifier = "reg_complex_mul";
    template<int H, int W, int NW, typename _K> __host__ static void host_func(
        const std::vector<float> &re_i_ref, 
        const std::vector<float> &im_i_ref,
        std::vector<float> &re_o_ref, std::vector<float> &im_o_ref
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
    template<int H, int W, int NW, typename _K> __device__ static void device_func(
        const kittens::bf16 *input_re, const kittens::bf16 *input_im,
        kittens::bf16 *output_re, kittens::bf16 *output_im
    ) {
        kittens::rt_cmplx_bf<H, W> a;
        kittens::rt_cmplx_bf<H, W> c;
        kittens::load(a, input_re, input_im, W*16, W*16);
        kittens::mul(c, a, a);
        kittens::store(output_re, c.real, W*16);
        kittens::store(output_im, c.imag, W*16);
    }
};

// Due to the strange sizes instantiated, we need a custom base wrapper here
template<typename test, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct cmplx_mul_wrapper_2d {
    static void run(test_data& results) {
        printf("Running cmplx_mul_wrapper_2d\n"); 
        using namespace kittens;
        constexpr int K = _K::value;
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,_K,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, _K, args...>::value) {

            // initialize
            kittens::bf16 *d_re_i, *d_im_i;
            kittens::bf16 *d_re_o, *d_im_o;
            std::vector<float> re_i_ref(H*W*256);
            std::vector<float> im_i_ref(H*W*256);
            std::vector<float> re_o_ref(H*W*256);
            std::vector<float> im_o_ref(H*W*256);
            initialize(&d_re_i, &d_re_o, re_i_ref, re_o_ref);
            initialize(&d_im_i, &d_im_o, im_i_ref, im_o_ref);

            // run kernel
            cudaFuncSetAttribute(
                global_cmplx_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, _K, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );

            // Can't use global_wrapper_2d b/c it only accepts 2 params and we need 4 for complex-valued function
            global_cmplx_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, _K, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(
                d_re_i, d_im_i, 
                d_re_o, d_im_o
            );

            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, _K, args...>(
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

void warp::reg::tile::mul::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/mul tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  : 
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  : 
                         INTENSITY_4 ? 16 : -1;
    cmplx_mul_sweep_size_warp<test_mul, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
}

#endif


