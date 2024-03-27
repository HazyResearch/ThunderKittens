#include "testing_flags.cuh"

#ifdef TEST_WARP_REGISTER_MMA

#include "testing_commons.cuh"

namespace warp {
namespace reg {
namespace mma {
struct mma {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma";
    template<int H, int W, int NW, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += i_ref[i*16*K + k]*i_ref[(256*H*K) + k*16*W + j];
                }
                o_ref[i*16*W + j] = sum;
            }
        }
    }
    template<int H, int W, int NW, typename _K> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        constexpr int K = _K::value;
        kittens::rt_bf<H, K> a;
        kittens::rt_bf<K, W, kittens::ducks::rt_layout::col> b;
        kittens::rt_fl<H, W> c;
        kittens::load(a, input, K*16);
        kittens::load(b, input+a.num_elements, W*16);
        kittens::zero(c);
        kittens::mma(c, a, b, c);
        kittens::store(output, c, W*16);
    }
};
struct dot {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_dot";
    template<int H, int W, int NW, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += i_ref[i*K*16+k]*i_ref[256*K*H + j*K*16+k];
                }
                o_ref[i*W*16+j] = sum;
            }
        }
    }
    template<int H, int W, int NW, typename _K> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        constexpr int K = _K::value;
        kittens::rt_bf<H, K> a;
        kittens::rt_bf<W, K> b;
        kittens::rt_fl<H, W> c;
        kittens::load(a, input, K*16);
        kittens::load(b, input+a.num_elements, K*16);
        kittens::zero(c);
        kittens::dot(c, a, b, c);
        kittens::store(output, c, W*16);
    }
};

// Due to the strange sizes instantiated, we need a custom base wrapper here
template<typename test, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct mma_wrapper_2d {
    static void run(test_data& results) {
        using namespace kittens;
        constexpr int K = _K::value;
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,_K,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, _K, args...>::value) {
            // initialize
            bf16 *d_i, *d_o;
            std::vector<float> i_ref((H+W)*K*256);
            std::vector<float> o_ref(H*W*256);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // run kernel
            cudaFuncSetAttribute(
                global_wrapper_2d<test, H, W, NUM_WORKERS, _K, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            global_wrapper_2d<test, H, W, NUM_WORKERS, _K, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(d_i, d_o);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, _K, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using mma_sweep_size = loop_h<mma_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using mma_sweep_size_warp = mma_sweep_size<test, MAX_H, MAX_W, 1, args...>;

void tests(test_data &results);
}
}
}

#endif