#include "pseudo_fp32_fp8.cuh"

#ifdef TEST_GROUP_MMA_WARPGROUP_PSEUDO_FP32_FP8

namespace {

using D_dtype = float;

template<typename T>
static std::string fp8_name() {
    if constexpr (std::is_same_v<T, kittens::fp8e4m3>) {
        return "fp8e4m3";
    }
    else {
        return "fp8e5m2";
    }
}

template<typename T_AB, bool use_mma>
struct pseudo_shared_abt_test {
    using AB_dtype = T_AB;

    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<
        NW == 4 && H % 4 == 0 && W % 2 == 0 && K::value % 2 == 0 &&
        (H*K::value + W*K::value + H*W) <= 256
    >;

    static inline const std::string test_identifier =
        std::string("pseudo_warpgroup_") + (use_mma ? "mma_" : "mm_") + "ABt_fp32_" + fp8_name<T_AB>();

    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        constexpr int M = H * 16;
        constexpr int N = W * 16;
        constexpr int R = K * 16;
        constexpr int b_offset = H * K * 256;

        for(int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++) {
                float sum = 0.f;
                for(int k = 0; k < R; k++) {
                    sum += i_ref[i * R + k] * i_ref[b_offset + j * R + k];
                }
                o_ref[i * N + j] = sum;
            }
        }
    }

    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K>
    __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, GTL_C &c_output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator<128> al((int*)&__shm[0]);
        kittens::st<AB_dtype, 16*H, 16*K> &a = al.allocate<kittens::st<AB_dtype, 16*H, 16*K>>();
        kittens::st<AB_dtype, 16*W, 16*K> &b = al.allocate<kittens::st<AB_dtype, 16*W, 16*K>>();
        kittens::st<AB_dtype, 16*H, 16*W> &c_out_st = al.allocate<kittens::st<AB_dtype, 16*H, 16*W>>();
        kittens::rt<D_dtype, 4*H, 16*W> c;
        kittens::rt<AB_dtype, 4*H, 16*W> c_out_reg;

        kittens::warp::zero(c);
        kittens::warpgroup::load(a, a_input, {});
        kittens::warpgroup::load(b, b_input, {});
        __syncthreads();

        if constexpr (use_mma) {
            kittens::warpgroup::mma_ABt(c, a, b);
        }
        else {
            kittens::warpgroup::mm_ABt(c, a, b);
        }
        kittens::warpgroup::mma_async_wait();

        kittens::warp::copy(c_out_reg, c);
        kittens::warpgroup::store(c_out_st, c_out_reg);
        kittens::warp::store(c_output, c_out_st, {});
    }

    template<int H, int W, typename K>
    using make_a_layout = typename kittens::gl<AB_dtype, 1, 1, 16*H, 16*K::value>;
    template<int H, int W, typename K>
    using make_b_layout = typename kittens::gl<AB_dtype, 1, 1, 16*W, 16*K::value>;
    template<int H, int W, typename K>
    using make_c_layout = typename kittens::gl<AB_dtype, 1, 1, 16*H, 16*W>;
};

template<typename T_AB, bool use_mma>
struct pseudo_reg_shared_abt_test {
    using AB_dtype = T_AB;

    template<int H, int W, int NW, typename K>
    using valid = typename pseudo_shared_abt_test<T_AB, use_mma>::template valid<H, W, NW, K>;

    static inline const std::string test_identifier =
        std::string("pseudo_warpgroup_reg_") + (use_mma ? "mma_" : "mm_") + "ABt_fp32_" + fp8_name<T_AB>();

    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        pseudo_shared_abt_test<T_AB, use_mma>
            ::template host_func<H, W, NW, GTL_A, GTL_B, GTL_C, _K>(i_ref, o_ref);
    }

    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K>
    __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, GTL_C &c_output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator<128> al((int*)&__shm[0]);
        kittens::st<AB_dtype, 16*H, 16*K> &a = al.allocate<kittens::st<AB_dtype, 16*H, 16*K>>();
        kittens::st<AB_dtype, 16*W, 16*K> &b = al.allocate<kittens::st<AB_dtype, 16*W, 16*K>>();
        kittens::st<AB_dtype, 16*H, 16*W> &c_out_st = al.allocate<kittens::st<AB_dtype, 16*H, 16*W>>();
        kittens::rt<AB_dtype, 4*H, 16*K> a_reg;
        kittens::rt<D_dtype, 4*H, 16*W> c;
        kittens::rt<AB_dtype, 4*H, 16*W> c_out_reg;

        kittens::warp::zero(c);
        kittens::warpgroup::load(a, a_input, {});
        kittens::warpgroup::load(b, b_input, {});
        __syncthreads();

        kittens::warpgroup::load(a_reg, a);
        if constexpr (use_mma) {
            kittens::warpgroup::mma_ABt(c, a_reg, b);
        }
        else {
            kittens::warpgroup::mm_ABt(c, a_reg, b);
        }
        kittens::warpgroup::mma_async_wait();

        kittens::warp::copy(c_out_reg, c);
        kittens::warpgroup::store(c_out_st, c_out_reg);
        kittens::warp::store(c_output, c_out_st, {});
    }

    template<int H, int W, typename K>
    using make_a_layout = typename kittens::gl<AB_dtype, 1, 1, 16*H, 16*K::value>;
    template<int H, int W, typename K>
    using make_b_layout = typename kittens::gl<AB_dtype, 1, 1, 16*W, 16*K::value>;
    template<int H, int W, typename K>
    using make_c_layout = typename kittens::gl<AB_dtype, 1, 1, 16*H, 16*W>;
};

template<typename Ker, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename... args>
static __global__ void pseudo_fp8_mma_global_wrapper_2d(
    const GTL_A a_input,
    const GTL_B b_input,
    GTL_C c_output
) {
    Ker::template device_func<H, W, NW, GTL_A, GTL_B, GTL_C, args...>(a_input, b_input, c_output);
}

template<typename test, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct pseudo_fp8_mma_wrapper_2d {
    static void run(test_data& results) {
        using namespace kittens;
        constexpr int K = _K::value;
        test_info this_result;
        this_result.label = generate_test_name<H, W, NUM_WORKERS, _K, args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, _K, args...>::value) {
            using AB_dtype = typename test::AB_dtype;
            AB_dtype *d_i, *d_o;
            std::vector<float> i_ref((H + W) * K * 256);
            std::vector<float> o_ref(H * W * 256);
            initialize<AB_dtype>(&d_i, &d_o, i_ref, o_ref);

            using GTL_A = test::template make_a_layout<H, W, _K>;
            using GTL_B = test::template make_b_layout<H, W, _K>;
            using GTL_C = test::template make_c_layout<H, W, _K>;
            GTL_A a_input(d_i, nullptr, nullptr, nullptr, nullptr);
            GTL_B b_input(d_i + H*K*256, nullptr, nullptr, nullptr, nullptr);
            GTL_C c_output(d_o, nullptr, nullptr, nullptr, nullptr);

            cudaFuncSetAttribute(
                pseudo_fp8_mma_global_wrapper_2d<test, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY - 1024
            );
            pseudo_fp8_mma_global_wrapper_2d<test, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>
                <<<1, NUM_WORKERS * 32, kittens::MAX_SHARED_MEMORY - 1024>>>(a_input, b_input, c_output);

            test::template host_func<H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>(i_ref, o_ref);
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W * 16, 1.0f);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename test, int H, int MAX_W, int NUM_WORKERS=4, typename... args>
using pseudo_fp8_mma_sweep_width = loop_w<pseudo_fp8_mma_wrapper_2d, test, H, MAX_W, NUM_WORKERS, H, MAX_W, args...>;

using I2_t = std::integral_constant<int, 2>;
using I4_t = std::integral_constant<int, 4>;
using I6_t = std::integral_constant<int, 6>;

template<typename T_AB, typename _K, int SIZE>
static void run_h4_for_k(test_data &results) {
    pseudo_fp8_mma_sweep_width<pseudo_shared_abt_test<T_AB, true>, 4, SIZE, 4, _K>::run(results);
    pseudo_fp8_mma_sweep_width<pseudo_shared_abt_test<T_AB, false>, 4, SIZE, 4, _K>::run(results);
    pseudo_fp8_mma_sweep_width<pseudo_reg_shared_abt_test<T_AB, true>, 4, SIZE, 4, _K>::run(results);
    pseudo_fp8_mma_sweep_width<pseudo_reg_shared_abt_test<T_AB, false>, 4, SIZE, 4, _K>::run(results);
}

} // namespace

void group::mma::warpgroup::pseudo_fp32_fp8::tests(test_data &results) {
    std::cout << " ----- Starting ops/mma/warpgroup/pseudo_fp32_fp8 tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2 :
                         INTENSITY_2 ? 4 :
                         INTENSITY_3 ? 8 :
                         INTENSITY_4 ? 16 : -1;

    run_h4_for_k<kittens::fp8e4m3, I2_t, SIZE>(results);
    run_h4_for_k<kittens::fp8e4m3, I4_t, SIZE>(results);
    run_h4_for_k<kittens::fp8e4m3, I6_t, SIZE>(results);
    run_h4_for_k<kittens::fp8e5m2, I2_t, SIZE>(results);
    run_h4_for_k<kittens::fp8e5m2, I4_t, SIZE>(results);
    run_h4_for_k<kittens::fp8e5m2, I6_t, SIZE>(results);

    pseudo_fp8_mma_sweep_width<pseudo_shared_abt_test<kittens::fp8e4m3, true>, 8, SIZE, 4, I2_t>::run(results);
    pseudo_fp8_mma_sweep_width<pseudo_shared_abt_test<kittens::fp8e4m3, false>, 8, SIZE, 4, I2_t>::run(results);
    pseudo_fp8_mma_sweep_width<pseudo_reg_shared_abt_test<kittens::fp8e4m3, true>, 8, SIZE, 4, I2_t>::run(results);
    pseudo_fp8_mma_sweep_width<pseudo_reg_shared_abt_test<kittens::fp8e4m3, false>, 8, SIZE, 4, I2_t>::run(results);
    pseudo_fp8_mma_sweep_width<pseudo_shared_abt_test<kittens::fp8e5m2, true>, 8, SIZE, 4, I2_t>::run(results);
    pseudo_fp8_mma_sweep_width<pseudo_shared_abt_test<kittens::fp8e5m2, false>, 8, SIZE, 4, I2_t>::run(results);
    pseudo_fp8_mma_sweep_width<pseudo_reg_shared_abt_test<kittens::fp8e5m2, true>, 8, SIZE, 4, I2_t>::run(results);
    pseudo_fp8_mma_sweep_width<pseudo_reg_shared_abt_test<kittens::fp8e5m2, false>, 8, SIZE, 4, I2_t>::run(results);

    std::cout << std::endl;
}

#endif
