#include "pseudo_fp32_bf16.cuh"

#ifdef TEST_GROUP_MMA_WARPGROUP_PSEUDO_FP32_BF16

namespace {

constexpr const char* op_name(int trans_a, int trans_b) {
    return trans_a == kittens::transpose::N
        ? (trans_b == kittens::transpose::N ? "AB" : "ABt")
        : (trans_b == kittens::transpose::N ? "AtB" : "AtBt");
}

template<int trans_a, int trans_b, bool use_mma>
struct pseudo_shared_test {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H % 4 == 0 && (H*K::value + W*K::value) <= 256>;

    static inline const std::string test_identifier = [] {
        return std::string("pseudo_warpgroup_") + (use_mma ? "mma_" : "mm_") + op_name(trans_a, trans_b) + "_fp32_bf16";
    }();

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
                    float a_val;
                    if constexpr (trans_a == kittens::transpose::N) {
                        a_val = i_ref[i * R + k];
                    }
                    else {
                        a_val = i_ref[k * M + i];
                    }

                    float b_val;
                    if constexpr (trans_b == kittens::transpose::N) {
                        b_val = i_ref[b_offset + k * N + j];
                    }
                    else {
                        b_val = i_ref[b_offset + j * R + k];
                    }
                    sum += a_val * b_val;
                }
                o_ref[i * N + j] = sum;
            }
        }
    }

    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K>
    __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, GTL_C &c_output) {
        constexpr int K = _K::value;
        using A_ST = std::conditional_t<
            trans_a == kittens::transpose::N,
            kittens::st_bf<16*H, 16*K>,
            kittens::st_bf<16*K, 16*H>
        >;
        using B_ST = std::conditional_t<
            trans_b == kittens::transpose::N,
            kittens::st_bf<16*K, 16*W>,
            kittens::st_bf<16*W, 16*K>
        >;

        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator<128> al((int*)&__shm[0]);
        A_ST &a = al.allocate<A_ST>();
        B_ST &b = al.allocate<B_ST>();
        kittens::rt_fl<4*H, 16*W> c;

        kittens::warp::zero(c);
        kittens::warpgroup::load(a, a_input, {});
        kittens::warpgroup::load(b, b_input, {});
        __syncthreads();

        kittens::warpgroup::mma_fence(c);
        if constexpr (use_mma) {
            if constexpr (trans_a == kittens::transpose::N && trans_b == kittens::transpose::N) {
                kittens::warpgroup::mma_AB(c, a, b);
            }
            else if constexpr (trans_a == kittens::transpose::N && trans_b == kittens::transpose::T) {
                kittens::warpgroup::mma_ABt(c, a, b);
            }
            else if constexpr (trans_a == kittens::transpose::T && trans_b == kittens::transpose::N) {
                kittens::warpgroup::mma_AtB(c, a, b);
            }
            else {
                kittens::warpgroup::mma_AtBt(c, a, b);
            }
        }
        else {
            if constexpr (trans_a == kittens::transpose::N && trans_b == kittens::transpose::N) {
                kittens::warpgroup::mm_AB(c, a, b);
            }
            else if constexpr (trans_a == kittens::transpose::N && trans_b == kittens::transpose::T) {
                kittens::warpgroup::mm_ABt(c, a, b);
            }
            else if constexpr (trans_a == kittens::transpose::T && trans_b == kittens::transpose::N) {
                kittens::warpgroup::mm_AtB(c, a, b);
            }
            else {
                kittens::warpgroup::mm_AtBt(c, a, b);
            }
        }
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();

        kittens::warpgroup::store(c_output, c, {});
    }

    template<int H, int W, typename K>
    using make_a_layout = typename std::conditional_t<
        trans_a == kittens::transpose::N,
        kittens::gl<kittens::bf16, 1, 1, 16*H, 16*K::value>,
        kittens::gl<kittens::bf16, 1, 1, 16*K::value, 16*H>
    >;
    template<int H, int W, typename K>
    using make_b_layout = typename std::conditional_t<
        trans_b == kittens::transpose::N,
        kittens::gl<kittens::bf16, 1, 1, 16*K::value, 16*W>,
        kittens::gl<kittens::bf16, 1, 1, 16*W, 16*K::value>
    >;
    template<int H, int W, typename K>
    using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*W>;
};

template<int trans_b, bool use_mma>
struct pseudo_reg_shared_test {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H % 4 == 0 && (H*K::value + W*K::value) <= 256>;

    static inline const std::string test_identifier = [] {
        return std::string("pseudo_warpgroup_reg_") + (use_mma ? "mma_" : "mm_") + op_name(kittens::transpose::N, trans_b) + "_fp32_bf16";
    }();

    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        pseudo_shared_test<kittens::transpose::N, trans_b, use_mma>
            ::template host_func<H, W, NW, GTL_A, GTL_B, GTL_C, _K>(i_ref, o_ref);
    }

    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K>
    __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, GTL_C &c_output) {
        constexpr int K = _K::value;
        using A_ST = kittens::st_bf<16*H, 16*K>;
        using B_ST = std::conditional_t<
            trans_b == kittens::transpose::N,
            kittens::st_bf<16*K, 16*W>,
            kittens::st_bf<16*W, 16*K>
        >;

        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator<128> al((int*)&__shm[0]);
        A_ST &a = al.allocate<A_ST>();
        B_ST &b = al.allocate<B_ST>();
        kittens::rt_bf<4*H, 16*K> a_reg;
        kittens::rt_fl<4*H, 16*W> c;

        kittens::warp::zero(c);
        kittens::warpgroup::load(a, a_input, {});
        kittens::warpgroup::load(b, b_input, {});
        __syncthreads();

        kittens::warpgroup::load(a_reg, a);
        kittens::warpgroup::mma_fence(c);
        if constexpr (use_mma) {
            if constexpr (trans_b == kittens::transpose::N) {
                kittens::warpgroup::mma_AB(c, a_reg, b);
            }
            else {
                kittens::warpgroup::mma_ABt(c, a_reg, b);
            }
        }
        else {
            if constexpr (trans_b == kittens::transpose::N) {
                kittens::warpgroup::mm_AB(c, a_reg, b);
            }
            else {
                kittens::warpgroup::mm_ABt(c, a_reg, b);
            }
        }
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();

        kittens::warpgroup::store(c_output, c, {});
    }

    template<int H, int W, typename K>
    using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*K::value>;
    template<int H, int W, typename K>
    using make_b_layout = typename std::conditional_t<
        trans_b == kittens::transpose::N,
        kittens::gl<kittens::bf16, 1, 1, 16*K::value, 16*W>,
        kittens::gl<kittens::bf16, 1, 1, 16*W, 16*K::value>
    >;
    template<int H, int W, typename K>
    using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*W>;
};

template<typename Ker, typename T, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename... args>
static __global__ void pseudo_mma_global_wrapper_2d(const GTL_A a_input, const GTL_B b_input, GTL_C c_output) {
    Ker::template device_func<H, W, NW, GTL_A, GTL_B, GTL_C, args...>(a_input, b_input, c_output);
}

template<typename test, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct pseudo_mma_wrapper_2d {
    static void run(test_data& results) {
        using namespace kittens;
        constexpr int K = _K::value;
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,_K,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, _K, args...>::value) {
            bf16 *d_i, *d_o;
            std::vector<float> i_ref((H+W)*K*256);
            std::vector<float> o_ref(H*W*256);
            initialize(&d_i, &d_o, i_ref, o_ref);

            using GTL_A = test::template make_a_layout<H, W, _K>;
            using GTL_B = test::template make_b_layout<H, W, _K>;
            using GTL_C = test::template make_c_layout<H, W, _K>;
            GTL_A a_input (d_i,           nullptr, nullptr, nullptr, nullptr);
            GTL_B b_input (d_i + H*K*256, nullptr, nullptr, nullptr, nullptr);
            GTL_C c_output(d_o,           nullptr, nullptr, nullptr, nullptr);

            cudaFuncSetAttribute(
                pseudo_mma_global_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY-1024
            );
            pseudo_mma_global_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>
                <<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY-1024>>>(a_input, b_input, c_output);

            test::template host_func<H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>(i_ref, o_ref);
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16, 0.05);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename test, int H, int MAX_W, int NUM_WORKERS=4, typename... args>
using pseudo_mma_sweep_width = loop_w<pseudo_mma_wrapper_2d, test, H, MAX_W, NUM_WORKERS, H, MAX_W, args...>;

template<typename _K, int SIZE>
static void run_all_for_k(test_data &results) {
    pseudo_mma_sweep_width<pseudo_shared_test<kittens::transpose::N, kittens::transpose::N, true>,  4, SIZE, 4, _K>::run(results);
    pseudo_mma_sweep_width<pseudo_shared_test<kittens::transpose::N, kittens::transpose::N, false>, 4, SIZE, 4, _K>::run(results);
    pseudo_mma_sweep_width<pseudo_shared_test<kittens::transpose::N, kittens::transpose::T, true>,  4, SIZE, 4, _K>::run(results);
    pseudo_mma_sweep_width<pseudo_shared_test<kittens::transpose::N, kittens::transpose::T, false>, 4, SIZE, 4, _K>::run(results);
    pseudo_mma_sweep_width<pseudo_shared_test<kittens::transpose::T, kittens::transpose::N, true>,  4, SIZE, 4, _K>::run(results);
    pseudo_mma_sweep_width<pseudo_shared_test<kittens::transpose::T, kittens::transpose::N, false>, 4, SIZE, 4, _K>::run(results);
    pseudo_mma_sweep_width<pseudo_shared_test<kittens::transpose::T, kittens::transpose::T, true>,  4, SIZE, 4, _K>::run(results);
    pseudo_mma_sweep_width<pseudo_shared_test<kittens::transpose::T, kittens::transpose::T, false>, 4, SIZE, 4, _K>::run(results);

    pseudo_mma_sweep_width<pseudo_shared_test<kittens::transpose::N, kittens::transpose::N, true>,  8, SIZE, 4, _K>::run(results);
    pseudo_mma_sweep_width<pseudo_shared_test<kittens::transpose::N, kittens::transpose::N, false>, 8, SIZE, 4, _K>::run(results);

    pseudo_mma_sweep_width<pseudo_reg_shared_test<kittens::transpose::N, true>,  4, SIZE, 4, _K>::run(results);
    pseudo_mma_sweep_width<pseudo_reg_shared_test<kittens::transpose::N, false>, 4, SIZE, 4, _K>::run(results);
    pseudo_mma_sweep_width<pseudo_reg_shared_test<kittens::transpose::T, true>,  4, SIZE, 4, _K>::run(results);
    pseudo_mma_sweep_width<pseudo_reg_shared_test<kittens::transpose::T, false>, 4, SIZE, 4, _K>::run(results);
}

} // namespace

void group::mma::warpgroup::pseudo_fp32_bf16::tests(test_data &results) {
    std::cout << " ----- Starting ops/mma/warpgroup/pseudo_fp32_bf16 tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 1 :
                         INTENSITY_2 ? 2 :
                         INTENSITY_3 ? 4 :
                         INTENSITY_4 ? 4 : -1;
    using I1_t = std::integral_constant<int, 1>;
    using I2_t = std::integral_constant<int, 2>;
    using I3_t = std::integral_constant<int, 3>;

    run_all_for_k<I1_t, SIZE>(results);
    run_all_for_k<I2_t, SIZE>(results);
    run_all_for_k<I3_t, SIZE>(results);

    std::cout << std::endl;
}

#endif
