#include "mma_fp16_fp8.cuh"

#ifdef TEST_GROUP_MMA_WARPGROUP_FP16_FP8

using dtype = kittens::fp8e4m3;

// shared+shared version
struct test_mma_ABt_fp16_fp8 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<
        (NW == 4 && H==4 && (2*W*H+W*K::value+H*K::value)<=256) && 
        (W%2 == 0)
    >; // this is warp-level
    static inline const std::string test_identifier = "warpgroup_mma_ABt_fp16_fp8";
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
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
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K>
    __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, GTL_C &c_output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, 16*H, 16*K> &a = al.allocate<kittens::st<dtype, 16*H, 16*K>>();
        kittens::st<dtype, 16*W, 16*K> &b = al.allocate<kittens::st<dtype, 16*W, 16*K>>();
        kittens::st<dtype, 16*4, 16*W> &c_out_st = al.allocate<kittens::st<dtype, 16*4, 16*W>>();
        kittens::rt<half, 16, 16*W> c;
        kittens::rt<dtype, 16, 16*W> c_out_reg;

        kittens::warpgroup::load_async(a, a_input, {}); // we don't have direct global to register right now
        kittens::warpgroup::load_async(b, b_input, {});
        kittens::warpgroup::load_async_wait(0);

        kittens::warpgroup::mm_ABt(c, a, b);
        kittens::warpgroup::mma_async_wait(); 

        kittens::copy(c_out_reg, c);  // we don't have direct global to register right now
        kittens::warpgroup::store(c_out_st, c_out_reg);
        kittens::warp::store(c_output, c_out_st, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<dtype, 1, 1, 16*H, 16*K::value>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<dtype, 1, 1, 16*W, 16*K::value>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<dtype, 1, 1, 16*H, 16*W>;
};

// register+shared version
struct reg_test_mma_ABt_fp16_fp8 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<
        (NW == 4 && H%4==0 && (2*W*H+W*K::value+H*K::value)<=256) && 
        (W%2 == 0)
    >; // this is warp-level
    static inline const std::string test_identifier = "warpgroup_reg_mma_ABt_fp16_fp8";
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
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
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K>
    __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, GTL_C &c_output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, 16*H, 16*K> &a = al.allocate<kittens::st<dtype, 16*H, 16*K>>();
        kittens::st<dtype, 16*W, 16*K> &b = al.allocate<kittens::st<dtype, 16*W, 16*K>>();
        kittens::st<dtype, 16*H, 16*W> &c_out_st = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        kittens::rt<dtype, H*4, 16*K> a_reg;
        kittens::rt<half,H*4, 16*W> c;
        kittens::rt<dtype, 16, 16*W> c_out;

        kittens::warp::load(a, a_input, {}); // we don't have direct global to register right now
        kittens::warp::load(b, b_input, {});
        kittens::warpgroup::load(a_reg, a);
        kittens::warpgroup::mma_fence(c);
        kittens::warpgroup::mm_ABt(c, a_reg, b);
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();

        kittens::copy(c_out, c); // we don't have direct global to register right now
        kittens::warpgroup::store(c_out_st, c_out);
        kittens::warp::store(c_output, c_out_st, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<dtype, 1, 1, 16*H, 16*K::value>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<dtype, 1, 1, 16*W, 16*K::value>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<dtype, 1, 1, 16*H, 16*W>;
};

// Due to the strange sizes instantiated, we need a custom base wrapper here
template<typename Ker, typename T, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename... args>
static __global__ void mma_global_wrapper_2d(const GTL_A a_input, const GTL_B b_input, GTL_C c_output) {
    Ker::template device_func<H, W, NW, GTL_A, GTL_B, GTL_C, args...>(a_input, b_input, c_output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct mma_wrapper_2d {
    static void run(test_data& results) {
        using namespace kittens;
        constexpr int K = _K::value;
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,_K,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, _K, args...>::value) {
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref((H+W)*K*256);
            std::vector<float> o_ref(H*W*256);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GTL_A = test::template make_a_layout<H, W, _K>;
            using GTL_B = test::template make_b_layout<H, W, _K>;
            using GTL_C = test::template make_c_layout<H, W, _K>;
            GTL_A a_input (d_i,           nullptr, nullptr, nullptr, nullptr);
            GTL_B b_input (d_i + H*K*256, nullptr, nullptr, nullptr, nullptr);
            GTL_C c_output(d_o,           nullptr, nullptr, nullptr, nullptr);
            // run kernel
            cudaFuncSetAttribute(
                mma_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            mma_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(a_input, b_input, c_output);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>(i_ref, o_ref);

            // check and cleanup
            // wgmma's sometimes produce small errors. this appears to be hardware. Larger threshold for fp8.
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16, 1); 
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int H, int MAX_W, int NUM_WORKERS=1, typename... args> using mma_sweep_width = loop_w<mma_wrapper_2d, test, H, MAX_W, NUM_WORKERS, H, MAX_W, args...>;
template<typename test, int MAX_W, typename... args> using mma_sweep_width_warpgroup = mma_sweep_width<test, 4, MAX_W, 4, args...>;
template<typename test, int MAX_W, typename... args> using mma_sweep_width_warpgroup_doubleheight = mma_sweep_width<test, 8, MAX_W, 4, args...>;

// If 1 and 3 work, the others likely will too.
using I2_t = std::integral_constant<int, 2>;
using I4_t = std::integral_constant<int, 4>;
using I6_t = std::integral_constant<int, 6>;
void group::mma::warpgroup::fp16_fp8::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/mma/warpgroup/fp16_fp8 tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2 :
                         INTENSITY_2 ? 4 : 
                         INTENSITY_3 ? 8 :
                         INTENSITY_4 ? 16 : -1;

    // shared+shared->reg
    mma_sweep_width_warpgroup<test_mma_ABt_fp16_fp8,  SIZE, I2_t>::run(results);
    mma_sweep_width_warpgroup<test_mma_ABt_fp16_fp8,  SIZE, I4_t>::run(results);
    mma_sweep_width_warpgroup<test_mma_ABt_fp16_fp8,  SIZE, I6_t>::run(results);

    // register+shared->reg
    mma_sweep_width_warpgroup<reg_test_mma_ABt_fp16_fp8, SIZE, I2_t>::run(results);
    mma_sweep_width_warpgroup<reg_test_mma_ABt_fp16_fp8, SIZE, I4_t>::run(results);
    mma_sweep_width_warpgroup<reg_test_mma_ABt_fp16_fp8, SIZE, I6_t>::run(results);
}

#endif
