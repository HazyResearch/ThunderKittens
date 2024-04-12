#include "mma.cuh"

#ifdef TEST_GROUP_WGMMA_MMA

struct test_mma {
    template<int H, int W, int NW, typename K, kittens::ducks::st_layout::wgmma_row L1, kittens::ducks::st_layout::wgmma_col L2>
    using valid = std::bool_constant<NW == 4 && H==4 && (2*W*H+W*K::value+H*K::value)<=64>;
    static inline const std::string test_identifier = "wgmma_mma";
    template<int H, int W, int NW, typename _K, kittens::ducks::st_layout::wgmma_row L1, kittens::ducks::st_layout::wgmma_col L2>
     __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
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
    template<int H, int W, int NW, typename _K, kittens::ducks::st_layout::wgmma_row L1, kittens::ducks::st_layout::wgmma_col L2>
    __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<1024> al((int*)&__shm[0]); 
        kittens::st_bf<H, K, L1> &a = al.allocate<kittens::st_bf<H, K, L1>>();
        kittens::st_bf<K, W, L2> &b = al.allocate<kittens::st_bf<K, W, L2>>();
        kittens::rt_fl<1, W> c;
        __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        if (threadIdx.x == 0) {init(&barrier, kittens::WARPGROUP_THREADS);}
        __syncthreads();
        kittens::warpgroup::load_async(a, input, K*16, barrier);
        kittens::warpgroup::load_async(b, input+a.num_elements, W*16, barrier);
        barrier.arrive_and_wait();
        kittens::warpgroup::mma_fence(c);
        kittens::warpgroup::mma_reset(c, a, b);
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(output, c, W*16);
    }
};
struct test_dot {
    template<int H, int W, int NW, typename K, kittens::ducks::st_layout::wgmma_row L1, kittens::ducks::st_layout::wgmma_row L2>
    using valid = std::bool_constant<NW == 4 && H==4 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "wgmma_dot";
    template<int H, int W, int NW, typename _K, kittens::ducks::st_layout::wgmma_row L1, kittens::ducks::st_layout::wgmma_row L2>
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
    template<int H, int W, int NW, typename _K, kittens::ducks::st_layout::wgmma_row L1, kittens::ducks::st_layout::wgmma_row L2>
    __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<1024> al((int*)&__shm[0]); 
        kittens::st_bf<H, K, L1> &a = al.allocate<kittens::st_bf<H, K, L1>>();
        kittens::st_bf<W, K, L2> &b = al.allocate<kittens::st_bf<W, K, L2>>();
        kittens::rt_fl<1, W> c;
        __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        if (threadIdx.x == 0) {init(&barrier, kittens::WARPGROUP_THREADS);}
        __syncthreads();
        kittens::warpgroup::load_async(a, input, K*16, barrier);
        kittens::warpgroup::load_async(b, input+a.num_elements, K*16, barrier);
        barrier.arrive_and_wait();
        kittens::warpgroup::mma_fence(c);
        kittens::warpgroup::dot_reset(c, a, b);
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(output, c, W*16);
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
template<typename test, int H, int MAX_W, int NUM_WORKERS=1, typename... args> using mma_sweep_width = loop_w<mma_wrapper_2d, test, H, MAX_W, NUM_WORKERS, H, MAX_W, args...>;
template<typename test, int MAX_W=4, typename... args> using mma_sweep_width_warpgroup = mma_sweep_width<test, 4, MAX_W, 4, args...>;

using namespace kittens::ducks::st_layout;
// If 1 and 3 work, the others likely will too.
using I1_t = std::integral_constant<int, 1>;
using I3_t = std::integral_constant<int, 3>;
void group::wgmma::mma::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warpgroup/wgmma/mma tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 1 :
                         INTENSITY_2 ? 2 : 
                         INTENSITY_3 ? 4 :
                         INTENSITY_4 ? 8 : -1;
    mma_sweep_width_warpgroup<test_mma, SIZE, I1_t, wgmma_row_0b , wgmma_col_t_0b >::run(results);
    // mma_sweep_width_warpgroup<test_mma, SIZE, I1_t, wgmma_row_0b , wgmma_col_t_32b>::run(results);
    mma_sweep_width_warpgroup<test_mma, SIZE, I1_t, wgmma_row_32b, wgmma_col_t_0b >::run(results);
    // mma_sweep_width_warpgroup<test_mma, SIZE, I1_t, wgmma_row_32b, wgmma_col_t_32b>::run(results);
    mma_sweep_width_warpgroup<test_mma, SIZE, I3_t, wgmma_row_0b , wgmma_col_t_0b >::run(results);
    // mma_sweep_width_warpgroup<test_mma, SIZE, I3_t, wgmma_row_0b , wgmma_col_t_32b>::run(results);
    mma_sweep_width_warpgroup<test_mma, SIZE, I3_t, wgmma_row_32b, wgmma_col_t_0b >::run(results);
    // mma_sweep_width_warpgroup<test_mma, SIZE, I3_t, wgmma_row_32b, wgmma_col_t_32b>::run(results);
    mma_sweep_width_warpgroup<test_dot, SIZE, I1_t, wgmma_row_0b , wgmma_row_0b >::run(results);
    mma_sweep_width_warpgroup<test_dot, SIZE, I1_t, wgmma_row_0b , wgmma_row_32b>::run(results);
    mma_sweep_width_warpgroup<test_dot, SIZE, I1_t, wgmma_row_32b, wgmma_row_0b >::run(results);
    mma_sweep_width_warpgroup<test_dot, SIZE, I1_t, wgmma_row_32b, wgmma_row_32b>::run(results);
    mma_sweep_width_warpgroup<test_dot, SIZE, I3_t, wgmma_row_0b , wgmma_row_0b >::run(results);
    mma_sweep_width_warpgroup<test_dot, SIZE, I3_t, wgmma_row_0b , wgmma_row_32b>::run(results);
    mma_sweep_width_warpgroup<test_dot, SIZE, I3_t, wgmma_row_32b, wgmma_row_0b >::run(results);
    mma_sweep_width_warpgroup<test_dot, SIZE, I3_t, wgmma_row_32b, wgmma_row_32b>::run(results);
}

#endif
