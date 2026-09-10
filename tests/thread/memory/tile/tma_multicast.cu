#include "tma_multicast.cuh"

#ifdef TEST_THREAD_MEMORY_TILE_TMA_MULTICAST

template<typename T, int _CLUSTER_SIZE=4>
struct test_load_multicast { // load with TMA, write out normally
    using dtype = T;
    static constexpr int CLUSTER_SIZE = _CLUSTER_SIZE;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H*sizeof(dtype)*256<=kittens::MAX_SHARED_MEMORY-4096>; // S%4 ensures alignment
    static inline const std::string test_identifier =
        std::string(std::is_same_v<T, kittens::bf16> ? "tma_multicast_load_gmem=bf16" :
                    std::is_same_v<T, kittens::half> ? "tma_multicast_load_gmem=half" :
                                                       "tma_multicast_load_gmem=float") +
        (CLUSTER_SIZE > 16 ? "_wide_mask_bit16" : "");
    template<int H, int W, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        const int tile_elements = i_ref.size()/CLUSTER_SIZE;
        for(int i = 0; i < tile_elements; i++) {
            for(int j = 0; j < CLUSTER_SIZE; j++) {
                o_ref[i+j*tile_elements] = i_ref[i];
            }
        }
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, 16*H, 16*W> (&shared_tile) = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        const int rank = kittens::cluster_ctarank();
        
        __shared__ kittens::semaphore smem_semaphore; 
        kittens::warp::init_semaphore(smem_semaphore, 0, 1);
        // *************************************************************************************************
        // Doing it this way would also work, but I want to illustrate the use of the cluster::expect, too.
        // kittens::tma::expect(smem_semaphore, shared_tile);
        // *************************************************************************************************
        kittens::warp::tma::cluster::expect(smem_semaphore, shared_tile);
        kittens::everyone::tma::cluster::sync(); // ensure everyone has initialized their semaphore

        if(rank < 2 && threadIdx.x == 0) {
            kittens::cluster_mask_t mask = 0;
            #pragma unroll
            for (int dst = rank; dst < CLUSTER_SIZE; dst += 2)
                mask |= kittens::cluster_mask_t{1} << dst;
            kittens::tma::cluster::load_async(shared_tile, input, {0, 0, 0, 0}, smem_semaphore, mask);
        }

        __syncwarp();
        kittens::wait(smem_semaphore, 0);
        kittens::warp::store(output, shared_tile, {0, 0, rank, 0});
        kittens::everyone::tma::cluster::sync();
    }
};

template<typename Ker, typename T, int H, int W, int NW, kittens::ducks::gl::all GL, typename... args>
static __global__ void tmamulti_global_wrapper_2d(const __grid_constant__ GL input, const __grid_constant__ GL output) {
    Ker::template device_func<H, W, NW, GL, args...>(input, output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct tmamulti_wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H, W, NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int CLUSTER_SIZE = test::CLUSTER_SIZE;
            constexpr int SIZE = H*W*256 * CLUSTER_SIZE;
            constexpr int SMEM_BYTES = CLUSTER_SIZE > 16 ? 4096 : kittens::MAX_SHARED_MEMORY-1024;
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, 1, 1, CLUSTER_SIZE*16*H, 16*W, kittens::st<dtype, 16*H, 16*W>>;
            GL input(d_i, nullptr, nullptr, nullptr, nullptr);
            GL output(d_o, nullptr, nullptr, nullptr, nullptr);
            // run kernel
            cudaFuncSetAttribute(
                tmamulti_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_BYTES
            );
            if constexpr (CLUSTER_SIZE > 8) {
                cudaFuncSetAttribute(
                    tmamulti_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, args...>,
                    cudaFuncAttributeNonPortableClusterSizeAllowed,
                    1
                );
            }
            kittens::LaunchConfig<true> launch_config(
                dim3(CLUSTER_SIZE), dim3(NUM_WORKERS*32), SMEM_BYTES, nullptr, dim3(CLUSTER_SIZE)
            );
            cudaLaunchKernelEx(
                launch_config,
                tmamulti_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, args...>,
                input, output
            );
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GL, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*kittens::TILE_COL_DIM<dtype>);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
using tmamulti_sweep_size_2d = loop_h<tmamulti_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<template<typename> typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
struct tmamulti_sweep_gmem_type_2d {
    static void run(test_data &results) {
        tmamulti_sweep_size_2d<test<float>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        tmamulti_sweep_size_2d<test<kittens::bf16>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        tmamulti_sweep_size_2d<test<kittens::half>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename> typename test, int MAX_H=8, int MAX_W=8, typename... args> using tmamulti_sweep_gmem_type_2d_warp = tmamulti_sweep_gmem_type_2d<test, MAX_H, MAX_W, 1, args...>;

void thread::memory::tile::tma_multicast::tests(test_data &results) {
    std::cout << " ----- Starting ops/thread/memory/tile/tma_multicast tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    tmamulti_sweep_gmem_type_2d_warp<test_load_multicast, SIZE, SIZE>::run(results);
#ifdef KITTENS_SM107
    tmamulti_wrapper_2d<test_load_multicast<float, 17>, 1, 1, 1>::run(results);
#endif
    std::cout << std::endl;
}

#endif
