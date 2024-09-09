#include "tma_multicast.cuh"

#ifdef TEST_WARP_MEMORY_VEC_TMA_MULTICAST

template<typename T>
struct test_load_multicast { // load with TMA, write out normally
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<NW == 1>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_multicast_load_vec_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_multicast_load_vec_gmem=half" :
                                                                                         "tma_multicast_load_vec_gmem=float";
    template<int S, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        int SIZE_DIV_4 = i_ref.size()/4;
        for(int i = 0; i < SIZE_DIV_4; i++) {
            for(int j = 0; j < 4; j++) {
                o_ref[i+j*SIZE_DIV_4] = i_ref[i];
            }
        }
    }
    template<int S, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st<dtype, S, S>> (&shared_vec) = al.allocate<kittens::row_vec<kittens::st<dtype, S, S>>>();
        int rank = blockIdx.x % 4;
        
        __shared__ kittens::barrier smem_barrier; 
        kittens::init_barrier(smem_barrier, 0, 1);
        // *************************************************************************************************
        // Doing it this way would also work, but I want to illustrate the use of the cluster::expect, too.
        // kittens::tma::expect<typeof(shared_vec)>(smem_barrier);
        // *************************************************************************************************
        kittens::tma::cluster::sync(); // ensure everyone has initialized their barrier

        if(rank == 0 && threadIdx.x == 0) { // only one block issues the multicast load for everyone
            for(int j = 0; j < 4; j++) { // expect on the whole block
                kittens::tma::cluster::expect(smem_barrier, j, shared_vec);
            }
            kittens::tma::cluster::load_async(shared_vec, input, {0, 0, 0, 0}, smem_barrier, 0b1111);
        }

        kittens::wait(smem_barrier, 0);
        kittens::store(output, shared_vec, {0, 0, rank, 0});
        kittens::tma::cluster::sync();
    }
};

template<typename Ker, typename T, int S, int NW, kittens::ducks::gl::all GL, typename... args>
static __global__ __cluster_dims__(4, 1, 1) void tmamulti_global_wrapper_1d(GL input, GL output) {
    Ker::template device_func<S, NW, GL, args...>(input, output);
}
template<typename test, int S, int NUM_WORKERS, typename... args>
struct tmamulti_wrapper_1d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<S,NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<S, NUM_WORKERS, args...>::value) {
            constexpr int SIZE = S*16 * 4; // 4 for additional TMA dimension
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, 1, 1, 4, S*16, kittens::sv<dtype, S>>;
            GL input(d_i, nullptr, nullptr, nullptr, nullptr);
            GL output(d_o, nullptr, nullptr, nullptr, nullptr);
            // run kernel
            cudaFuncSetAttribute(
                tmamulti_global_wrapper_1d<test, dtype, S, NUM_WORKERS, GL, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            tmamulti_global_wrapper_1d<test, dtype, S, NUM_WORKERS, GL, args...><<<4, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(input, output);
            // fill in correct results on cpu
            test::template host_func<S, NUM_WORKERS, GL, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, S*16);
            input.cleanup();
            output.cleanup();
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_S, int NW, typename... args>
using tmamulti_sweep_size_1d = loop_s<tmamulti_wrapper_1d, test, MAX_S, NW, MAX_S, args...>;

template<template<typename> typename test, int MAX_S=8, int NUM_WORKERS=1, typename... args>
struct tmamulti_sweep_gmem_type_1d {
    static void run(test_data &results) {
        tmamulti_sweep_size_1d<test<float>, MAX_S, NUM_WORKERS, args...>::run(results);
        tmamulti_sweep_size_1d<test<kittens::bf16>, MAX_S, NUM_WORKERS, args...>::run(results);
        tmamulti_sweep_size_1d<test<kittens::half>, MAX_S, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename> typename test, int MAX_S=8, typename... args> using tmamulti_sweep_gmem_type_1d_warp = tmamulti_sweep_gmem_type_1d<test, MAX_S, 1, args...>;

void warp::memory::vec::tma_multicast::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/tma_multicast tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    tmamulti_sweep_gmem_type_1d_warp<test_load_multicast, SIZE>::run(results);
}

#endif