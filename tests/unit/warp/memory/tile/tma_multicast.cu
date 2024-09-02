#include "tma_multicast.cuh"

#ifdef TEST_WARP_MEMORY_TILE_TMA_MULTICAST

template<typename T>
struct test_load_multicast { // load with TMA, write out normally
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H*sizeof(dtype)*256<=kittens::MAX_SHARED_MEMORY-4096>; // S%4 ensures alignment
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_multicast_load_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_multicast_load_gmem=half" :
                                                                                         "tma_multicast_load_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gt::l::all GTL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        int SIZE_DIV_4 = i_ref.size()/4;
        for(int i = 0; i < SIZE_DIV_4; i++) {
            for(int j = 0; j < 4; j++) {
                o_ref[i+j*SIZE_DIV_4] = i_ref[i];
            }
        }
    }
    template<int H, int W, int NW, kittens::ducks::gt::l::all GTL>
    __device__ static void device_func(const GTL &input, const GTL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, H, W> (&shared_tile) = al.allocate<kittens::st<dtype, H, W>>();
        int rank = blockIdx.x % 4;
        
        __shared__ kittens::barrier smem_barrier; 
        kittens::init_barrier(smem_barrier, 0, 1);
        // *************************************************************************************************
        // Doing it this way would also work, but I want to illustrate the use of the cluster::expect, too.
        // kittens::tma::expect(smem_barrier, shared_tile);
        // *************************************************************************************************
        kittens::tma::cluster::sync(); // ensure everyone has initialized their barrier

        if(rank == 0 && threadIdx.x == 0) { // only one block issues the multicast load for everyone
            for(int j = 0; j < 4; j++) { // expect on the whole block
                kittens::tma::cluster::expect(smem_barrier, j, shared_tile);
            }
            kittens::tma::cluster::load_async(shared_tile, input, {0, 0, 0, 0}, smem_barrier, 0b1111);
        }

        kittens::wait(smem_barrier, 0);
        kittens::store(output, shared_tile, {0, 0, rank, 0});
        kittens::tma::cluster::sync();
    }
};

template<typename Ker, typename T, int H, int W, int NW, kittens::ducks::gt::l::all GTL, typename... args>
static __global__ __cluster_dims__(4, 1, 1) void tmamulti_global_wrapper_2d(GTL input, GTL output) {
    Ker::template device_func<H, W, NW, GTL, args...>(input, output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct tmamulti_wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H, W, NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int SIZE = H*W*256 * 4; // 4 for additional TMA dimension
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GTL = typename kittens::gt<dtype, H, W, true, true>::l<1, 1, 4, 1>;
            GTL input(d_i, nullptr, nullptr, nullptr, nullptr);
            GTL output(d_o, nullptr, nullptr, nullptr, nullptr);
            // run kernel
            cudaFuncSetAttribute(
                tmamulti_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GTL, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            tmamulti_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GTL, args...><<<4, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(input, output);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GTL, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*kittens::TILE_DIM);
            input.cleanup();
            output.cleanup();
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

void warp::memory::tile::tma_multicast::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/tma_multicast tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    tmamulti_sweep_gmem_type_2d_warp<test_load_multicast, SIZE, SIZE>::run(results);
}

#endif