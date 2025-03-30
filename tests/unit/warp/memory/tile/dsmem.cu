#include "dsmem.cuh"
// #include <cooperative_groups.h>

#ifdef TEST_WARP_MEMORY_TILE_DSMEM

template<typename T>
struct test_dsmem { // load with dsmem, write out normally
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H*sizeof(dtype)*256*2<=kittens::MAX_SHARED_MEMORY-8192>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "dsmem_transfer_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "dsmem_transfer_gmem=half" :
                                                                                         "dsmem_transfer_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < H*W*256; j++) {
                o_ref[i*H*W*256 + j] = i_ref[((i+1)%4)*H*W*256 + j];
            }
        }
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL input, const GL output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, 16*H, 16*W> (&src_tile) = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        kittens::st<dtype, 16*H, 16*W> (&dst_tile) = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        
        kittens::warp::load(src_tile, input, kittens::coord{0, (int)blockIdx.x, 0, 0});

        __shared__ kittens::semaphore dsmem_semaphore;
        kittens::warp::init_semaphore(dsmem_semaphore, 0, 1);
        kittens::warp::tma::expect(dsmem_semaphore, dst_tile);
        kittens::everyone::tma::cluster::sync();
        kittens::warp::tma::cluster::store_async(dst_tile, src_tile, (blockIdx.x+3)%4, dsmem_semaphore);
        kittens::wait(dsmem_semaphore, 0);

        kittens::warp::store(output, dst_tile, kittens::coord{0, (int)blockIdx.x, 0, 0});
    }
};

template<typename Ker, typename T, int H, int W, int NW, kittens::ducks::gl::all GL, typename... args>
static __global__ __cluster_dims__(4, 1, 1) void dsmem_global_wrapper_2d(const GL input, const GL output) {
    Ker::template device_func<H, W, NW, GL, args...>(input, output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct dsmem_wrapper_2d {
    using dtype = gmem_dtype<test>;
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int D = 4;
            constexpr int SIZE = H*W*256 * D; // D for additional dsmem cluster dimension
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize<dtype, initializers::ARANGE>(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, 1, D, H*16, W*16>;
            GL input(d_i, nullptr, nullptr, nullptr, nullptr);
            GL output(d_o, nullptr, nullptr, nullptr, nullptr);
            // run kernel
            cudaFuncSetAttribute(
                dsmem_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            dsmem_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, args...><<<4, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(input, output);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GL, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
using dsmem_sweep_size_2d = loop_h<dsmem_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<template<typename> typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
struct dsmem_sweep_gmem_type_2d {
    static void run(test_data &results) {
        dsmem_sweep_size_2d<test<float>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        dsmem_sweep_size_2d<test<kittens::bf16>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        dsmem_sweep_size_2d<test<kittens::half>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename> typename test, int MAX_H=8, int MAX_W=8, typename... args> using dsmem_sweep_gmem_type_2d_warp = dsmem_sweep_gmem_type_2d<test, MAX_H, MAX_W, 1, args...>;


void warp::memory::tile::dsmem::tests(test_data &results) {
    std::cout << " ----- Starting ops/warp/memory/tile/dsmem tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    dsmem_sweep_gmem_type_2d_warp<test_dsmem, SIZE, SIZE>::run(results);
    std::cout << std::endl;
}

#endif