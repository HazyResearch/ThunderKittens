#include "global_to_shared.cuh"

#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED

template<typename Ker, typename T, int H, int W, int NW, kittens::ducks::gl::all GL, typename... args>
static __global__ void g2s_global_wrapper_2d(const __grid_constant__ GL input, const __grid_constant__ GL output) {
    Ker::template device_func<H, W, NW, GL, args...>(input, output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename axis, typename... args>
struct g2s_wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS, axis, args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, axis, args...>::value) {
            constexpr int B = 3, D = 1, R = 4, C = 5;
            constexpr int SIZE = H*W*B*D*R*C*256;
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, -1, -1, -1, 16*C*W>;
            static_assert(axis::value==0 || axis::value==1 || axis::value==2, "Axis must be 0, 1, or 2.");
            GL input  (d_i, (axis::value==0?H*16:1)*B, (axis::value==1?H*16:1)*D, (axis::value==2?H*16:1)*R, nullptr);
            GL output (d_o, (axis::value==0?H*16:1)*B, (axis::value==1?H*16:1)*D, (axis::value==2?H*16:1)*R, nullptr);
            // run kernel
            cudaFuncSetAttribute(
                global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, axis, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, axis, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(input, output);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GL, axis, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using g2s_sweep_size_2d = loop_h<g2s_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<template<typename> typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
struct g2s_sweep_gmem_type_2d {
    static void run(test_data &results) {
        g2s_sweep_size_2d<test<float>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        g2s_sweep_size_2d<test<kittens::bf16>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        g2s_sweep_size_2d<test<kittens::half>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        #ifdef KITTENS_HOPPER
        g2s_sweep_size_2d<test<kittens::fp8e4m3>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        g2s_sweep_size_2d<test<kittens::fp8e5m2>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        #endif
    }
};

template<typename T>
struct group_shared_load_store {
    using dtype = T;
    template<int H, int W, int NW, typename axis> using valid = std::bool_constant<
        (H%NW==0 && W*H<=64)
        #ifdef KITTENS_HOPPER
        && ( (!std::is_same_v<T, kittens::fp8e4m3> && !std::is_same_v<T, kittens::fp8e5m2>) || W%2 == 0 )
        #endif 
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_shared_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_shared_loadstore_gmem=half" :
                                                      #ifdef KITTENS_HOPPER
                                                        std::is_same_v<T, kittens::fp8e4m3> ? "group_shared_loadstore_gmem=fp8e4m3" :
                                                        std::is_same_v<T, kittens::fp8e5m2> ? "group_shared_loadstore_gmem=fp8e5m2" :
                                                        #endif
                                                                                         "group_shared_loadstore_gmem=float";
    template<int H, int W, int NW, gl_t GL, typename axis> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, gl_t GL, typename axis> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]);
        using ST = kittens::st<dtype, 16*H, 16*W>;
        ST &shared_tile = al.allocate<ST>();
        int num_batches = axis::value==0?((int)input.batch()/shared_tile.rows):(int)input.batch();
        int num_depths = axis::value==1?((int)input.depth()/shared_tile.rows):(int)input.depth();
        int num_rows = axis::value==2?((int)input.rows()/shared_tile.rows):(int)input.rows();
        for(int i = 0; i < num_batches; i++)
            for(int j = 0; j < num_depths; j++)
                for(int k = 0; k < num_rows; k++)
                    for(int l = 0; l < (input.cols()/shared_tile.cols); l++) {
            G::template load <axis::value, false, ST, GL>(shared_tile, input,  {i, j, k, l});
            G::template store<axis::value, false, ST, GL>(output, shared_tile, {i, j, k, l});
        }
    }
};
template<typename T>
struct group_shared_load_store_async {
    using dtype = T;
    template<int H, int W, int NW, typename axis> using valid = std::bool_constant<
        (H%NW==0 && H*W<=64) && (!(sizeof(T) == 1) || W%2 == 0)
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_shared_loadstore_async_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_shared_loadstore_async_gmem=half" :
                                                      #ifdef KITTENS_HOPPER
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "group_shared_loadstore_async_gmem=fp8e4m3" :
                                                        std::is_same_v<T, kittens::fp8e5m2> ? "group_shared_loadstore_async_gmem=fp8e5m2" :
                                                        #endif
                                                                                         "group_shared_loadstore_async_gmem=float";
    template<int H, int W, int NW, gl_t GL, typename axis> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, gl_t GL, typename axis> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]);
        using ST = kittens::st<dtype, 16*H, 16*W>;
        ST &shared_tile = al.allocate<ST>();
        int num_batches = axis::value==0?((int)input.batch()/shared_tile.rows):(int)input.batch();
        int num_depths = axis::value==1?((int)input.depth()/shared_tile.rows):(int)input.depth();
        int num_rows = axis::value==2?((int)input.rows()/shared_tile.rows):(int)input.rows();
        for(int i = 0; i < num_batches; i++)
            for(int j = 0; j < num_depths; j++)
                for(int k = 0; k < num_rows; k++)
                    for(int l = 0; l < (input.cols()/shared_tile.cols); l++) {
            G::template load_async<axis::value, false, ST, GL>(shared_tile, input, {i, j, k, l});
            G::load_async_wait(0);
            G::template store<axis::value, false, ST, GL>(output, shared_tile, {i, j, k, l});
        }
    }
};

using I0_t = std::integral_constant<int, 0>;
using I1_t = std::integral_constant<int, 1>;
using I2_t = std::integral_constant<int, 2>;
void group::memory::tile::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/tile/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    g2s_sweep_gmem_type_2d<group_shared_load_store, SIZE, SIZE, 1, I0_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store, SIZE, SIZE, 2, I0_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store, SIZE, SIZE, 4, I0_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store, SIZE, 4, 12, I0_t>::run(results);

    g2s_sweep_gmem_type_2d<group_shared_load_store_async, SIZE, SIZE, 1, I0_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store_async, SIZE, SIZE, 2, I0_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store_async, SIZE, SIZE, 4, I0_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store_async, SIZE, 4, 12, I0_t>::run(results);

    g2s_sweep_gmem_type_2d<group_shared_load_store, SIZE, SIZE, 1, I1_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store, SIZE, SIZE, 2, I1_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store, SIZE, SIZE, 4, I1_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store, SIZE, 4, 12, I1_t>::run(results);

    g2s_sweep_gmem_type_2d<group_shared_load_store_async, SIZE, SIZE, 1, I1_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store_async, SIZE, SIZE, 2, I1_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store_async, SIZE, SIZE, 4, I1_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store_async, SIZE, 4, 12, I1_t>::run(results);

    g2s_sweep_gmem_type_2d<group_shared_load_store, SIZE, SIZE, 1, I2_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store, SIZE, SIZE, 2, I2_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store, SIZE, SIZE, 4, I2_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store, SIZE, 4, 12, I2_t>::run(results);

    g2s_sweep_gmem_type_2d<group_shared_load_store_async, SIZE, SIZE, 1, I2_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store_async, SIZE, SIZE, 2, I2_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store_async, SIZE, SIZE, 4, I2_t>::run(results);
    g2s_sweep_gmem_type_2d<group_shared_load_store_async, SIZE, 4, 12, I2_t>::run(results);
}

#endif