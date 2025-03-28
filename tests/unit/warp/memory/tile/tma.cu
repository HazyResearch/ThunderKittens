#include "tma.cuh"

#ifdef TEST_WARP_MEMORY_TILE_TMA

template<typename T>
struct test_load { // load with TMA, write out normally
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant< 
        ( NW == 1 && W*H*sizeof(dtype)*256*4<=kittens::MAX_SHARED_MEMORY-1024 ) && ( ( sizeof(T) != 1 ) || W%2 == 0 ) 
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_load_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_load_gmem=half" :
                                                        #ifdef KITTENS_HOPPER
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "tma_load_gmem=fp8e4m3":
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "tma_load_gmem=fp8e5m2":
                                                        #endif
                                                                                         "tma_load_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]);
        kittens::st<T, 16*H, 16*W> (&shared_tile)[2][2] = al.allocate<kittens::st<T, 16*H, 16*W>, 2, 2>(); // assuming compile-time known dimensions
        
        __shared__ kittens::semaphore smem_semaphore; 
        kittens::warp::init_semaphore(smem_semaphore, 0, 1);
        __syncwarp();
        for(int a = 0; a < input.batch(); a++) for(int b = 0; b < input.depth(); b++) {
            kittens::warp::tma::expect_bytes(smem_semaphore, kittens::size_bytes<kittens::st<T, 16*H, 16*W>> * 2 * 2);
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                kittens::warp::tma::load_async(shared_tile[i][j], input, {a, b, i, j}, smem_semaphore);
            }
            kittens::wait(smem_semaphore, (a*input.depth()+b)%2);
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                kittens::store(output, shared_tile[i][j], {a, b, i, j});
            }
        }
    }
};

template<typename T>
struct test_load_oob { // load oob memory via TMA
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<
        ( NW == 1 && W*H*sizeof(dtype)*256*4<=kittens::MAX_SHARED_MEMORY-1024 ) && ( ( sizeof(T) != 1 ) || W%2 == 0 ) 
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_load_oob_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_load_oob_gmem=half" :
                                                                                         "tma_load_oob_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // everything should be zero
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = 0;
        }
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]);

        kittens::st<T, 16*H, 16*W> (&shared_tile)[2][2] = al.allocate<kittens::st<T, 16*H, 16*W>, 2, 2>(); // assuming compile-time known dimensions

        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
            kittens::rt<T, 16*H, 16*W> reg_tile;
            kittens::one(reg_tile);
            kittens::store(shared_tile[i][j], reg_tile);
        }
        __syncthreads(); 
        
        __shared__ kittens::semaphore smem_semaphore; 
        kittens::warp::init_semaphore(smem_semaphore, 0, 1);
        __syncwarp();
        for(int a = 0; a < input.batch(); a++) for(int b = 0; b < input.depth(); b++) {
            kittens::warp::tma::expect_bytes(smem_semaphore, kittens::size_bytes<kittens::st<T, 16*H, 16*W>> * 2 * 2);
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                kittens::warp::tma::load_async(shared_tile[i][j], input, {input.batch() + a, input.depth() + b, i + 2, j + 2}, smem_semaphore);
            }
            kittens::wait(smem_semaphore, (a*input.depth()+b)%2);
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                kittens::store(output, shared_tile[i][j], {a, b, i, j});
            }
        }
    }
};

template<typename T>
struct test_store { // load normally, store with TMA
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<
        ( NW == 1 && W*H*sizeof(dtype)*256*4<=kittens::MAX_SHARED_MEMORY-1024 )  && ( ( sizeof(T) != 1 ) || W%2 == 0 ) 
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_gmem=half" :
                                                      #ifdef KITTENS_HOPPER
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "tma_load_gmem=fp8e4m3":
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "tma_load_gmem=fp8e5m2":
                                                      #endif
                                                                                         "tma_store_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]);
        kittens::st<T, 16*H, 16*W> (&shared_tile)[2][2] = al.allocate<kittens::st<T, 16*H, 16*W>, 2, 2>();
        __syncwarp();
        for(int a = 0; a < input.batch(); a++) for(int b = 0; b < input.depth(); b++) {
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                kittens::warp::tma::store_async_read_wait<3>(); // make sure next tile is ready for write
                kittens::load(shared_tile[i][j], input, {a, b, i, j});
            }
            __syncwarp(); // mem must be visible before store
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                kittens::warp::tma::store_async(output, shared_tile[i][j], {a, b, i, j});
            }
        }
    }
};

template<typename T>
struct test_store_add_reduce {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<
        ( NW == 1 && W*H*sizeof(dtype)*256*4<=kittens::MAX_SHARED_MEMORY-1024 )  && ( sizeof(T) != 1 ) // not supported for fp8 
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_add_reduce_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_add_reduce_gmem=half" :
                                                      #ifdef KITTENS_HOPPER
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "tma_load_gmem=fp8e4m3":
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "tma_load_gmem=fp8e5m2":
                                                        #endif
                                                                                         "tma_store_add_reduce_gmem=float";

    template<int H, int W, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // i_ref is reduced onto output
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = i_ref[i] + i_ref[i]; 
        }
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]);
        kittens::st<T, 16*H, 16*W> (&shared_tile)[2][2] = al.allocate<kittens::st<T, 16*H, 16*W>, 2, 2>();
        __syncwarp();
        for(int a = 0; a < input.batch(); a++) for(int b = 0; b < input.depth(); b++) {
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                kittens::warp::tma::store_async_read_wait<6>(); // make sure next tile is ready for write
                kittens::load(shared_tile[i][j], input, {a, b, i, j});
            }
            __syncwarp(); // mem must be visible before store
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                kittens::warp::tma::store_add_async(output, shared_tile[i][j], {a, b, i, j});
                kittens::warp::tma::store_add_async(output, shared_tile[i][j], {a, b, i, j});
            }
        }
    }
};

template<typename T>
struct test_store_min_reduce {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<
        ( !std::is_same_v<T, float> && NW == 1 && W*H*sizeof(dtype)*256*4<=kittens::MAX_SHARED_MEMORY-1024 )  && ( sizeof(T) != 1 ) // not supported for fp8 
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_min_reduce_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_min_reduce_gmem=half" :
                                                      #ifdef KITTENS_HOPPER
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "tma_load_gmem=fp8e4m3":
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "tma_load_gmem=fp8e5m2":
                                                        #endif
                                                                                         "tma_store_min_reduce_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // i_ref is reduced onto output
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = std::min(i_ref[i], 0.f);
        }
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st<T, 16*H, 16*W> (&shared_tile)[2][2] = al.allocate<kittens::st<T, 16*H, 16*W>, 2, 2>();
        __syncwarp();
        for(int a = 0; a < input.batch(); a++) for(int b = 0; b < input.depth(); b++) {
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                kittens::warp::tma::store_async_read_wait<6>(); // make sure next tile is ready for write
                kittens::load(shared_tile[i][j], input, {a, b, i, j});
            }
            __syncwarp(); // mem must be visible before store
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                kittens::warp::tma::store_min_async(output, shared_tile[i][j], {a, b, i, j}); // output is zero-initialized
                kittens::warp::tma::store_min_async(output, shared_tile[i][j], {a, b, i, j});
            }
        }
    }
};

template<typename T>
struct test_store_max_reduce {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<
        ( !std::is_same_v<T, float> && NW == 1 && W*H*sizeof(dtype)*256*4<=kittens::MAX_SHARED_MEMORY-1024 )  && ( sizeof(T) != 1  ) // not supported for fp8
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_max_reduce_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_max_reduce_gmem=half" :
                                                      #ifdef KITTENS_HOPPER
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "tma_load_gmem=fp8e4m3":
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "tma_load_gmem=fp8e5m2":
                                                        #endif
                                                                                         "tma_store_max_reduce_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // i_ref is reduced onto output
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = std::max(i_ref[i], 0.f);
        }
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]);
        kittens::st<T, 16*H, 16*W> (&shared_tile)[2][2] = al.allocate<kittens::st<T, 16*H, 16*W>, 2, 2>();
        __syncwarp();
        for(int a = 0; a < input.batch(); a++) for(int b = 0; b < input.depth(); b++) {
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                kittens::warp::tma::store_async_read_wait<6>(); // make sure next tile is ready for write
                kittens::load(shared_tile[i][j], input, {a, b, i, j});
            }
            __syncwarp(); // mem must be visible before store
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                kittens::warp::tma::store_max_async(output, shared_tile[i][j], {a, b, i, j}); // output is zero-initialized
                kittens::warp::tma::store_max_async(output, shared_tile[i][j], {a, b, i, j});
            }
        }
    }
};

template<typename Ker, typename T, int H, int W, int NW, kittens::ducks::gl::all GL, typename... args>
static __global__ void tma_global_wrapper_2d(const __grid_constant__ GL input, const __grid_constant__ GL output) {
    Ker::template device_func<H, W, NW, GL, args...>(input, output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct tma_wrapper_2d {
    using dtype = gmem_dtype<test>;
    static void run(test_data& results) {
        CudaCheckError();
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int B = 3, D = 5, R = 2, C = 2;
            constexpr int SIZE = H*W*256 * B*D*R*C; // B*D*R*C for additional TMA dimensions
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);

            initialize<dtype, initializers::RANDOM>(&d_i, &d_o, i_ref, o_ref); 
            
            // make descriptors
            using GL = typename kittens::gl<dtype, -1, -1, R*H*16, C*W*16, kittens::st<dtype, 16*H, 16*W>>;
            GL input(d_i, B, D, nullptr, nullptr);
            GL output(d_o, B, D, nullptr, nullptr);
            // run kernel
            cudaFuncSetAttribute(
                tma_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            tma_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(input, output);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GL, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, 2*W*16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
        CudaCheckError();
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
using tma_sweep_size_2d = loop_h<tma_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<template<typename> typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
struct tma_sweep_gmem_type_2d {
    static void run(test_data &results) {
        tma_sweep_size_2d<test<float>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        tma_sweep_size_2d<test<kittens::bf16>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        tma_sweep_size_2d<test<kittens::half>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        #ifdef KITTENS_HOPPER
        tma_sweep_size_2d<test<kittens::fp8e4m3>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        tma_sweep_size_2d<test<kittens::fp8e5m2>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        #endif
    }
};
template<template<typename> typename test, int MAX_H=8, int MAX_W=8, typename... args> using tma_sweep_gmem_type_2d_warp = tma_sweep_gmem_type_2d<test, MAX_H, MAX_W, 1, args...>;


void warp::memory::tile::tma::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/tma tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    tma_sweep_gmem_type_2d<test_load,             SIZE, SIZE>::run(results);
    tma_sweep_gmem_type_2d<test_load_oob,         SIZE, SIZE>::run(results);
    tma_sweep_gmem_type_2d<test_store,            SIZE, SIZE>::run(results);
    tma_sweep_gmem_type_2d<test_store_add_reduce, SIZE, SIZE>::run(results);
    tma_sweep_gmem_type_2d<test_store_min_reduce, SIZE, SIZE>::run(results);
    tma_sweep_gmem_type_2d<test_store_max_reduce, SIZE, SIZE>::run(results);
}

#endif