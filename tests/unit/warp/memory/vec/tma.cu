#include "tma.cuh"

#ifdef TEST_WARP_MEMORY_VEC_TMA

template<typename T>
struct test_load { // load with TMA, write out normally
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<NW == 1>; // WARNING: can have misalignment if multiple shared vectors are initialized in an array
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_load_vec_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_load_vec_gmem=half" :
                                                                                         "tma_load_vec_gmem=float";
    template<int S, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st<dtype, 16*S, 16*S>> (&shared_vec) = al.allocate<kittens::row_vec<kittens::st<dtype, 16*S, 16*S>>>();
        
        __shared__ kittens::barrier smem_barrier; 
        kittens::init_barrier(smem_barrier, 0, 1);
        __syncwarp();
        int tic = 0;
        for(int a = 0; a < input.batch; a++) for(int b = 0; b < input.depth; b++) {
            for(int c = 0; c < input.rows; c++) for(int d = 0; d < input.cols/shared_vec.length; d++) {
                kittens::tma::expect(smem_barrier, shared_vec);
                kittens::tma::load_async(shared_vec, input, {a, b, c, d}, smem_barrier);
                kittens::wait(smem_barrier, tic);
                kittens::store(output, shared_vec, {a, b, c, d});
                tic^=1;
            }
        }
    }
};
template<typename T>
struct test_store { // load normally, store with TMA
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<NW == 1>; // WARNING: can have misalignment if multiple shared vectors are initialized in an array
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_vec_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_vec_gmem=half" :
                                                                                         "tma_store_vec_gmem=float";
    template<int S, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st<dtype, 16*S, 16*S>> (&shared_vec) = al.allocate<kittens::row_vec<kittens::st<dtype, 16*S, 16*S>>>();
        for(int a = 0; a < input.batch; a++) for(int b = 0; b < input.depth; b++) {
            for(int c = 0; c < input.rows; c++) for(int d = 0; d < input.cols/shared_vec.length; d++) {
                kittens::load(shared_vec, input, {a, b, c, d});
                __syncwarp();
                kittens::tma::store_async(output, shared_vec, {a, b, c, d});
                kittens::tma::store_async_read_wait();
            }
        }
    }
};


template<typename T>
struct test_store_add_reduce {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<NW == 1>; // S%4 ensures alignment
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_add_reduce_vec_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_add_reduce_vec_gmem=half" :
                                                                                         "tma_store_add_reduce_vec_gmem=float";
    template<int S, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // i_ref is reduced onto output
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = i_ref[i] + i_ref[i]; 
        }
    }
    template<int S, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st<dtype, 16*S, 16*S>> (&shared_vec) = al.allocate<kittens::row_vec<kittens::st<dtype, 16*S, 16*S>>>();
        for(int a = 0; a < input.batch; a++) for(int b = 0; b < input.depth; b++) {
            for(int c = 0; c < input.rows; c++) for(int d = 0; d < input.cols/shared_vec.length; d++) {
                kittens::load(shared_vec, input, {a, b, c, d});
                __syncwarp();
                kittens::tma::store_add_async(output, shared_vec, {a, b, c, d});
                kittens::tma::store_add_async(output, shared_vec, {a, b, c, d});
                kittens::tma::store_async_read_wait();
            }
        }
    }
};


template<typename T>
struct test_store_min_reduce {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<!std::is_same_v<T, float> && NW == 1>; // S%4 ensures alignment
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_min_reduce_vec_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_min_reduce_vec_gmem=half" :
                                                                                         "tma_store_min_reduce_vec_gmem=float";
    template<int S, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // i_ref is reduced onto output
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = std::min(i_ref[i], 0.f);
        }
    }
    template<int S, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st<dtype, 16*S, 16*S>> (&shared_vec) = al.allocate<kittens::row_vec<kittens::st<dtype, 16*S, 16*S>>>();
        for(int a = 0; a < input.batch; a++) for(int b = 0; b < input.depth; b++) {
            for(int c = 0; c < input.rows; c++) for(int d = 0; d < input.cols/shared_vec.length; d++) {
                kittens::load(shared_vec, input, {a, b, c, d});
                __syncwarp();
                kittens::tma::store_min_async(output, shared_vec, {a, b, c, d});
                kittens::tma::store_async_read_wait();
            }
        }
    }
};

template<typename T>
struct test_store_max_reduce {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<!std::is_same_v<T, float> && NW == 1>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_max_reduce_vec_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_max_reduce_vec_gmem=half" :
                                                                                         "tma_store_max_reduce_vec_gmem=float";
    template<int S, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // i_ref is reduced onto output
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = std::max(i_ref[i], 0.f);
        }
    }
    template<int S, int NW, kittens::ducks::gl::all GL>
    __device__ static void device_func(const GL &input, GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st<dtype, 16*S, 16*S>> (&shared_vec) = al.allocate<kittens::row_vec<kittens::st<dtype, 16*S, 16*S>>>();
        for(int a = 0; a < input.batch; a++) for(int b = 0; b < input.depth; b++) {
            for(int c = 0; c < input.rows; c++) for(int d = 0; d < input.cols/shared_vec.length; d++) {
                kittens::load(shared_vec, input, {a, b, c, d});
                __syncwarp();
                kittens::tma::store_max_async(output, shared_vec, {a, b, c, d});
                kittens::tma::store_async_read_wait();
            }
        }
    }
};

template<typename Ker, typename T, int S, int NW, kittens::ducks::gl::all GL, typename... args>
static __global__ void tma_global_wrapper_1d(GL input, GL output) {
    Ker::template device_func<S, NW, GL, args...>(input, output);
}
template<typename test, int S, int NUM_WORKERS, typename... args>
struct tma_wrapper_1d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<S,NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<S, NUM_WORKERS, args...>::value) {
            constexpr int B = 3, D = 5, R = 2, C = 2;
            constexpr int SIZE = S*16 * B*D*R*C; // B*D*R*C for additional TMA dimensions
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, -1, -1, R, C*S*16, kittens::sv<dtype, 16*S>>;
            GL input(d_i, B, D, nullptr, nullptr);
            GL output(d_o, B, D, nullptr, nullptr);
            // run kernel
            cudaFuncSetAttribute(
                tma_global_wrapper_1d<test, dtype, S, NUM_WORKERS, GL, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            tma_global_wrapper_1d<test, dtype, S, NUM_WORKERS, GL, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(input, output);
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
using tma_sweep_size_1d = loop_s<tma_wrapper_1d, test, MAX_S, NW, MAX_S, args...>;

template<template<typename> typename test, int MAX_S=8, int NUM_WORKERS=1, typename... args>
struct tma_sweep_gmem_type_1d {
    static void run(test_data &results) {
        tma_sweep_size_1d<test<float>, MAX_S, NUM_WORKERS, args...>::run(results);
        tma_sweep_size_1d<test<kittens::bf16>, MAX_S, NUM_WORKERS, args...>::run(results);
        tma_sweep_size_1d<test<kittens::half>, MAX_S, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename> typename test, int MAX_S=8, typename... args> using tma_sweep_gmem_type_1d_warp = tma_sweep_gmem_type_1d<test, MAX_S, 1, args...>;

void warp::memory::vec::tma::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/tma tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    tma_sweep_gmem_type_1d_warp<test_load,             SIZE>::run(results);
    tma_sweep_gmem_type_1d_warp<test_store,            SIZE>::run(results);
    tma_sweep_gmem_type_1d_warp<test_store_add_reduce, SIZE>::run(results);
    tma_sweep_gmem_type_1d_warp<test_store_min_reduce, SIZE>::run(results);
    tma_sweep_gmem_type_1d_warp<test_store_max_reduce, SIZE>::run(results);
}

#endif