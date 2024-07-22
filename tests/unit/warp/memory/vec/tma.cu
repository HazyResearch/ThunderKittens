#include "tma.cuh"

#ifdef TEST_WARP_MEMORY_VEC_TMA

template<typename T>
struct test_load { // load with TMA, write out normally
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<NW == 1 && S<=64 && S%4==0>; // S%4 ensures alignment
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_load_vec_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_load_vec_gmem=half" :
                                                                                         "tma_load_vec_gmem=float";
    template<int S, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW>
    __device__ static void device_func(const dtype *input, dtype *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st<dtype, S, S>> (&shared_vec)[4] = al.allocate<kittens::row_vec<kittens::st<dtype, S, S>>, 4>();
        
        __shared__ kittens::barrier smem_barrier; 
        kittens::init_barrier(smem_barrier, 0, 1);
        kittens::tma::expect<typeof(shared_vec[0]), 4>(smem_barrier);
        for(int i = 0; i < 4; i++) {
            kittens::tma::load_async(shared_vec[i], tma_desc_input, smem_barrier, i);
        }
        kittens::wait(smem_barrier, 0);
        kittens::store(output, shared_vec[0]);
        kittens::store(output + shared_vec[0].length, shared_vec[1]);
        kittens::store(output + 2*shared_vec[0].length, shared_vec[2]);
        kittens::store(output + 3*shared_vec[0].length, shared_vec[3]);
    }
};
template<typename T>
struct test_store { // load normally, store with TMA
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<NW == 1 && S<=64 && S%4==0>; // S%4 ensures alignment
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_vec_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_vec_gmem=half" :
                                                                                         "tma_store_vec_gmem=float";
    template<int S, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW>
    __device__ static void device_func(const dtype *input, dtype *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st<dtype, S, S>> (&shared_vec)[4] = al.allocate<kittens::row_vec<kittens::st<dtype, S, S>>, 4>();
        
        kittens::load(shared_vec[0], input);
        kittens::load(shared_vec[1], input + shared_vec[0].length);
        kittens::load(shared_vec[2], input + 2*shared_vec[0].length);
        kittens::load(shared_vec[3], input + 3*shared_vec[0].length);
        __syncwarp();
        for(int i = 0; i < 4; i++) {
            kittens::tma::store_async(tma_desc_output, shared_vec[i], i);
        }
        kittens::tma::store_commit_group();
        kittens::tma::store_async_wait<0>();
    }
};


template<typename T>
struct test_store_add_reduce {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<NW == 1 && S<=64 && S%4==0>; // S%4 ensures alignment
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_add_reduce_vec_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_add_reduce_vec_gmem=half" :
                                                                                         "tma_store_add_reduce_vec_gmem=float";
    template<int S, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // i_ref is reduced onto output
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = i_ref[i] + i_ref[i]; 
        }
    }
    template<int S, int NW>
    __device__ static void device_func(const dtype *input, dtype *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st<dtype, S, S>> (&shared_vec)[4] = al.allocate<kittens::row_vec<kittens::st<dtype, S, S>>, 4>();
        
        kittens::load(shared_vec[0], input);
        kittens::load(shared_vec[1], input + shared_vec[0].length);
        kittens::load(shared_vec[2], input + 2*shared_vec[0].length);
        kittens::load(shared_vec[3], input + 3*shared_vec[0].length);
        __syncwarp();
        for(int i = 0; i < 4; i++) {
            kittens::tma::store_add_async(tma_desc_output, shared_vec[i], i);
        }
        kittens::tma::store_commit_group();
        for(int i = 0; i < 4; i++) {
            kittens::tma::store_add_async(tma_desc_output, shared_vec[i], i);
        }
        kittens::tma::store_commit_group();
        kittens::tma::store_async_wait<0>();
    }
};


template<typename T>
struct test_store_min_reduce {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<!std::is_same_v<T, float> && NW == 1 && S<=64 && S%4==0>; // S%4 ensures alignment
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_min_reduce_vec_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_min_reduce_vec_gmem=half" :
                                                                                         "tma_store_min_reduce_vec_gmem=float";
    template<int S, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // i_ref is reduced onto output
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = std::min(i_ref[i], i_ref[i]);
        }
    }
    template<int S, int NW>
    __device__ static void device_func(const dtype *input, dtype *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st<dtype, S, S>> (&shared_vec)[4] = al.allocate<kittens::row_vec<kittens::st<dtype, S, S>>, 4>();
        
        kittens::load(shared_vec[0], input);
        kittens::load(shared_vec[1], input + shared_vec[0].length);
        kittens::load(shared_vec[2], input + 2*shared_vec[0].length);
        kittens::load(shared_vec[3], input + 3*shared_vec[0].length);
        __syncwarp();
        for(int i = 0; i < 4; i++) {
            kittens::tma::store_add_async(tma_desc_output, shared_vec[i], i);
        }
        kittens::tma::store_commit_group();
        kittens::tma::store_async_wait<0>();
    }
};

template<typename T>
struct test_store_max_reduce {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<!std::is_same_v<T, float> && NW == 1 && S<=64 && S%4==0>; // S%4 ensures alignment
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_max_reduce_vec_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_max_reduce_vec_gmem=half" :
                                                                                         "tma_store_max_reduce_vec_gmem=float";
    template<int S, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // i_ref is reduced onto output
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = std::max(i_ref[i], i_ref[i]);
        }
    }
    template<int S, int NW>
    __device__ static void device_func(const dtype *input, dtype *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st<dtype, S, S>> (&shared_vec)[4] = al.allocate<kittens::row_vec<kittens::st<dtype, S, S>>, 4>();
        
        kittens::load(shared_vec[0], input);
        kittens::load(shared_vec[1], input + shared_vec[0].length);
        kittens::load(shared_vec[2], input + 2*shared_vec[0].length);
        kittens::load(shared_vec[3], input + 3*shared_vec[0].length);
        __syncwarp();
        for(int i = 0; i < 4; i++) {
            kittens::tma::store_add_async(tma_desc_output, shared_vec[i], i);
        }
        kittens::tma::store_commit_group();
        kittens::tma::store_async_wait<0>();
    }
};

template<typename Ker, typename T, int S, int NW, typename... args>
static __global__ void tma_global_wrapper_1d(const T *input, T *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
    Ker::template device_func<S, NW, args...>(input, output, tma_desc_input, tma_desc_output);
}
template<typename test, int S, int NUM_WORKERS, typename... args>
struct tma_wrapper_1d {
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
            // initialize TMA descriptors
            CUtensorMap *i_desc = kittens::tma::allocate_and_create_tensor_map<kittens::row_vec<kittens::st<dtype, S, S>>>(d_i, 4);
            CUtensorMap *o_desc = kittens::tma::allocate_and_create_tensor_map<kittens::row_vec<kittens::st<dtype, S, S>>>(d_o, 4);
            // run kernel
            cudaFuncSetAttribute(
                tma_global_wrapper_1d<test, dtype, S, NUM_WORKERS, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            tma_global_wrapper_1d<test, dtype, S, NUM_WORKERS, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(d_i, d_o, i_desc, o_desc);
            // fill in correct results on cpu
            test::template host_func<S, NUM_WORKERS, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, S*16);
            cudaFree(i_desc);
            cudaFree(o_desc);
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