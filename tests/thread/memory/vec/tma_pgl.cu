#include "tma_pgl.cuh"

#ifdef TEST_THREAD_MEMORY_VEC_TMA_PGL

template<typename Ker, int S, int NW, kittens::ducks::pgl::all PGL, typename... args>
static __global__ void tma_pgl_vec_global_wrapper_1d(const __grid_constant__ PGL input, const __grid_constant__ PGL output, const __grid_constant__ int dev_idx) {
    Ker::template device_func<S, NW, PGL, args...>(input, output, dev_idx);
}

constexpr static int B = 3, D = 5; // Arbitrary, can change
constexpr static int R = 2, C = 2; // Fixed, must change the test code to change these

template<typename test, int NUM_DEVICES, int S, int NUM_WORKERS, typename... args>
struct tma_pgl_vec_test_wrapper_1d {
    constexpr static int SIZE = S*16 * B*D*R*C;

    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    using PGL = kittens::pgl<kittens::gl<dtype, -1, -1, -1, -1>, NUM_DEVICES, true, kittens::sv<dtype, 16*S>>;

    static void run(test_data& results) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        CudaCheckError();
        if (device_count < NUM_DEVICES) {
            std::cerr << "Warning: Not enough GPU devices to run test " << test::test_identifier << std::endl;
            return;
        }
        test_info this_result;
        this_result.label = generate_test_name<S, NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<S, NUM_WORKERS>::value) {

            // initialize
            dtype *d_i_arr[NUM_DEVICES];
            dtype *d_o_arr[NUM_DEVICES];
            dtype *d_i_mc;
            dtype *d_o_mc;
            size_t d_i_alloc_size, d_o_alloc_size, d_i_mc_alloc_size, d_o_mc_alloc_size;
            std::vector<std::vector<float>> i_ref(NUM_DEVICES, std::vector<float>(SIZE));
            std::vector<std::vector<float>> o_ref(NUM_DEVICES, std::vector<float>(SIZE));
            initialize<NUM_DEVICES>(
                d_i_arr, d_o_arr, &d_i_mc, &d_o_mc,
                &d_i_alloc_size, &d_o_alloc_size, &d_i_mc_alloc_size, &d_o_mc_alloc_size,
                i_ref, o_ref
            );

            // Prepare PGLs
            PGL input {d_i_mc, d_i_arr, B, D, R, 16*S*C};
            PGL output {d_o_mc, d_o_arr, B, D, R, 16*S*C};

            // run kernel
            for (int dev_idx = 0; dev_idx < (test::single_run ? 1 : NUM_DEVICES); ++dev_idx) {
                cudaSetDevice(dev_idx);
                cudaFuncSetAttribute(
                    tma_pgl_vec_global_wrapper_1d<test, S, NUM_WORKERS, PGL, args...>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kittens::MAX_SHARED_MEMORY-1024
                );
                tma_pgl_vec_global_wrapper_1d<test, S, NUM_WORKERS, PGL, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY-1024>>>(input, output, dev_idx);
            }

            // fill in correct results on cpu
            test::template host_func(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate<NUM_DEVICES, PGL, dtype>(
                input, output,
                d_i_alloc_size, d_o_alloc_size, d_i_mc_alloc_size, d_o_mc_alloc_size,
                i_ref, o_ref, this_result.label, S * 16
            );
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
        CudaCheckError();
    }
};

template<typename T>
struct tma_pgl_vec_store_async_test {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant
        <NW == 1 && (S*16 <= 256 || (S*16*sizeof(dtype)) % 128 == 0)>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_pgl_vec_store_async=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_pgl_vec_store_async=half" :
                                                                                         "tma_pgl_vec_store_async=float";
    static inline const bool single_run = true; // run on device 0 vs all devices
    __host__ static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                o_ref[dev_idx][i] += i_ref[0][i];
            }
        }
    }
    template<int S, int NW, kittens::ducks::pgl::all PGL>
    __device__ static void device_func(const PGL &input, const PGL &output, const int dev_idx) {
        extern __shared__ kittens::alignment_dummy __shm[]; // smem
        kittens::tma_allocator al((int*)&__shm[0]);
        kittens::row_vec<kittens::st<dtype, 16*S, 16*S>> (&shared_vec) = al.allocate<kittens::row_vec<kittens::st<dtype, 16*S, 16*S>>>();
        __syncwarp();
        for(int a = 0; a < B; a++) {
            for(int b = 0; b < D; b++) {
                for(int i = 0; i < R; i++) {
                    for(int j = 0; j < C; j++) {
                        kittens::warp::load(shared_vec, input[dev_idx], {a, b, i, j});
                        __syncwarp(); // mem must be visible before store
                        if (threadIdx.x == 0)
                            kittens::tma::store_async(output, shared_vec, {a, b, i, j});
                        kittens::tma::store_async_read_wait(); 
                    }
                }
            }
        }
    }
};

template<typename T>
struct tma_pgl_vec_store_add_async_test {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant
        <NW == 1 && (S*16 <= 256 || (S*16*sizeof(dtype)) % 128 == 0)>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_pgl_vec_store_add_async=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_pgl_vec_store_add_async=half" :
                                                                                         "tma_pgl_vec_store_add_async=float";
    static inline const bool single_run = false; // run on device 0 vs all devices
    __host__ static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                o_ref[dev_idx][i] = 0;
            }
        }
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int other_dev_idx = 0; other_dev_idx < i_ref.size(); ++other_dev_idx) {
                for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                    o_ref[other_dev_idx][i] += i_ref[dev_idx][i];
                }
            }
        }
    }
    template<int S, int NW, kittens::ducks::pgl::all PGL>
    __device__ static void device_func(const PGL &input, const PGL &output, const int dev_idx) {
        extern __shared__ kittens::alignment_dummy __shm[]; // smem
        kittens::tma_allocator al((int*)&__shm[0]);
        kittens::row_vec<kittens::st<dtype, 16*S, 16*S>> (&shared_vec) = al.allocate<kittens::row_vec<kittens::st<dtype, 16*S, 16*S>>>();
        __syncwarp();
        for(int a = 0; a < B; a++) {
            for(int b = 0; b < D; b++) {
                for(int i = 0; i < R; i++) {
                    for(int j = 0; j < C; j++) {
                        kittens::warp::load(shared_vec, input[dev_idx], {a, b, i, j});
                        __syncwarp(); // mem must be visible before store
                        if (threadIdx.x == 0)
                            kittens::tma::store_add_async(output, shared_vec, {a, b, i, j});
                        kittens::tma::store_async_read_wait(); 
                    }
                }
            }
        }
    }
};

template<typename T>
struct tma_pgl_vec_store_min_async_test {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant
        <NW == 1 && (S*16 <= 256 || (S*16*sizeof(dtype)) % 128 == 0)>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_pgl_vec_store_min_async=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_pgl_vec_store_min_async=half" :
                                                                                         "tma_pgl_vec_store_min_async=float";
    static inline const bool single_run = false; // run on device 0 vs all devices
    __host__ static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                for (int other_dev_idx = 0; other_dev_idx < i_ref.size(); ++other_dev_idx) {
                    o_ref[dev_idx][i] = std::min(i_ref[other_dev_idx][i], o_ref[dev_idx][i]); // should work as tma reduction is atomic
                }
            }
        }
    }
    template<int S, int NW, kittens::ducks::pgl::all PGL>
    __device__ static void device_func(const PGL &input, const PGL &output, const int dev_idx) {
        extern __shared__ kittens::alignment_dummy __shm[]; // smem
        kittens::tma_allocator al((int*)&__shm[0]);
        kittens::row_vec<kittens::st<dtype, 16*S, 16*S>> (&shared_vec) = al.allocate<kittens::row_vec<kittens::st<dtype, 16*S, 16*S>>>();
        __syncwarp();
        for(int a = 0; a < B; a++) {
            for(int b = 0; b < D; b++) {
                for(int i = 0; i < R; i++) {
                    for(int j = 0; j < C; j++) {
                        kittens::warp::load(shared_vec, input[dev_idx], {a, b, i, j});
                        __syncwarp(); // mem must be visible before store
                        if (threadIdx.x == 0)
                            kittens::tma::store_min_async(output, shared_vec, {a, b, i, j});
                        kittens::tma::store_async_read_wait(); 
                    }
                }
            }
        }
    }
};

template<typename T>
struct tma_pgl_vec_store_max_async_test {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant
        <NW == 1 && (S*16 <= 256 || (S*16*sizeof(dtype)) % 128 == 0)>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_pgl_vec_store_max_async=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_pgl_vec_store_max_async=half" :
                                                                                         "tma_pgl_vec_store_max_async=float";
    static inline const bool single_run = false; // run on device 0 vs all devices
    __host__ static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                for (int other_dev_idx = 0; other_dev_idx < i_ref.size(); ++other_dev_idx) {
                    o_ref[dev_idx][i] = std::max(i_ref[other_dev_idx][i], o_ref[dev_idx][i]); // should work as tma reduction is atomic
                }
            }
        }
    }
    template<int S, int NW, kittens::ducks::pgl::all PGL>
    __device__ static void device_func(const PGL &input, const PGL &output, const int dev_idx) {
        extern __shared__ kittens::alignment_dummy __shm[]; // smem
        kittens::tma_allocator al((int*)&__shm[0]);
        kittens::row_vec<kittens::st<dtype, 16*S, 16*S>> (&shared_vec) = al.allocate<kittens::row_vec<kittens::st<dtype, 16*S, 16*S>>>();
        __syncwarp();
        for(int a = 0; a < B; a++) {
            for(int b = 0; b < D; b++) {
                for(int i = 0; i < R; i++) {
                    for(int j = 0; j < C; j++) {
                        kittens::warp::load(shared_vec, input[dev_idx], {a, b, i, j});
                        __syncwarp(); // mem must be visible before store
                        if (threadIdx.x == 0)
                            kittens::tma::store_max_async(output, shared_vec, {a, b, i, j});
                        kittens::tma::store_async_read_wait(); 
                    }
                }
            }
        }
    }
};

template<typename test, int NUM_DEVICES, int MAX_S, int NUM_WORKERS, typename... args> 
using tma_pgl_vec_sweep_size_1d = mg_loop_s<tma_pgl_vec_test_wrapper_1d, test, NUM_DEVICES, MAX_S, NUM_WORKERS, MAX_S, args...>;

template<typename test, int NUM_DEVICES, int MAX_S, typename... args> 
using tma_pgl_vec_sweep_size_1d_warp = tma_pgl_vec_sweep_size_1d<test, NUM_DEVICES, MAX_S, 1, args...>;

template<typename test, int NUM_DEVICES, int MAX_S, typename... args>
struct tma_pgl_vec_sweep_size_1d_warp_axes {
    using I0_t = std::integral_constant<int, 0>;
    using I1_t = std::integral_constant<int, 1>;
    using I2_t = std::integral_constant<int, 2>;

    static void run(test_data &results) {
        tma_pgl_vec_sweep_size_1d_warp<test, NUM_DEVICES, MAX_S>::run(results);

        // Only axis = 2 is supported for now
        // tma_pgl_vec_sweep_size_1d_warp<test, NUM_DEVICES, MAX_S, I2_t>::run(results);
        // tma_pgl_vec_sweep_size_1d_warp<test, NUM_DEVICES, MAX_S, I1_t>::run(results);
        // tma_pgl_vec_sweep_size_1d_warp<test, NUM_DEVICES, MAX_S, I0_t>::run(results);
    }
};

template<typename T, int NUM_DEVICES, int MAX_S, typename... args>
struct tma_pgl_vec_sweep_size_1d_warp_axes_ops {
    static void run(test_data &results) {
        tma_pgl_vec_sweep_size_1d_warp_axes<tma_pgl_vec_store_async_test<T>, NUM_DEVICES, MAX_S>::run(results);
        tma_pgl_vec_sweep_size_1d_warp_axes<tma_pgl_vec_store_add_async_test<T>, NUM_DEVICES, MAX_S>::run(results);
        if constexpr (!std::is_same<T, float>::value) {
            tma_pgl_vec_sweep_size_1d_warp_axes<tma_pgl_vec_store_min_async_test<T>, NUM_DEVICES, MAX_S>::run(results);
            tma_pgl_vec_sweep_size_1d_warp_axes<tma_pgl_vec_store_max_async_test<T>, NUM_DEVICES, MAX_S>::run(results);
        }
    }
};

void thread::memory::vec::tma_pgl::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/tma_pgl tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 8  : 
                         INTENSITY_3 ? 32  :
                         INTENSITY_4 ? 64 : -1;

    if (check_multi_gpus()) {
        tma_pgl_vec_sweep_size_1d_warp_axes_ops<float, NUM_GPUS, SIZE>::run(results);
        tma_pgl_vec_sweep_size_1d_warp_axes_ops<kittens::bf16, NUM_GPUS, SIZE>::run(results);
        tma_pgl_vec_sweep_size_1d_warp_axes_ops<kittens::half, NUM_GPUS, SIZE>::run(results);
    }
}

#endif
