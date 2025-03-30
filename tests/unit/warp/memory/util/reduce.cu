/*
    Basic check on all the wrapper functions around multimem red/ld_reduce operation
*/
#include "reduce.cuh"
#include <algorithm>

#ifdef TEST_WARP_MEMORY_UTIL_REDUCE

template<typename Ker, typename T>
static __global__ void global_wrapper(const __grid_constant__ int dev_idx, const __grid_constant__ T input, const __grid_constant__ T output) {
    Ker::device_func(dev_idx, input, output);
}

template<typename test, int NUM_DEVICES>
struct multimem_test_wrapper {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    using shared_layout = shared_layouts<dtype, NUM_DEVICES>;

    static void run(test_data& results) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count < NUM_DEVICES) {
            std::cerr << "Warning: Not enough GPU devices to run test " << test::test_identifier << std::endl;
            return;
        }
        test_info this_result;
        this_result.label = test::test_identifier;
        if constexpr (test::valid::value) {

            // initialize
            dtype *d_i_arr[NUM_DEVICES];
            dtype *d_o_arr[NUM_DEVICES];
            std::vector<std::vector<float>> i_ref(NUM_DEVICES, std::vector<float>(test::test_size));
            std::vector<std::vector<float>> o_ref(NUM_DEVICES, std::vector<float>(test::test_size));
            int device_ids[NUM_DEVICES];
            for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;
            initialize<NUM_DEVICES>(device_ids, d_i_arr, d_o_arr, i_ref, o_ref);

            // set parralel global layouts
            shared_pgl_replace_gl<dtype, NUM_DEVICES>(d_i_arr, d_o_arr, 1, 1, 1, test::test_size);
            shared_layout::input_pgl->multicast_bind();
            shared_layout::output_pgl->multicast_bind();

            // run kernel
            for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
                cudaSetDevice(dev_idx);
                cudaFuncSetAttribute(
                    global_wrapper<test, typename shared_layout::PGL>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kittens::MAX_SHARED_MEMORY
                );
                global_wrapper<test, typename shared_layout::PGL><<<1, 1>>>(dev_idx, *shared_layout::input_pgl, *shared_layout::output_pgl);
                cudaDeviceSynchronize();
                CudaCheckError();
            }

            // fill in correct results on cpu
            test::host_func(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate<NUM_DEVICES, typename shared_layout::PGL, dtype>(*shared_layout::input_pgl, *shared_layout::output_pgl, i_ref, o_ref, this_result.label);

            shared_layout::input_pgl->multicast_unbind();
            shared_layout::output_pgl->multicast_unbind();
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename T, kittens::ReduceOp op>
struct test_multimem_ld_reduce_vec {
    using dtype = T;
    using valid = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    static inline const std::string op_identifier = op == kittens::ReduceOp::ADD ? "ADD" : op == kittens::ReduceOp::MIN ? "MIN" : "MAX";
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "multimem_ld_reduce_vec=bf16,op=" + op_identifier :
                                                      std::is_same_v<T, kittens::half> ? "multimem_ld_reduce_vec=half,op=" + op_identifier :
                                                                                         "multimem_ld_reduce_vec=float,op=" + op_identifier;
    static inline const int test_size = 16 / sizeof(T);
    __host__ static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                o_ref[dev_idx][i] = i_ref[0][i];
                for (int other_dev_idx = 1; other_dev_idx < i_ref.size(); ++other_dev_idx) {
                    if constexpr (op == kittens::ReduceOp::ADD) {
                        o_ref[dev_idx][i] += i_ref[other_dev_idx][i];
                    } else if constexpr (op == kittens::ReduceOp::MIN) {
                        o_ref[dev_idx][i] = std::min(o_ref[dev_idx][i], i_ref[other_dev_idx][i]);
                    } else if constexpr (op == kittens::ReduceOp::MAX) {
                        o_ref[dev_idx][i] = std::max(o_ref[dev_idx][i], i_ref[other_dev_idx][i]);
                    }
                }
            }
        }
    }
    template <kittens::ducks::pgl::all PGL>
    __device__ static void device_func(const int dev_idx, const PGL &input, const PGL &output) {
        kittens::multimem_ld_reduce_op<T, op>::apply_vec((float4 *)output[dev_idx].raw_ptr, input.mc_vas[dev_idx]);
    }
};

template<typename T, kittens::ReduceOp op>
struct test_multimem_ld_reduce_packed_scalar {
    using dtype = T;
    using packed_dtype = std::conditional_t<std::is_same_v<T, kittens::half>, kittens::half_2,
                         std::conditional_t<std::is_same_v<T, kittens::bf16>, kittens::bf16_2,
                         std::conditional_t<std::is_same_v<T, float>, float2, void>>>;
    using valid = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    static inline const std::string op_identifier = op == kittens::ReduceOp::ADD ? "ADD" : op == kittens::ReduceOp::MIN ? "MIN" : "MAX";
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "multimem_ld_reduce_packed_scalar=bf16,op=" + op_identifier :
                                                      std::is_same_v<T, kittens::half> ? "multimem_ld_reduce_packed_scalar=half,op=" + op_identifier :
                                                                                         "multimem_ld_reduce_packed_scalar=float,op=" + op_identifier;
    static inline const int test_size = 2; // currently tk only supports x2 packed types (very easy to extend in the future though)
    __host__ static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                o_ref[dev_idx][i] = 0;
                for (int other_dev_idx = 0; other_dev_idx < i_ref.size(); ++other_dev_idx) {
                    if constexpr (op == kittens::ReduceOp::ADD) {
                        o_ref[dev_idx][i] += i_ref[other_dev_idx][i];
                    } else if constexpr (op == kittens::ReduceOp::MIN) {
                        o_ref[dev_idx][i] = std::min(o_ref[dev_idx][i], i_ref[other_dev_idx][i]);
                    } else if constexpr (op == kittens::ReduceOp::MAX) {
                        o_ref[dev_idx][i] = std::max(o_ref[dev_idx][i], i_ref[other_dev_idx][i]);
                    }
                }
            }
        }
    }
    template <kittens::ducks::pgl::all PGL>
    __device__ static void device_func(const int dev_idx, const PGL &input, const PGL &output) {
        kittens::multimem_ld_reduce_op<packed_dtype, op>::apply((packed_dtype *)output[dev_idx].raw_ptr, (packed_dtype *)input.mc_vas[dev_idx]);
    }
};

template<typename T>
struct test_multimem_reduce_vec {
    using dtype = T;
    using valid = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "multimem_reduce_vec=bf16,op=ADD" :
                                                      std::is_same_v<T, kittens::half> ? "multimem_reduce_vec=half,op=ADD" :
                                                                                         "multimem_reduce_vec=float,op=ADD";
    static inline const int test_size = 16 / sizeof(T);
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
    template <kittens::ducks::pgl::all PGL>
    __device__ static void device_func(const int dev_idx, const PGL &input, const PGL &output) {
        kittens::multimem_reduce_op<T, kittens::ReduceOp::ADD>::apply_vec(output.mc_vas[dev_idx], input[dev_idx].raw_ptr);
    }
};

template<typename T>
struct test_multimem_reduce_packed_scalar {
    using dtype = T;
    using packed_dtype = std::conditional_t<std::is_same_v<T, kittens::half>, kittens::half_2,
                         std::conditional_t<std::is_same_v<T, kittens::bf16>, kittens::bf16_2,
                         std::conditional_t<std::is_same_v<T, float>, float2, void>>>;
    using valid = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "multimem_reduce_vec=bf16,op=ADD" :
                                                      std::is_same_v<T, kittens::half> ? "multimem_reduce_vec=half,op=ADD" :
                                                                                         "multimem_reduce_vec=float,op=ADD";
    static inline const int test_size = 2; // currently tk only supports x2 packed types (very easy to extend in the future though)
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
    template <kittens::ducks::pgl::all PGL>
    __device__ static void device_func(const int dev_idx, const PGL &input, const PGL &output) {
        kittens::multimem_reduce_op<packed_dtype, kittens::ReduceOp::ADD>::apply((packed_dtype *)output.mc_vas[dev_idx], (packed_dtype *)input[dev_idx].raw_ptr);
    }
};

template<typename T, int NUM_DEVICES>
struct multimem_sweep_op {
    static void run(test_data &results) {
        using shared_layout = shared_layouts<T, NUM_DEVICES>;

        // Initialize shared PGLs
        int device_ids[NUM_DEVICES];
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;
        T *d_arr[NUM_DEVICES] = {}; // nullptrs (create dummy GLs)
        shared_layout::input_pgl = new shared_layout::PGL(device_ids, d_arr, 1, 1, 1, 16);
        shared_layout::output_pgl = new shared_layout::PGL(device_ids, d_arr, 1, 1, 1, 16);
        shared_layout::input_pgl->multicast_init(); // can't to bind, but init is fine with just the dimensions
        shared_layout::output_pgl->multicast_init();

        multimem_test_wrapper<test_multimem_ld_reduce_vec<T, kittens::ReduceOp::ADD>, NUM_DEVICES>::run(results);
        if constexpr (!std::is_same<T, float>::value) { // float32 is not supported for MIN/MAX ops
            multimem_test_wrapper<test_multimem_ld_reduce_vec<T, kittens::ReduceOp::MIN>, NUM_DEVICES>::run(results);
            multimem_test_wrapper<test_multimem_ld_reduce_vec<T, kittens::ReduceOp::MAX>, NUM_DEVICES>::run(results);
        }
    
        multimem_test_wrapper<test_multimem_ld_reduce_packed_scalar<T, kittens::ReduceOp::ADD>, NUM_DEVICES>::run(results);
        if constexpr (!std::is_same<T, float>::value) { // float32 is not supported for MIN/MAX ops
            multimem_test_wrapper<test_multimem_ld_reduce_packed_scalar<T, kittens::ReduceOp::MIN>, NUM_DEVICES>::run(results);
            multimem_test_wrapper<test_multimem_ld_reduce_packed_scalar<T, kittens::ReduceOp::MAX>, NUM_DEVICES>::run(results);
        }
        
        multimem_test_wrapper<test_multimem_reduce_vec<T>, NUM_DEVICES>::run(results);
        multimem_test_wrapper<test_multimem_reduce_packed_scalar<T>, NUM_DEVICES>::run(results);

        // Delete shared PGLs
        shared_layout::input_pgl->multicast_destroy();
        shared_layout::output_pgl->multicast_destroy();
        delete shared_layout::input_pgl;
        delete shared_layout::output_pgl;
    }
};

void warp::memory::util::reduce::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/util/reduce tests! -----\n" << std::endl;

    if (check_multi_gpus()) {
        multimem_sweep_op<float, NUM_GPUS>::run(results);
        multimem_sweep_op<kittens::bf16, NUM_GPUS>::run(results);
        multimem_sweep_op<kittens::half, NUM_GPUS>::run(results);
    }
}

#endif
