/*
    Basic check on all the wrapper functions around multimem red/ld_reduce operation
*/
#include "reduce.cuh"
#include <algorithm>

#ifdef TEST_WARP_MEMORY_UTIL_REDUCE

extern int should_write_outputs;

template<typename Ker, typename T>
static __global__ void global_wrapper(const __grid_constant__ int dev_idx, const __grid_constant__ T input, const __grid_constant__ T output) {
    Ker::device_func(dev_idx, input, output);
}

template<typename test, int NUM_DEVICES>
struct multimem_test_wrapper {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    using PGL = kittens::pgl<kittens::gl<dtype, 1, 1, 1, -1>, 8, true>;
    template<typename T, initializers initializer=initializers::RANDOM, int SEED=42>
    static void initialize(PGL **d_i, PGL **d_o, std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        using namespace kittens;

        const int input_size  = i_ref[0].size();
        const int output_size = o_ref[0].size();

        // Initialize matrices
        std::vector<T> i_t(input_size);
        dtype *d_i_raw[NUM_DEVICES];
        dtype *d_o_raw[NUM_DEVICES];

        int device_ids[NUM_DEVICES];
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;

        std::mt19937 gen(SEED); // Standard mersenne_twister_engine
        std::uniform_real_distribution<float> dis(-1.0, 1.0);
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
            for(int idx = 0; idx < input_size; idx++) {
                float f;
                if constexpr (initializer == initializers::RANDOM) {
                    f = dis(gen);
                }
                else if constexpr (initializer == initializers::ARANGE) {
                    f = float(idx);
                }
                else {
                    f = i_ref[dev_idx][idx];
                }
                if constexpr (std::is_same_v<T, bf16>) {
                    i_t[idx] = __float2bfloat16(f); // fill in for transfer to device
                    i_ref[dev_idx][idx] = __bfloat162float(i_t[idx]); // ensure lossiness of fp16 is captured on cpu
                }
                else if constexpr (std::is_same_v<T, float>) {
                    i_t[idx] = f;
                    i_ref[dev_idx][idx] = f;
                }
                else if constexpr (std::is_same_v<T, half>) {
                    i_t[idx] = __float2half(f);
                    i_ref[dev_idx][idx] = __half2float(i_t[idx]);
                }
                else {
                    assert(false && "Unsupported data type");
                }
            }

            cudaSetDevice(dev_idx);
            CUmemGenericAllocationHandle dev_handle; // no need to keep track of this in the tests
            pglCudaMalloc<true>(NUM_DEVICES, device_ids, dev_idx, &d_i_raw[dev_idx], &dev_handle, input_size * sizeof(T));
            pglCudaMalloc<true>(NUM_DEVICES, device_ids, dev_idx, &d_o_raw[dev_idx], &dev_handle, output_size * sizeof(T));
            cudaMemcpy(d_i_raw[dev_idx], i_t.data(), input_size * sizeof(T), cudaMemcpyHostToDevice);
            CudaCheckError();
        }

        *d_i = new PGL(device_ids, d_i_raw, nullptr, nullptr, nullptr, input_size);
        *d_o = new PGL(device_ids, d_o_raw, nullptr, nullptr, nullptr, output_size);
    }
    static test_result validate(PGL *d_i, PGL *d_o, const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref, std::string test_name, int cols=16, float eps=5e-2) { // default eps has to be fairly high due to lots of different types
        using namespace kittens;
        const int input_size  = i_ref[0].size();
        const int output_size = o_ref[0].size();
        // copy back
        dtype* o_t = new dtype[output_size];
        float *o = new float[output_size];

        std::cout << "test `" << test_name << "`";
        bool good = true;

        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
            cudaDeviceSynchronize();
            CudaCheckError();
            cudaMemcpy(o_t, d_o->gls[dev_idx].raw_ptr, output_size * sizeof(dtype), cudaMemcpyDeviceToHost);
            CudaCheckError();

            for(int idx = 0; idx < output_size; idx++) {
                if constexpr (std::is_same_v<dtype, bf16>) {
                    o[idx] = __bfloat162float(o_t[idx]);
                    o_ref[dev_idx][idx] = __bfloat162float(__float2bfloat16(o_ref[dev_idx][idx]));
                }
                else if constexpr (std::is_same_v<dtype, half>) {
                    o[idx] = __half2float(o_t[idx]);
                    o_ref[dev_idx][idx] = __half2float(__float2half(o_ref[dev_idx][idx]));
                }
                else if constexpr(std::is_same_v<dtype, float>) {
                    o[idx] = o_t[idx];
                    o_ref[dev_idx][idx] = o_ref[dev_idx][idx];
                }
                #ifdef KITTENS_HOPPER
                else if constexpr(std::is_same_v<dtype, fp8e4m3>) {
                    o[idx] = float(o_t[idx]);
                    o_ref[dev_idx][idx] = float(__nv_fp8_e4m3(o_ref[dev_idx][idx])); 
                }
                else if constexpr(std::is_same_v<dtype, fp8e5m2>) {
                    o[idx] = float(o_t[idx]);
                    o_ref[dev_idx][idx] = float(__nv_fp8_e5m2(o_ref[dev_idx][idx])); 
                }
                #endif
                else {
                    assert(false && "Unsupported data type");
                }
            }
            // check
            for(int i = 0; i < output_size; i++) {
                if(abs(o_ref[dev_idx][i] - o[i]) > eps) {
                    good = false;
                    break;
                }
            }
            if (!good) break;
        }
        if(good) std::cout << " -- PASSED" << std::endl;
        else std::cout << " ----- ALERT! FAILED test `" << test_name << "` -----" << std::endl;
        if(should_write_outputs && !good) {
            std::ofstream reffile("outputs/"+test_name+"_ref.txt");
            std::ofstream outfile("outputs/"+test_name+"_out.txt");
            for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
                outfile << "Device " << dev_idx << ":\n\n";
                reffile << "Device " << dev_idx << ":\n\n";
                for(int i = 0; i < output_size; i++) {
                    reffile << o_ref[dev_idx][i] << ' ';
                    outfile << o[i] << ' ';
                    if(i%cols == cols-1) {
                        reffile << '\n';
                        outfile << '\n';
                    }
                }
                reffile << "\n\n\nINPUTS:\n\n";
                for(int i = 0; i < input_size; i++) {
                    reffile << i_ref[dev_idx][i] << ' ';
                    if(i%cols == cols-1) {
                        reffile << '\n';
                    }
                }
                outfile << "\n\n\n\n";
                reffile << "\n\n\n\n";
            }
            reffile.close();
            outfile.close();
        }

        pglFree(*d_i);
        pglFree(*d_o);
        free(d_i);
        free(d_o);
        delete[] o_t, o;
        CudaCheckError();
        return good ? test_result::PASSED : test_result::FAILED;
    }
    static void run(test_data& results) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count < NUM_DEVICES) {
            std::cerr << "Not enough GPU devices to run test" << std::endl;
            return;
        }
        test_info this_result;
        this_result.label = test::test_identifier;
        if constexpr (test::valid::value) {
            // initialize
            PGL *d_i, *d_o; // we only use PGL for conveniently allocating MC objects, nothing else
            std::vector<std::vector<float>> i_ref(NUM_DEVICES, std::vector<float>(test::test_size));
            std::vector<std::vector<float>> o_ref(NUM_DEVICES, std::vector<float>(test::test_size));
            initialize<dtype>(&d_i, &d_o, i_ref, o_ref);
            // run kernel
            for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
                cudaSetDevice(dev_idx);
                cudaFuncSetAttribute(
                    global_wrapper<test, PGL>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kittens::MAX_SHARED_MEMORY
                );
                global_wrapper<test, PGL><<<1, 1>>>(dev_idx, *d_i, *d_o);
                cudaDeviceSynchronize();
                CudaCheckError();
            }
            test::host_func(i_ref, o_ref);
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label);
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
    using PGL = kittens::pgl<kittens::gl<dtype, 1, 1, 1, -1>, 8, true>;
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
                o_ref[dev_idx][i] = 0;
                for (int other_dev_idx = 0; other_dev_idx < i_ref.size(); ++other_dev_idx) {
                    if (op == kittens::ReduceOp::ADD) {
                        o_ref[dev_idx][i] += i_ref[other_dev_idx][i];
                    } else if (op == kittens::ReduceOp::MIN) {
                        o_ref[dev_idx][i] = std::min(o_ref[dev_idx][i], i_ref[other_dev_idx][i]);
                    } else if (op == kittens::ReduceOp::MAX) {
                        o_ref[dev_idx][i] = std::max(o_ref[dev_idx][i], i_ref[other_dev_idx][i]);
                    }
                }
            }
        }
    }
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
    using PGL = kittens::pgl<kittens::gl<dtype, 1, 1, 1, -1>, 8, true>;
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
                    if (op == kittens::ReduceOp::ADD) {
                        o_ref[dev_idx][i] += i_ref[other_dev_idx][i];
                    } else if (op == kittens::ReduceOp::MIN) {
                        o_ref[dev_idx][i] = std::min(o_ref[dev_idx][i], i_ref[other_dev_idx][i]);
                    } else if (op == kittens::ReduceOp::MAX) {
                        o_ref[dev_idx][i] = std::max(o_ref[dev_idx][i], i_ref[other_dev_idx][i]);
                    }
                }
            }
        }
    }
    __device__ static void device_func(const int dev_idx, const PGL &input, const PGL &output) {
        kittens::multimem_ld_reduce_op<packed_dtype, op>::apply((packed_dtype *)output[dev_idx].raw_ptr, (packed_dtype *)input.mc_vas[dev_idx]);
    }
};

template<typename T>
struct test_multimem_reduce_vec {
    using dtype = T;
    using PGL = kittens::pgl<kittens::gl<dtype, 1, 1, 1, -1>, 8, true>;
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
    __device__ static void device_func(const int dev_idx, const PGL &input, const PGL &output) {
        kittens::multimem_reduce_op<T, kittens::ReduceOp::ADD>::apply_vec(output.mc_vas[dev_idx], input[dev_idx].raw_ptr);
    }
};

template<typename T>
struct test_multimem_packed_packed_scalar {
    using dtype = T;
    using packed_dtype = std::conditional_t<std::is_same_v<T, kittens::half>, kittens::half_2,
                         std::conditional_t<std::is_same_v<T, kittens::bf16>, kittens::bf16_2,
                         std::conditional_t<std::is_same_v<T, float>, float2, void>>>;
    using PGL = kittens::pgl<kittens::gl<dtype, 1, 1, 1, -1>, 8, true>;
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
    __device__ static void device_func(const int dev_idx, const PGL &input, const PGL &output) {
        kittens::multimem_reduce_op<packed_dtype, kittens::ReduceOp::ADD>::apply((packed_dtype *)output.mc_vas[dev_idx], (packed_dtype *)input[dev_idx].raw_ptr);
    }
};

void warp::memory::util::reduce::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/util/reduce tests! -----\n" << std::endl;
    constexpr int NUM_DEVICES = 8;

    multimem_test_wrapper<test_multimem_ld_reduce_vec<float, kittens::ReduceOp::ADD>, NUM_DEVICES>::run(results); // MIN/MAX ops are NOT supported on float32
    multimem_test_wrapper<test_multimem_ld_reduce_vec<kittens::bf16, kittens::ReduceOp::ADD>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_ld_reduce_vec<kittens::bf16, kittens::ReduceOp::MIN>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_ld_reduce_vec<kittens::bf16, kittens::ReduceOp::MAX>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_ld_reduce_vec<kittens::half, kittens::ReduceOp::ADD>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_ld_reduce_vec<kittens::half, kittens::ReduceOp::MIN>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_ld_reduce_vec<kittens::half, kittens::ReduceOp::MAX>, NUM_DEVICES>::run(results);
    
    multimem_test_wrapper<test_multimem_ld_reduce_packed_scalar<float, kittens::ReduceOp::ADD>, NUM_DEVICES>::run(results); // MIN/MAX ops are NOT supported on float32
    multimem_test_wrapper<test_multimem_ld_reduce_packed_scalar<kittens::bf16, kittens::ReduceOp::ADD>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_ld_reduce_packed_scalar<kittens::bf16, kittens::ReduceOp::MIN>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_ld_reduce_packed_scalar<kittens::bf16, kittens::ReduceOp::MAX>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_ld_reduce_packed_scalar<kittens::half, kittens::ReduceOp::ADD>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_ld_reduce_packed_scalar<kittens::half, kittens::ReduceOp::MIN>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_ld_reduce_packed_scalar<kittens::half, kittens::ReduceOp::MAX>, NUM_DEVICES>::run(results);

    multimem_test_wrapper<test_multimem_reduce_vec<float>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_reduce_vec<kittens::bf16>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_reduce_vec<kittens::half>, NUM_DEVICES>::run(results);

    multimem_test_wrapper<test_multimem_packed_packed_scalar<float>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_packed_packed_scalar<kittens::bf16>, NUM_DEVICES>::run(results);
    multimem_test_wrapper<test_multimem_packed_packed_scalar<kittens::half>, NUM_DEVICES>::run(results);
}

#endif
