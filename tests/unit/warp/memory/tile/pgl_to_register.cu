#include "pgl_to_register.cuh"

#ifdef TEST_WARP_MEMORY_TILE_PGL_TO_REGISTER

extern int should_write_outputs;

template<typename Ker, int H, int W, int NW, kittens::ducks::pgl::all PGL>
static __global__ void p2r_global_wrapper_2d(const __grid_constant__ PGL input, const __grid_constant__ PGL output, const __grid_constant__ int dev_idx) {
    Ker::template device_func<H, W, NW, PGL>(input, output, dev_idx);
}

template<typename test, int NUM_DEVICES, int H, int W, int NUM_WORKERS, typename... args>
struct p2r_test_wrapper_2d {
    constexpr static int B = 3, D = 1, R = 4, C = 5; // arbitrary mix of prime/composite numbers
    constexpr static int SIZE = H*W*256 * B * D * R * C;

    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    using PGL = kittens::pgl<kittens::gl<dtype, -1, D, -1, 16*C*W>, NUM_DEVICES, true>;

    template<typename T, initializers initializer=initializers::RANDOM, int SEED=42>
    static void initialize(PGL **d_i, PGL **d_o, std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {

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
                if constexpr (std::is_same_v<T, kittens::bf16>) {
                    i_t[idx] = __float2bfloat16(f); // fill in for transfer to device
                    i_ref[dev_idx][idx] = __bfloat162float(i_t[idx]); // ensure lossiness of fp16 is captured on cpu
                }
                else if constexpr (std::is_same_v<T, float>) {
                    i_t[idx] = f;
                    i_ref[dev_idx][idx] = f;
                }
                else if constexpr (std::is_same_v<T, kittens::half>) {
                    i_t[idx] = __float2half(f);
                    i_ref[dev_idx][idx] = __half2float(i_t[idx]);
                }
                else {
                    assert(false && "Unsupported data type");
                }
            }

            cudaSetDevice(dev_idx);
            CUmemGenericAllocationHandle dev_handle; // no need to keep track of this in the tests
            kittens::pglCudaMalloc<true>(NUM_DEVICES, device_ids, dev_idx, &d_i_raw[dev_idx], &dev_handle, input_size * sizeof(T));
            kittens::pglCudaMalloc<true>(NUM_DEVICES, device_ids, dev_idx, &d_o_raw[dev_idx], &dev_handle, output_size * sizeof(T));
            cudaMemcpy(d_i_raw[dev_idx], i_t.data(), input_size * sizeof(T), cudaMemcpyHostToDevice);
            CudaCheckError();
        }

        *d_i = new PGL(device_ids, d_i_raw, B, nullptr, 16*R*H, nullptr);
        *d_o = new PGL(device_ids, d_o_raw, B, nullptr, 16*R*H, nullptr);
    }

    static test_result validate(PGL *d_i, PGL *d_o, const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref, std::string test_name, int cols=16, float eps=5e-2) { // default eps has to be fairly high due to lots of different types
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
                if constexpr (std::is_same_v<dtype, kittens::bf16>) {
                    o[idx] = __bfloat162float(o_t[idx]);
                    o_ref[dev_idx][idx] = __bfloat162float(__float2bfloat16(o_ref[dev_idx][idx]));
                }
                else if constexpr (std::is_same_v<dtype, kittens::half>) {
                    o[idx] = __half2float(o_t[idx]);
                    o_ref[dev_idx][idx] = __half2float(__float2half(o_ref[dev_idx][idx]));
                }
                else if constexpr(std::is_same_v<dtype, float>) {
                    o[idx] = o_t[idx];
                    o_ref[dev_idx][idx] = o_ref[dev_idx][idx];
                }
                #ifdef KITTENS_HOPPER
                else if constexpr(std::is_same_v<dtype, kittens::fp8e4m3>) {
                    o[idx] = float(o_t[idx]);
                    o_ref[dev_idx][idx] = float(__nv_fp8_e4m3(o_ref[dev_idx][idx])); 
                }
                else if constexpr(std::is_same_v<dtype, kittens::fp8e5m2>) {
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
            std::cerr << "Warning: Not enough GPU devices to run test " << test::test_identifier << std::endl;
            return;
        }
        test_info this_result;
        this_result.label = generate_test_name<H, W, NUM_WORKERS>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS>::value) {
            // initialize
            PGL *d_i, *d_o;
            std::vector<std::vector<float>> i_ref(NUM_DEVICES, std::vector<float>(SIZE));
            std::vector<std::vector<float>> o_ref(NUM_DEVICES, std::vector<float>(SIZE));
            initialize<dtype>(&d_i, &d_o, i_ref, o_ref);
            // run kernel
            for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
                cudaSetDevice(dev_idx);
                cudaFuncSetAttribute(
                    p2r_global_wrapper_2d<test, H, W, NUM_WORKERS, PGL>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kittens::MAX_SHARED_MEMORY
                );
                p2r_global_wrapper_2d<test, H, W, NUM_WORKERS, PGL><<<1, NUM_WORKERS*32>>>(*d_i, *d_o, dev_idx);
                cudaDeviceSynchronize();
                CudaCheckError();
            }
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, PGL>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W * 16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename T, kittens::ReduceOp op>
struct p2r_all_reduce_test {
    using dtype = T;
    using PGL = kittens::pgl<kittens::gl<dtype, 1, 1, 1, -1>, 8, true>;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int H, int W, int NW> using _valid_dims = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    template<int H, int W, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<H, W, NW>::value>;

    static inline const std::string op_identifier = op == kittens::ReduceOp::ADD ? "ADD" : op == kittens::ReduceOp::MIN ? "MIN" : "MAX";
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "p2r_all_reduce=bf16,op=" + op_identifier :
                                                      std::is_same_v<T, kittens::half> ? "p2r_all_reduce=half,op=" + op_identifier :
                                                                                         "p2r_all_reduce=float,op=" + op_identifier;
    template<int H, int W, int NW, kittens::ducks::pgl::all PGL>
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
    template<int H, int W, int NW, kittens::ducks::pgl::all PGL>
    __device__ static void device_func(const PGL &input, const PGL &output, const int dev_idx) {
        kittens::rt<dtype,  16*H, 16*W, kittens::ducks::rt_layout::row> reg_tile; // currently only row-major layout is supported
        for(int i = 0; i < input[dev_idx].batch(); i++) {
            for(int j = 0; j < input[dev_idx].depth(); j++) {
                for(int k = 0; k < input[dev_idx].rows()/reg_tile.rows; k++) {
                    for(int l = 0; l < input[dev_idx].cols()/reg_tile.cols; l++) {
                        if constexpr (op == kittens::ReduceOp::ADD) {
                            kittens::all_reduce_add(reg_tile, input, dev_idx, {i, j, k, l});
                        } else if constexpr (op == kittens::ReduceOp::MIN) {
                            kittens::all_reduce_min(reg_tile, input, dev_idx, {i, j, k, l});
                        } else if constexpr (op == kittens::ReduceOp::MAX) {
                            kittens::all_reduce_max(reg_tile, input, dev_idx, {i, j, k, l});
                        }
                        kittens::store(output[dev_idx], reg_tile, {i, j, k, l});
                    }
                }
            }
        }
    }
};

template<typename test, int NUM_DEVICES, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> 
using p2r_sweep_size_2d = mg_loop_h<p2r_test_wrapper_2d, test, NUM_DEVICES, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;

template<typename test, int NUM_DEVICES, int MAX_H=8, int MAX_W=8, typename... args> 
using p2r_sweep_size_2d_warp = p2r_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 1, args...>;

void warp::memory::tile::pgl_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/pgl_to_register tests! -----\n" << std::endl;
    constexpr int NUM_DEVICES = 8;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    p2r_sweep_size_2d_warp<p2r_all_reduce_test<float, kittens::ReduceOp::ADD>, NUM_DEVICES, SIZE, SIZE>::run(results);
}

#endif
