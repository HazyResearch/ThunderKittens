#include "pgl_to_shared.cuh"

#ifdef TEST_WARP_MEMORY_TILE_PGL_TO_SHARED

template<typename Ker, int H, int W, int NW, kittens::ducks::pgl::all PGL>
static __global__ void p2s_global_wrapper_2d(const __grid_constant__ PGL input, const __grid_constant__ PGL output, const __grid_constant__ int dev_idx) {
    Ker::template device_func<H, W, NW, PGL>(input, output, dev_idx);
}

template<typename test, int NUM_DEVICES, int H, int W, int NUM_WORKERS, typename... args>
struct p2s_test_wrapper_2d {
    constexpr static int B = 3, D = 1, R = 4, C = 5; // arbitrary mix of prime/composite numbers
    constexpr static int SIZE = H*W*256 * B * D * R * C;

    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    using PGL = kittens::pgl<kittens::gl<dtype, -1, D, -1, 16*C*W>, NUM_DEVICES, true>;

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
            dtype *d_i_arr[NUM_DEVICES];
            dtype *d_o_arr[NUM_DEVICES];
            std::vector<std::vector<float>> i_ref(NUM_DEVICES, std::vector<float>(SIZE));
            std::vector<std::vector<float>> o_ref(NUM_DEVICES, std::vector<float>(SIZE));
            int device_ids[NUM_DEVICES];
            for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;
            initialize<NUM_DEVICES>(device_ids, d_i_arr, d_o_arr, i_ref, o_ref);
            // make parralel global layouts
            PGL input(device_ids, d_i_arr, B, nullptr, 16*R*H, nullptr);
            PGL output(device_ids, d_o_arr, B, nullptr, 16*R*H, nullptr);
            // run kernel
            for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
                cudaSetDevice(dev_idx);
                cudaFuncSetAttribute(
                    p2s_global_wrapper_2d<test, H, W, NUM_WORKERS, PGL>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kittens::MAX_SHARED_MEMORY
                );
                p2s_global_wrapper_2d<test, H, W, NUM_WORKERS, PGL><<<1, NUM_WORKERS*32>>>(input, output, dev_idx);
                cudaDeviceSynchronize();
                CudaCheckError();
            }
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, PGL>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate<NUM_DEVICES, PGL, dtype>(input, output, i_ref, o_ref, this_result.label, W * 16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename T, kittens::ReduceOp op>
struct p2s_all_reduce_test {
    using dtype = T;
    using PGL = kittens::pgl<kittens::gl<dtype, 1, 1, 1, -1>, 8, true>;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int H, int W, int NW> using _valid_dims = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    template<int H, int W, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<H, W, NW>::value>;

    static inline const std::string op_identifier = op == kittens::ReduceOp::ADD ? "ADD" : op == kittens::ReduceOp::MIN ? "MIN" : "MAX";
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "p2s_all_reduce=bf16,op=" + op_identifier :
                                                      std::is_same_v<T, kittens::half> ? "p2s_all_reduce=half,op=" + op_identifier :
                                                                                         "p2s_all_reduce=float,op=" + op_identifier;
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
using p2s_sweep_size_2d = mg_loop_h<p2s_test_wrapper_2d, test, NUM_DEVICES, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;

template<typename test, int NUM_DEVICES, int MAX_H=8, int MAX_W=8, typename... args> 
using p2s_sweep_size_2d_warp = p2s_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 1, args...>;

void warp::memory::tile::pgl_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/pgl_to_shared tests! -----\n" << std::endl;
    constexpr int NUM_DEVICES = 8;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    p2s_sweep_size_2d_warp<p2s_all_reduce_test<float, kittens::ReduceOp::ADD>, NUM_DEVICES, SIZE, SIZE>::run(results);
}

#endif
