#include "pgl_to_register.cuh"

#ifdef TEST_GROUP_MEMORY_TILE_PGL_TO_REGISTER

template<typename Ker, int H, int W, int NW, kittens::ducks::pgl::all PGL, typename... args>
static __global__ void group_p2r_global_wrapper_2d(const __grid_constant__ PGL input, const __grid_constant__ PGL output, const __grid_constant__ int dev_idx) {
    Ker::template device_func<H, W, NW, PGL, args...>(input, output, dev_idx);
}

constexpr static int B = 3, D = 1, R = 4, C = 5;

template<typename test, int NUM_DEVICES, int H, int W, int NUM_WORKERS, typename axis, typename... args>
struct group_p2r_test_wrapper_2d {
    constexpr static int SIZE = H*W*256 * B * D * R * C;

    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    using PGL = kittens::pgl<kittens::gl<dtype, -1, -1, -1, -1>, NUM_DEVICES>;

    static void run(test_data& results) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count < NUM_DEVICES) {
            std::cerr << "Warning: Not enough GPU devices to run test " << test::test_identifier << std::endl;
            return;
        }
        test_info this_result;
        this_result.label = generate_test_name<H, W, NUM_WORKERS, axis, args...>(test::test_identifier);
        static_assert(axis::value==0 || axis::value==1 || axis::value==2, "Axis must be 0, 1, or 2.");
        if constexpr (test::template valid<H, W, NUM_WORKERS>::value) {

            // initialize
            dtype *d_i_arr[NUM_DEVICES];
            dtype *d_o_arr[NUM_DEVICES];
            dtype *d_i_mc_arr[NUM_DEVICES];
            dtype *d_o_mc_arr[NUM_DEVICES];
            size_t d_i_alloc_size, d_o_alloc_size, d_i_mc_alloc_size, d_o_mc_alloc_size;
            std::vector<std::vector<float>> i_ref(NUM_DEVICES, std::vector<float>(SIZE));
            std::vector<std::vector<float>> o_ref(NUM_DEVICES, std::vector<float>(SIZE));
            initialize<NUM_DEVICES>(
                d_i_arr, d_o_arr, d_i_mc_arr, d_o_mc_arr,
                &d_i_alloc_size, &d_o_alloc_size, &d_i_mc_alloc_size, &d_o_mc_alloc_size,
                i_ref, o_ref
            );

            // Prepare PGLs
            std::vector<PGL> inputs;
            std::vector<PGL> outputs;
            for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
                inputs.emplace_back(d_i_mc_arr[dev_idx], d_i_arr, (axis::value==0?H*16:1)*B, (axis::value==1?H*16:1)*D, (axis::value==2?H*16:1)*R, 16*C*W);
                outputs.emplace_back(d_o_mc_arr[dev_idx], d_o_arr, (axis::value==0?H*16:1)*B, (axis::value==1?H*16:1)*D, (axis::value==2?H*16:1)*R, 16*C*W);
            }

            // run kernel
            for (int dev_idx = 0; dev_idx < (test::single_run ? 1 : NUM_DEVICES); ++dev_idx) {
                cudaSetDevice(dev_idx);
                cudaFuncSetAttribute(
                    group_p2r_global_wrapper_2d<test, H, W, NUM_WORKERS, PGL, axis, args...>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kittens::MAX_SHARED_MEMORY
                );
                group_p2r_global_wrapper_2d<test, H, W, NUM_WORKERS, PGL, axis, args...><<<1, NUM_WORKERS*32>>>(inputs[dev_idx], outputs[dev_idx], dev_idx);
            }

            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, PGL, axis>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate<NUM_DEVICES, PGL, dtype>(
                inputs, outputs,
                d_i_alloc_size, d_o_alloc_size, d_i_mc_alloc_size, d_o_mc_alloc_size,
                i_ref, o_ref, this_result.label, W * 16
            );
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename T, kittens::ReduceOp op>
struct group_p2r_all_reduce_test {
    using dtype = T;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int H, int W, int NW> using _valid_dims = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    template<int H, int W, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<H, W, NW>::value>;
    static inline const bool single_run = false; // run device_func on all devices
    static inline const std::string op_identifier = op == kittens::ReduceOp::ADD ? "ADD" : op == kittens::ReduceOp::MIN ? "MIN" : "MAX";
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_p2r_all_reduce=bf16,op=" + op_identifier :
                                                      std::is_same_v<T, kittens::half> ? "group_p2r_all_reduce=half,op=" + op_identifier :
                                                                                         "group_p2r_all_reduce=float,op=" + op_identifier;
    template<int H, int W, int NW, kittens::ducks::pgl::all PGL, typename axis>
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
    template<int H, int W, int NW, kittens::ducks::pgl::all PGL, typename axis>
    __device__ static void device_func(const PGL &input, const PGL &output, const int dev_idx) {
        using G = kittens::group<NW>;
        using RT = kittens::rt<dtype, 16*H/NW, 16*W, kittens::ducks::rt_layout::row>;
        RT reg_tile; // currently only row-major layout is supported
        int num_batches = axis::value==0 ? (int)input.batch()/(reg_tile.rows*NW) : (int)input.batch();
        int num_depths = axis::value==1 ? (int)input.depth()/(reg_tile.rows*NW) : (int)input.depth();
        int num_rows = axis::value==2 ? (int)input.rows()/(reg_tile.rows*NW) : (int)input.rows();
        for(int i = 0; i < num_batches; i++) {
            for(int j = 0; j < num_depths; j++) {
                for(int k = 0; k < num_rows; k++) {
                    for(int l = 0; l < input.cols()/reg_tile.cols; l++) {
                        if constexpr (op == kittens::ReduceOp::ADD) {
                            G::template all_reduce_add<axis::value>(reg_tile, input, {i, j, k, l});
                        } else if constexpr (op == kittens::ReduceOp::MIN) {
                            G::template all_reduce_min<axis::value>(reg_tile, input, {i, j, k, l});
                        } else if constexpr (op == kittens::ReduceOp::MAX) {
                            G::template all_reduce_max<axis::value>(reg_tile, input, {i, j, k, l});
                        }
                        G::template store<axis::value>(output[dev_idx], reg_tile, {i, j, k, l});
                    }
                }
            }
        }
    }
};

template<typename test, int NUM_DEVICES, int MAX_H, int MAX_W, int NUM_WORKERS, typename... args> 
using group_p2r_sweep_size_2d = mg_loop_h<group_p2r_test_wrapper_2d, test, NUM_DEVICES, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;

template<typename test, int NUM_DEVICES, int MAX_H, int MAX_W, typename... args>
struct group_p2r_sweep_size_2d_group_axes_nw {
    using I0_t = std::integral_constant<int, 0>;
    using I1_t = std::integral_constant<int, 1>;
    using I2_t = std::integral_constant<int, 2>;

    static void run(test_data &results) {    
        group_p2r_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 2, I2_t>::run(results);
        group_p2r_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 4, I2_t>::run(results);
        group_p2r_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 12, I2_t>::run(results);
        group_p2r_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 2, I1_t>::run(results);
        group_p2r_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 4, I1_t>::run(results);
        group_p2r_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 12, I1_t>::run(results);
        group_p2r_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 2, I0_t>::run(results);
        group_p2r_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 4, I0_t>::run(results);
        group_p2r_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 12, I0_t>::run(results);
    }
};

template<typename T, int NUM_DEVICES, int MAX_H, int MAX_W, typename... args>
struct group_p2r_sweep_size_2d_group_axes_ops {
    static void run(test_data &results) {    
        group_p2r_sweep_size_2d_group_axes_nw<group_p2r_all_reduce_test<T, kittens::ReduceOp::ADD>, NUM_DEVICES, MAX_H, MAX_W>::run(results);
        if constexpr (!std::is_same<T, float>::value) {
            group_p2r_sweep_size_2d_group_axes_nw<group_p2r_all_reduce_test<T, kittens::ReduceOp::MIN>, NUM_DEVICES, MAX_H, MAX_W>::run(results);
            group_p2r_sweep_size_2d_group_axes_nw<group_p2r_all_reduce_test<T, kittens::ReduceOp::MAX>, NUM_DEVICES, MAX_H, MAX_W>::run(results);
        }
    }
};

void group::memory::tile::pgl_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/tile/pgl_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    if (check_multi_gpus()) {
        group_p2r_sweep_size_2d_group_axes_ops<float, NUM_GPUS, SIZE, SIZE>::run(results);
        group_p2r_sweep_size_2d_group_axes_ops<kittens::bf16, NUM_GPUS, SIZE, SIZE>::run(results);
        group_p2r_sweep_size_2d_group_axes_ops<kittens::half, NUM_GPUS, SIZE, SIZE>::run(results);
    }
}

#endif
