#include "pgl_to_register.cuh"

#ifdef TEST_GROUP_MEMORY_VEC_PGL_TO_REGISTER

template<typename Ker, int S, int NW, kittens::ducks::pgl::all PGL, typename... args>
static __global__ void group_vec_p2r_global_wrapper_1d(const __grid_constant__ PGL input, const __grid_constant__ PGL output, const __grid_constant__ int dev_idx) {
    Ker::template device_func<S, NW, PGL, args...>(input, output, dev_idx);
}

template<typename test, int NUM_DEVICES, int S, int NUM_WORKERS, typename... args>
struct group_vec_p2r_test_wrapper_1d {
    constexpr static int SIZE = S * 16 * 3;

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
        this_result.label = generate_test_name<S, NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<S, NUM_WORKERS>::value) {

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
                inputs.emplace_back(d_i_mc_arr[dev_idx], d_i_arr, 1, 1, 1, 16*S*3);
                outputs.emplace_back(d_o_mc_arr[dev_idx], d_o_arr, 1, 1, 1, 16*S*3);
            }

            // run kernel
            for (int dev_idx = 0; dev_idx < (test::single_run ? 1 : NUM_DEVICES); ++dev_idx) {
                cudaSetDevice(dev_idx);
                cudaFuncSetAttribute(
                    group_vec_p2r_global_wrapper_1d<test, S, NUM_WORKERS, PGL, args...>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kittens::MAX_SHARED_MEMORY
                );
                group_vec_p2r_global_wrapper_1d<test, S, NUM_WORKERS, PGL, args...><<<1, NUM_WORKERS*32>>>(inputs[dev_idx], outputs[dev_idx], dev_idx);
            }

            // fill in correct results on cpu
            test::template host_func<S, NUM_WORKERS, PGL>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate<NUM_DEVICES, PGL, dtype>(
                inputs, outputs,
                d_i_alloc_size, d_o_alloc_size, d_i_mc_alloc_size, d_o_mc_alloc_size,
                i_ref, o_ref, this_result.label, S * 16
            );
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename T, kittens::ReduceOp OP>
struct group_vec_p2r_all_reduce_test {
    using dtype = T;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int S, int NW> using _valid_dims = std::bool_constant<S <= 64>;
    template<int S, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<S, NW>::value>;
    static inline const bool single_run = false;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_vec_p2r_all_reduce=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_vec_p2r_all_reduce=half" :
                                                                                         "group_vec_p2r_all_reduce=float";
    template<int S, int NW, kittens::ducks::pgl::all PGL, typename... args>
    __host__ static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                o_ref[dev_idx][i] = i_ref[0][i];
                for (int other_dev_idx = 1; other_dev_idx < i_ref.size(); ++other_dev_idx) {
                    if constexpr (OP == kittens::ReduceOp::ADD) {
                        o_ref[dev_idx][i] += i_ref[other_dev_idx][i];
                    } else if constexpr (OP == kittens::ReduceOp::MIN) {
                        o_ref[dev_idx][i] = std::min(o_ref[dev_idx][i], i_ref[other_dev_idx][i]);
                    } else if constexpr (OP == kittens::ReduceOp::MAX) {
                        o_ref[dev_idx][i] = std::max(o_ref[dev_idx][i], i_ref[other_dev_idx][i]);
                    }
                }
            }
        }
    }
    template<int S, int NW, kittens::ducks::pgl::all PGL, kittens::ducks::rv_layout::all L>
    __device__ static void device_func(const PGL &input, const PGL &output, const int dev_idx) {
        using G = kittens::group<NW>;
        using RV = kittens::rv<dtype, 16*S, L>;
        RV reg_vec;
        int num_iters = (input.cols() + reg_vec.length - 1) / reg_vec.length;
        for (int i = 0; i < num_iters; i++) {
            if constexpr (OP == kittens::ReduceOp::ADD) {
                G::template all_reduce_add(reg_vec, input, {i});
            } else if constexpr (OP == kittens::ReduceOp::MIN) {
                G::template all_reduce_min(reg_vec, input, {i});
            } else if constexpr (OP == kittens::ReduceOp::MAX) {
                G::template all_reduce_max(reg_vec, input, {i});
            }
            G::template store(output[dev_idx], reg_vec, {i});
        }
    }
};

template<typename test, int NUM_DEVICES, int MAX_S, int NUM_WORKERS, typename... args> 
using group_vec_p2r_sweep_size_1d = mg_loop_s<group_vec_p2r_test_wrapper_1d, test, NUM_DEVICES, MAX_S, NUM_WORKERS, MAX_S, args...>;

template<typename test, int NUM_DEVICES, int MAX_S, typename... args>
struct group_vec_p2r_sweep_size_1d_warp_layouts_nw {
    static void run(test_data &results) {    
        group_vec_p2r_sweep_size_1d<test, NUM_DEVICES, MAX_S, 2, kittens::ducks::rv_layout::ortho>::run(results);
        group_vec_p2r_sweep_size_1d<test, NUM_DEVICES, MAX_S, 4, kittens::ducks::rv_layout::ortho>::run(results);
        group_vec_p2r_sweep_size_1d<test, NUM_DEVICES, MAX_S, 12, kittens::ducks::rv_layout::ortho>::run(results);
        group_vec_p2r_sweep_size_1d<test, NUM_DEVICES, MAX_S, 2, kittens::ducks::rv_layout::align>::run(results);
        group_vec_p2r_sweep_size_1d<test, NUM_DEVICES, MAX_S, 4, kittens::ducks::rv_layout::align>::run(results);
        group_vec_p2r_sweep_size_1d<test, NUM_DEVICES, MAX_S, 12, kittens::ducks::rv_layout::align>::run(results);
        group_vec_p2r_sweep_size_1d<test, NUM_DEVICES, MAX_S, 2, kittens::ducks::rv_layout::naive>::run(results);
        group_vec_p2r_sweep_size_1d<test, NUM_DEVICES, MAX_S, 4, kittens::ducks::rv_layout::naive>::run(results);
        group_vec_p2r_sweep_size_1d<test, NUM_DEVICES, MAX_S, 12, kittens::ducks::rv_layout::naive>::run(results);
    }
};

template<typename T, int NUM_DEVICES, int MAX_S, typename... args>
struct group_vec_p2r_sweep_size_1d_warp_axes_ops {
    static void run(test_data &results) {    
        group_vec_p2r_sweep_size_1d_warp_layouts_nw<group_vec_p2r_all_reduce_test<T, kittens::ReduceOp::ADD>, NUM_DEVICES, MAX_S>::run(results);
        if constexpr (!std::is_same<T, float>::value) {
            group_vec_p2r_sweep_size_1d_warp_layouts_nw<group_vec_p2r_all_reduce_test<T, kittens::ReduceOp::MIN>, NUM_DEVICES, MAX_S>::run(results);
            group_vec_p2r_sweep_size_1d_warp_layouts_nw<group_vec_p2r_all_reduce_test<T, kittens::ReduceOp::MAX>, NUM_DEVICES, MAX_S>::run(results);
        }
    }
};

void group::memory::vec::pgl_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/vec/pgl_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    if (check_multi_gpus()) {
        group_vec_p2r_sweep_size_1d_warp_axes_ops<float, NUM_GPUS, SIZE>::run(results);
        group_vec_p2r_sweep_size_1d_warp_axes_ops<kittens::bf16, NUM_GPUS, SIZE>::run(results);
        group_vec_p2r_sweep_size_1d_warp_axes_ops<kittens::half, NUM_GPUS, SIZE>::run(results);
    }
}

#endif
