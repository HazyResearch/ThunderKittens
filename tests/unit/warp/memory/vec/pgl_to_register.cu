#include "pgl_to_register.cuh"

#ifdef TEST_WARP_MEMORY_VEC_PGL_TO_REGISTER

template<typename Ker, int S, int NW, kittens::ducks::pgl::all PGL, typename... args>
static __global__ void p2r_global_wrapper_1d(const __grid_constant__ PGL input, const __grid_constant__ PGL output, const __grid_constant__ int dev_idx) {
    Ker::template device_func<S, NW, PGL, args...>(input, output, dev_idx);
}

template<typename test, int NUM_DEVICES, int S, int NUM_WORKERS, typename... args>
struct p2r_test_wrapper_1d {
    constexpr static int SIZE = S * 16 * 3;

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
        this_result.label = generate_test_name<S, NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<S, NUM_WORKERS>::value) {

            // initialize
            dtype *d_i_arr[NUM_DEVICES];
            dtype *d_o_arr[NUM_DEVICES];
            std::vector<std::vector<float>> i_ref(NUM_DEVICES, std::vector<float>(SIZE));
            std::vector<std::vector<float>> o_ref(NUM_DEVICES, std::vector<float>(SIZE));
            int device_ids[NUM_DEVICES];
            for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;
            initialize<NUM_DEVICES>(device_ids, d_i_arr, d_o_arr, i_ref, o_ref);

            // set parralel global layouts
            shared_pgl_replace_gl<dtype, NUM_DEVICES>(d_i_arr, d_o_arr, 1, 1, 1, 16*S*3);
            shared_layout::input_pgl->multicast_bind();
            shared_layout::output_pgl->multicast_bind();

            // run kernel
            for (int dev_idx = 0; dev_idx < (test::single_run ? 1 : NUM_DEVICES); ++dev_idx) {
                cudaSetDevice(dev_idx);
                cudaFuncSetAttribute(
                    p2r_global_wrapper_1d<test, S, NUM_WORKERS, typename shared_layout::PGL, args...>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kittens::MAX_SHARED_MEMORY
                );
                p2r_global_wrapper_1d<test, S, NUM_WORKERS, typename shared_layout::PGL, args...><<<1, NUM_WORKERS*32>>>(*shared_layout::input_pgl, *shared_layout::output_pgl, dev_idx);
            }

            // fill in correct results on cpu
            test::template host_func<S, NUM_WORKERS, typename shared_layout::PGL>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate<NUM_DEVICES, typename shared_layout::PGL, dtype>(*shared_layout::input_pgl, *shared_layout::output_pgl, i_ref, o_ref, this_result.label, S * 16);

            shared_layout::input_pgl->multicast_unbind();
            shared_layout::output_pgl->multicast_unbind();
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename T, kittens::ReduceOp OP>
struct p2r_all_reduce_test {
    using dtype = T;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int S, int NW> using _valid_dims = std::bool_constant<NW == 1 && S <= 64>;
    template<int S, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<S, NW>::value>;
    static inline const bool single_run = false;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "p2r_all_reduce=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "p2r_all_reduce=half" :
                                                                                         "p2r_all_reduce=float";
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
        using RV = kittens::rv<dtype, 16*S, L>;
        RV reg_vec;
        int num_iters = (input.gl_size() + reg_vec.length - 1) / reg_vec.length;
        for (int i = 0; i < num_iters; i++) {
            if constexpr (OP == kittens::ReduceOp::ADD) {
                kittens::all_reduce_add(reg_vec, input, dev_idx, {i});
            } else if constexpr (OP == kittens::ReduceOp::MIN) {
                kittens::all_reduce_min(reg_vec, input, dev_idx, {i});
            } else if constexpr (OP == kittens::ReduceOp::MAX) {
                kittens::all_reduce_max(reg_vec, input, dev_idx, {i});
            }
            kittens::store(output[dev_idx], reg_vec, {i});
        }
    }
};

template<typename T>
struct p2r_atomic_add_test {
    using dtype = T;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int S, int NW> using _valid_dims = std::bool_constant<NW == 1 && S <= 64>;
    template<int S, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<S, NW>::value>;
    static inline const bool single_run = false;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "p2r_atomic_add=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "p2r_atomic_add=half" :
                                                                                         "p2r_atomic_add=float";
    template<int S, int NW, kittens::ducks::pgl::all PGL, typename... args>
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
    template<int S, int NW, kittens::ducks::pgl::all PGL, kittens::ducks::rv_layout::all L>
    __device__ static void device_func(const PGL &input, const PGL &output, const int dev_idx) {
        using RV = kittens::rv<dtype, 16*S, L>;
        RV reg_vec;
        int num_iters = (input.gl_size() + reg_vec.length - 1) / reg_vec.length;
        for (int i = 0; i < num_iters; i++) {
            kittens::load(reg_vec, input[dev_idx], {i});
            kittens::atomic_add(output, reg_vec, dev_idx, {i});
        }
    }
};

template<typename T>
struct p2r_broadcast_test {
    using dtype = T;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int S, int NW> using _valid_dims = std::bool_constant<NW == 1 && S <= 64>;
    template<int S, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<S, NW>::value>;
    static inline const bool single_run = true;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "p2r_broadcast=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "p2r_broadcast=half" :
                                                                                         "p2r_broadcast=float";
    template<int S, int NW, kittens::ducks::pgl::all PGL, typename... args>
    __host__ static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                o_ref[dev_idx][i] = i_ref[0][i];
            }
        }
    }
    template<int S, int NW, kittens::ducks::pgl::all PGL, kittens::ducks::rv_layout::all L>
    __device__ static void device_func(const PGL &input, const PGL &output, const int dev_idx) {
        using RV = kittens::rv<dtype, 16*S, L>;
        RV reg_vec;
        int num_iters = (input.gl_size() + reg_vec.length - 1) / reg_vec.length;
        for (int i = 0; i < num_iters; i++) {
            kittens::load(reg_vec, input[dev_idx], {i});
            kittens::broadcast(output, reg_vec, dev_idx, {i});
        }
    }
};

template<typename test, int NUM_DEVICES, int MAX_S=8, int NUM_WORKERS=1, typename... args> 
using p2r_sweep_size_1d = mg_loop_s<p2r_test_wrapper_1d, test, NUM_DEVICES, MAX_S, NUM_WORKERS, MAX_S, args...>;

template<typename test, int NUM_DEVICES, int MAX_S=8, typename... args>
using p2r_sweep_size_1d_warp = p2r_sweep_size_1d<test, NUM_DEVICES, MAX_S, 1, args...>;

template<typename test, int NUM_DEVICES, int MAX_S=8, typename... args>
struct p2r_sweep_size_1d_warp_layouts {
    static void run(test_data &results) {    
        p2r_sweep_size_1d_warp<test, NUM_DEVICES, MAX_S, kittens::ducks::rv_layout::ortho>::run(results);
        p2r_sweep_size_1d_warp<test, NUM_DEVICES, MAX_S, kittens::ducks::rv_layout::align>::run(results);
        p2r_sweep_size_1d_warp<test, NUM_DEVICES, MAX_S, kittens::ducks::rv_layout::naive>::run(results);
    }
};

// This might seem like an overkill, but is needed to minimize the number of PGL instantiations
template<typename T, int NUM_DEVICES, int MAX_S=8, typename... args>
struct p2r_sweep_size_1d_warp_axes_ops {
    static void run(test_data &results) {    
        using shared_layout = shared_layouts<T, NUM_DEVICES>;

        // Initialize shared PGLs
        int device_ids[NUM_DEVICES];
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;
        T *d_arr[NUM_DEVICES] = {}; // nullptrs (create dummy GLs)
        shared_layout::input_pgl = new shared_layout::PGL(device_ids, d_arr, 1, 1, 1, 16*MAX_S*3); // dummy dimensions, only used for mc allocation
        shared_layout::output_pgl = new shared_layout::PGL(device_ids, d_arr, 1, 1, 1, 16*MAX_S*3);
        shared_layout::input_pgl->multicast_init(); // can't to bind, but init is fine with just the dimensions
        shared_layout::output_pgl->multicast_init();

        p2r_sweep_size_1d_warp_layouts<p2r_all_reduce_test<T, kittens::ReduceOp::ADD>, NUM_DEVICES, MAX_S>::run(results);
        if constexpr (!std::is_same<T, float>::value) {
            p2r_sweep_size_1d_warp_layouts<p2r_all_reduce_test<T, kittens::ReduceOp::MIN>, NUM_DEVICES, MAX_S>::run(results);
            p2r_sweep_size_1d_warp_layouts<p2r_all_reduce_test<T, kittens::ReduceOp::MAX>, NUM_DEVICES, MAX_S>::run(results);
        }

        p2r_sweep_size_1d_warp_layouts<p2r_atomic_add_test<T>, NUM_DEVICES, MAX_S>::run(results);
        p2r_sweep_size_1d_warp_layouts<p2r_broadcast_test<T>, NUM_DEVICES, MAX_S>::run(results);

        // Delete shared PGLs
        shared_layout::input_pgl->multicast_destroy();
        shared_layout::output_pgl->multicast_destroy();
        delete shared_layout::input_pgl;
        delete shared_layout::output_pgl;
    }
};

void warp::memory::vec::pgl_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/pgl_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    if (check_multi_gpus()) {
        p2r_sweep_size_1d_warp_axes_ops<float, NUM_GPUS, SIZE>::run(results);
        p2r_sweep_size_1d_warp_axes_ops<kittens::bf16, NUM_GPUS, SIZE>::run(results);
        p2r_sweep_size_1d_warp_axes_ops<kittens::half, NUM_GPUS, SIZE>::run(results);
    }
}

#endif
