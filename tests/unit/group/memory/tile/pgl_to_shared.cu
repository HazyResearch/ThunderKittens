#include "pgl_to_shared.cuh"

#ifdef TEST_GROUP_MEMORY_TILE_PGL_TO_SHARED

template<typename Ker, int H, int W, int NW, kittens::ducks::pgl::all PGL, typename... args>
static __global__ void group_p2s_global_wrapper_2d(const __grid_constant__ PGL input, const __grid_constant__ PGL output, const __grid_constant__ int dev_idx) {
    Ker::template device_func<H, W, NW, PGL, args...>(input, output, dev_idx);
}

constexpr static int B = 3, D = 1, R = 4, C = 5;

template<typename test, int NUM_DEVICES, int H, int W, int NUM_WORKERS, typename axis, typename... args>
struct group_p2s_test_wrapper_2d {
    constexpr static int SIZE = H*W*256 * B * D * R * C;

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
        this_result.label = generate_test_name<H, W, NUM_WORKERS, axis, args...>(test::test_identifier);
        static_assert(axis::value==0 || axis::value==1 || axis::value==2, "Axis must be 0, 1, or 2.");
        if constexpr (test::template valid<H, W, NUM_WORKERS>::value) {

            // initialize
            dtype *d_i_arr[NUM_DEVICES];
            dtype *d_o_arr[NUM_DEVICES];
            std::vector<std::vector<float>> i_ref(NUM_DEVICES, std::vector<float>(SIZE));
            std::vector<std::vector<float>> o_ref(NUM_DEVICES, std::vector<float>(SIZE));
            int device_ids[NUM_DEVICES];
            for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;
            initialize<NUM_DEVICES>(device_ids, d_i_arr, d_o_arr, i_ref, o_ref);

            // set parralel global layouts
            shared_pgl_replace_gl<dtype, NUM_DEVICES>(d_i_arr, d_o_arr, (axis::value==0?H*16:1)*B, (axis::value==1?H*16:1)*D, (axis::value==2?H*16:1)*R, 16*C*W);
            shared_layout::input_pgl->multicast_bind();
            shared_layout::output_pgl->multicast_bind();

            // run kernel
            for (int dev_idx = 0; dev_idx < (test::single_run ? 1 : NUM_DEVICES); ++dev_idx) {
                cudaSetDevice(dev_idx);
                cudaFuncSetAttribute(
                    group_p2s_global_wrapper_2d<test, H, W, NUM_WORKERS, typename shared_layout::PGL, axis, args...>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kittens::MAX_SHARED_MEMORY
                );
                group_p2s_global_wrapper_2d<test, H, W, NUM_WORKERS, typename shared_layout::PGL, axis, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(*shared_layout::input_pgl, *shared_layout::output_pgl, dev_idx);
            }

            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, typename shared_layout::PGL, axis>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate<NUM_DEVICES, typename shared_layout::PGL, dtype>(*shared_layout::input_pgl, *shared_layout::output_pgl, i_ref, o_ref, this_result.label, W * 16);

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
struct group_p2s_all_reduce_test {
    using dtype = T;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int H, int W, int NW> using _valid_dims = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    template<int H, int W, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<H, W, NW>::value>;
    static inline const bool single_run = false; // run device_func on all devices
    static inline const std::string op_identifier = op == kittens::ReduceOp::ADD ? "ADD" : op == kittens::ReduceOp::MIN ? "MIN" : "MAX";
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_p2s_all_reduce=bf16,op=" + op_identifier :
                                                      std::is_same_v<T, kittens::half> ? "group_p2s_all_reduce=half,op=" + op_identifier :
                                                                                         "group_p2s_all_reduce=float,op=" + op_identifier;
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
        extern __shared__ kittens::alignment_dummy __shm[]; // smem
        kittens::shared_allocator<1024> al((int*)&__shm[0]);
        using ST = kittens::st<T, 16*H, 16*W>;
        ST &shared_tile = al.allocate<ST>();
        int num_batches = axis::value==0 ? (int)input.batch()/shared_tile.rows : (int)input.batch();
        int num_depths = axis::value==1 ? (int)input.depth()/shared_tile.rows : (int)input.depth();
        int num_rows = axis::value==2 ? (int)input.rows()/shared_tile.rows : (int)input.rows();
        for(int i = 0; i < num_batches; i++) {
            for(int j = 0; j < num_depths; j++) {
                for(int k = 0; k < num_rows; k++) {
                    for(int l = 0; l < input.cols()/shared_tile.cols; l++) {
                        if constexpr (op == kittens::ReduceOp::ADD) {
                            G::template all_reduce_add<axis::value, false>(shared_tile, input, dev_idx, {i, j, k, l});
                        } else if constexpr (op == kittens::ReduceOp::MIN) {
                            G::template all_reduce_min<axis::value, false>(shared_tile, input, dev_idx, {i, j, k, l});
                        } else if constexpr (op == kittens::ReduceOp::MAX) {
                            G::template all_reduce_max<axis::value, false>(shared_tile, input, dev_idx, {i, j, k, l});
                        }
                        G::template store<axis::value, false>(output[dev_idx], shared_tile, {i, j, k, l});
                    }
                }
            }
        }
    }
};

template<typename T>
struct group_p2s_atomic_add_test {
    using dtype = T;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int H, int W, int NW> using _valid_dims = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    template<int H, int W, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<H, W, NW>::value>;
    static inline const bool single_run = false;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_p2s_atomic_add=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_p2s_atomic_add=half" :
                                                                                         "group_p2s_atomic_add=float";
    template<int H, int W, int NW, kittens::ducks::pgl::all PGL, typename axis>
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
    template<int H, int W, int NW, kittens::ducks::pgl::all PGL, typename axis>
    __device__ static void device_func(const PGL &input, const PGL &output, const int dev_idx) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // smem
        kittens::shared_allocator<1024> al((int*)&__shm[0]);
        using ST = kittens::st<T, 16*H, 16*W>;
        ST &shared_tile = al.allocate<ST>();
        int num_batches = axis::value==0 ? ((int)input.batch()/shared_tile.rows) : (int)input.batch();
        int num_depths = axis::value==1 ? ((int)input.depth()/shared_tile.rows) : (int)input.depth();
        int num_rows = axis::value==2 ? ((int)input.rows()/shared_tile.rows) : (int)input.rows();
        for(int i = 0; i < num_batches; i++) {
            for(int j = 0; j < num_depths; j++) {
                for(int k = 0; k < num_rows; k++) {
                    for(int l = 0; l < input.cols()/shared_tile.cols; l++) {
                        G::template load<axis::value, false>(shared_tile, input[dev_idx], {i, j, k, l});
                        G::template atomic_add<axis::value, false>(output, shared_tile, dev_idx, {i, j, k, l});
                    }
                }
            }
        }
    }
};

template<typename T>
struct group_p2s_broadcast_test {
    using dtype = T;
    using _valid_type = std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, kittens::bf16> || std::is_same_v<T, kittens::half>>;
    template<int H, int W, int NW> using _valid_dims = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    template<int H, int W, int NW> using valid = std::bool_constant<_valid_type::value && _valid_dims<H, W, NW>::value>;
    static inline const bool single_run = true;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_p2s_broadcast=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_p2s_broadcast=half" :
                                                                                         "group_p2s_broadcast=float";
    template<int H, int W, int NW, kittens::ducks::pgl::all PGL, typename axis>
    __host__ static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                o_ref[dev_idx][i] = i_ref[0][i];
            }
        }
    }
    template<int H, int W, int NW, kittens::ducks::pgl::all PGL, typename axis>
    __device__ static void device_func(const PGL &input, const PGL &output, const int dev_idx) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // smem
        kittens::shared_allocator<1024> al((int*)&__shm[0]);
        using ST = kittens::st<T, 16*H, 16*W>;
        ST &shared_tile = al.allocate<ST>();
        int num_batches = axis::value==0 ? ((int)input.batch()/shared_tile.rows) : (int)input.batch();
        int num_depths = axis::value==1 ? ((int)input.depth()/shared_tile.rows) : (int)input.depth();
        int num_rows = axis::value==2 ? ((int)input.rows()/shared_tile.rows) : (int)input.rows();
        for(int i = 0; i < num_batches; i++) {
            for(int j = 0; j < num_depths; j++) {
                for(int k = 0; k < num_rows; k++) {
                    for(int l = 0; l < input.cols()/shared_tile.cols; l++) {
                        G::template load<axis::value, false>(shared_tile, input[dev_idx], {i, j, k, l});
                        G::template broadcast<axis::value, false>(output, shared_tile, dev_idx, {i, j, k, l});
                    }
                }
            }
        }
    }
};

template<typename test, int NUM_DEVICES, int MAX_H, int MAX_W, int NUM_WORKERS, typename... args> 
using group_p2s_sweep_size_2d = mg_loop_h<group_p2s_test_wrapper_2d, test, NUM_DEVICES, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;

template<typename test, int NUM_DEVICES, int MAX_H, int MAX_W, typename... args>
struct group_p2s_sweep_size_2d_group_axes_nw {
    using I0_t = std::integral_constant<int, 0>;
    using I1_t = std::integral_constant<int, 1>;
    using I2_t = std::integral_constant<int, 2>;

    static void run(test_data &results) {
        group_p2s_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 2, I2_t>::run(results);
        group_p2s_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 4, I2_t>::run(results);
        group_p2s_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 12, I2_t>::run(results);
        group_p2s_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 2, I1_t>::run(results);
        group_p2s_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 4, I1_t>::run(results);
        group_p2s_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 12, I1_t>::run(results);
        group_p2s_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 2, I0_t>::run(results);
        group_p2s_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 4, I0_t>::run(results);
        group_p2s_sweep_size_2d<test, NUM_DEVICES, MAX_H, MAX_W, 12, I0_t>::run(results);
    }
};

// This might seem like an overkill, but is needed to minimize the number of PGL instantiations
template<typename T, int NUM_DEVICES, int MAX_H, int MAX_W, typename... args>
struct group_p2s_sweep_size_2d_group_axes_ops {
    static void run(test_data &results) {    
        using shared_layout = shared_layouts<T, NUM_DEVICES>;

        // Initialize shared PGLs
        int device_ids[NUM_DEVICES];
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;
        T *d_arr[NUM_DEVICES] = {}; // nullptrs (create dummy GLs)
        shared_layout::input_pgl = new shared_layout::PGL(device_ids, d_arr, B, D, 16*MAX_H*R, 16*MAX_W*C); // dummy dimensions, only used for mc allocation
        shared_layout::output_pgl = new shared_layout::PGL(device_ids, d_arr, B, D, 16*MAX_H*R, 16*MAX_W*C);
        shared_layout::input_pgl->multicast_init(); // can't to bind, but init is fine with just the dimensions
        shared_layout::output_pgl->multicast_init();

        group_p2s_sweep_size_2d_group_axes_nw<group_p2s_all_reduce_test<T, kittens::ReduceOp::ADD>, NUM_DEVICES, MAX_H, MAX_W>::run(results);
        if constexpr (!std::is_same<T, float>::value) {
            group_p2s_sweep_size_2d_group_axes_nw<group_p2s_all_reduce_test<T, kittens::ReduceOp::MIN>, NUM_DEVICES, MAX_H, MAX_W>::run(results);
            group_p2s_sweep_size_2d_group_axes_nw<group_p2s_all_reduce_test<T, kittens::ReduceOp::MAX>, NUM_DEVICES, MAX_H, MAX_W>::run(results);
        }

        group_p2s_sweep_size_2d_group_axes_nw<group_p2s_atomic_add_test<T>, NUM_DEVICES, MAX_H, MAX_W>::run(results);
        group_p2s_sweep_size_2d_group_axes_nw<group_p2s_broadcast_test<T>, NUM_DEVICES, MAX_H, MAX_W>::run(results);

        // Delete shared PGLs
        shared_layout::input_pgl->multicast_destroy();
        shared_layout::output_pgl->multicast_destroy();
        delete shared_layout::input_pgl;
        delete shared_layout::output_pgl;
    }
};

void group::memory::tile::pgl_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/tile/pgl_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    if (check_multi_gpus()) {
        group_p2s_sweep_size_2d_group_axes_ops<float, NUM_GPUS, SIZE, SIZE>::run(results);
        group_p2s_sweep_size_2d_group_axes_ops<kittens::bf16, NUM_GPUS, SIZE, SIZE>::run(results);
        group_p2s_sweep_size_2d_group_axes_ops<kittens::half, NUM_GPUS, SIZE, SIZE>::run(results);
    }
}

#endif
