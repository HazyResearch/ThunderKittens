#pragma once

/* testing_commons.cuh
 * 
 * This file contains a bunch of moderately test-specific utils.
 * For example, test_name constructors and __device__ kernel wrappers.
 * This file is distinguished from testing_utils.cuh in that you
 * might need to add to this file in order to add more tests,
 * but you shouldn't need to modify that testing_utils at all.
 */

#include <cuda/pipeline>
#include <cooperative_groups.h>

#include "kittens.cuh"

#include "testing_utils.cuh"

/* ---------- TEST NAMES ---------- */

// This how we generate parameterized names for tests.
// test_id is defined by the test, like "reg_mma" --
// then these templates build the rest of the test name.
// Note use of concepts to prevent template arg collisions!

// 1D test names
template<int S, int NW> std::string generate_test_name(std::string test_id) {
    std::string label = test_id+"_["+std::to_string(S)+"]";
    if constexpr (NW > 1) {
        label += "_["+std::to_string(NW)+"warps]";
    }
    return label;
}
template<int S, int NW, kittens::ducks::rt_layout::all L> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<S,NW>(test_id);
    if constexpr (std::is_same_v<L, kittens::ducks::rt_layout::row>) label += "_[rt_row_layout]";
    else label += "_[rt_col_layout]";
    return label;
}
template<int S, int NW, kittens::ducks::rt_layout::all L1, kittens::ducks::rt_layout::all L2> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<S,NW,L1>(test_id);
    if constexpr (std::is_same_v<L2, kittens::ducks::rt_layout::row>) label += "_[rt_row_layout]";
    else label += "_[rt_col_layout]";
    return label;
}
template<int S, int NW, kittens::ducks::rv_layout::all L> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<S,NW>(test_id);
    if constexpr (std::is_same_v<L, kittens::naive_l>) label += "_[rv_naive_layout]";
    else if constexpr (std::is_same_v<L, kittens::ortho_l>) label += "_[rv_ortho_layout]";
    else label += "_[rv_align_layout]";
    return label;
}
template<int S, int NW, kittens::ducks::rv_layout::all L1, kittens::ducks::rv_layout::all L2> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<S,NW,L1>(test_id);
    if constexpr (std::is_same_v<L2, kittens::naive_l>) label += "_[rv_naive_layout]";
    else if constexpr (std::is_same_v<L2, kittens::ortho_l>) label += "_[rv_ortho_layout]";
    else label += "_[rv_align_layout]";
    return label;
}

// 2D test names

template<int H, int W, int NW> std::string generate_test_name(std::string test_id) {
    std::string label = test_id+"_["+std::to_string(H)+"x"+std::to_string(W)+"]";
    if constexpr (NW > 1) {
        label += "_["+std::to_string(NW)+"warps]";
    }
    return label;
}
template <typename T> concept integral_wrapper = std::is_integral_v<decltype(T::value)>;
template<int H, int W, int NW, integral_wrapper _K> std::string generate_test_name(std::string test_id) {
    constexpr int K = _K::value;
    std::string label = test_id+"_["+std::to_string(H)+"x"+std::to_string(W)+"]x"+std::to_string(K);
    if constexpr (NW > 1) {
        label += "_["+std::to_string(NW)+"warps]";
    }
    return label;
}
template<int H, int W, int NW, kittens::ducks::rt_layout::all L> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<H,W,NW>(test_id);
    if constexpr (std::is_same_v<L, kittens::ducks::rt_layout::row>) label += "_[rt_row_layout]";
    else label += "_[rt_col_layout]";
    return label;
}
template<int H, int W, int NW, integral_wrapper _J, integral_wrapper _K> std::string generate_test_name(std::string test_id) {
    constexpr int J = _J::value, K = _K::value;
    std::string label = test_id+"_["+std::to_string(H)+"x"+std::to_string(W)+"_"+std::to_string(J)+"x"+std::to_string(K)+"]";
    if constexpr (NW > 1) {
        label += "_["+std::to_string(NW)+"warps]";
    }
    return label;
}
template<int H, int W, int NW, kittens::ducks::base_types::T1 T2, kittens::ducks::base_types::T1 U2> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<H,W,NW>(test_id);
    if constexpr (std::is_same_v<U2, float>) label += "_[float->";
    else if constexpr (std::is_same_v<U2, kittens::bf16>) label += "_[bf16->";
    #ifdef KITTENS_HOPPER
    else if constexpr (std::is_same_v<U2, kittens::fp8e4m3>) label += "_[e4m3->";
    else if constexpr (std::is_same_v<U2, kittens::fp8e5m2>) label += "_[e5m2->";
    #endif
    else label += "_[half->";
    if constexpr (std::is_same_v<T2, float>) label += "float]";
    else if constexpr (std::is_same_v<T2, kittens::bf16>) label += "bf16]";
    #ifdef KITTENS_HOPPER
    else if constexpr (std::is_same_v<T2, kittens::fp8e4m3>) label += "e4m3]";
    else if constexpr (std::is_same_v<T2, kittens::fp8e5m2>) label += "e5m2]";
    #endif
    else label += "half]";
    return label;
}


/* ---------- TEST WRAPPERS ---------- */

// These are wrappers to make it really easy to call and run tests.
// The basic wrappers:
// - Check if the test is valid and not compile it otherwise (the if constexpr)
// - Initialize input and output memory on both host and device
// - Call test functions on host and device
// - Validate outputs, append the result to test_data& results
// - Cleanup
// Additionally, the templated wrappers:
// - Loop through lots of template args in a grid to check validity.

template<typename T> concept has_dtype = requires { typename T::dtype; };
template<typename T>  struct gmem_wrapper    { using dtype = kittens::bf16; };
template<has_dtype T> struct gmem_wrapper<T> { using dtype = typename T::dtype; };
template<typename T> using gmem_dtype = typename gmem_wrapper<T>::dtype;

// ----- 1D Wrappers -----

template<typename Ker, typename T, int S, int NW, kittens::ducks::gl::all GL, typename... args>
static __global__ void global_wrapper_1d(GL input, const GL output) {
    Ker::template device_func<S, NW, GL, args...>(input, output);
}
template<typename test, int S, int NUM_WORKERS, typename... args>
struct wrapper_1d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<S,NUM_WORKERS,args...>(test::test_identifier);
        if constexpr (test::template valid<S, NUM_WORKERS, args...>::value) {
            constexpr int SIZE = S*16;
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, 1, 1, 1, S*16>;
            GL input(d_i, nullptr, nullptr, nullptr, nullptr);
            GL output(d_o, nullptr, nullptr, nullptr, nullptr);
            // run kernel
            cudaFuncSetAttribute(
                global_wrapper_1d<test, dtype, S, NUM_WORKERS, GL, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            global_wrapper_1d<test, dtype, S, NUM_WORKERS, GL, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(input, output);
            // fill in correct results on cpu
            test::template host_func<S, NUM_WORKERS, GL, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, S*16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int S, typename... args> using wrapper_1d_warp      = wrapper_1d<test, S, 1, args...>;
template<typename test, int S, typename... args> using wrapper_1d_warpgroup = wrapper_1d<test, S, 4, args...>;
template<typename test, int S, typename... args> using wrapper_1d_block     = wrapper_1d<test, S, 8, args...>;

template<typename test, int MAX_S=8, int NUM_WORKERS=1, typename... args> using sweep_size_1d = loop_s<wrapper_1d, test, MAX_S, NUM_WORKERS, MAX_S, args...>;
template<typename test, int MAX_S=8, typename... args> using sweep_size_1d_warp = sweep_size_1d<test, MAX_S, 1, args...>;


template<template<typename> typename test, int MAX_S=8, int NUM_WORKERS=1, typename... args>
struct sweep_gmem_type_1d {
    static void run(test_data &results) {
        sweep_size_1d<test<float>, MAX_S, NUM_WORKERS, args...>::run(results);
        sweep_size_1d<test<kittens::bf16>, MAX_S, NUM_WORKERS, args...>::run(results);
        sweep_size_1d<test<kittens::half>, MAX_S, NUM_WORKERS, args...>::run(results);
        #ifdef KITTENS_HOPPER
        sweep_size_1d<test<kittens::fp8e4m3>, MAX_S, NUM_WORKERS, args...>::run(results);
        sweep_size_1d<test<kittens::fp8e5m2>, MAX_S, NUM_WORKERS, args...>::run(results);
        #endif
    }
};
template<template<typename> typename test, int MAX_S=8, typename... args> using sweep_gmem_type_1d_warp = sweep_gmem_type_1d<test, MAX_S, 1, args...>;

// ----- 2D Wrappers -----

template<typename Ker, typename T, int H, int W, int NW, typename G, typename... args>
static __global__ void global_wrapper_2d(const __grid_constant__ G input, const __grid_constant__ G output) {
    Ker::template device_func<H, W, NW, G, args...>(input, output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int SIZE = H*W*256;
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, 1, 1, H*16, W*16>;
            GL input(d_i, nullptr, nullptr, nullptr, nullptr);
            GL output(d_o, nullptr, nullptr, nullptr, nullptr);
            // run kernel
            cudaFuncSetAttribute(
                global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(input, output);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GL, args...>(i_ref, o_ref);
            // check and cleanup

            int is_fp8 = (this_result.label.find("fp8") != std::string::npos) || (this_result.label.find("e4m3") != std::string::npos) || (this_result.label.find("e5m2") != std::string::npos);
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16, is_fp8 ? 0.1 : 5e-2); // mma's sometimes produce small errors. this appears to be hardware.
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int H, int W, typename... args> using wrapper_2d_warp      = wrapper_2d<test, H, W, 1, args...>;

template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using sweep_size_2d = loop_h<wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_size_2d_warp = sweep_size_2d<test, MAX_H, MAX_W, 1, args...>;

template<template<typename> typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
struct sweep_gmem_type_2d {
    static void run(test_data &results) {
        sweep_size_2d<test<float>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        sweep_size_2d<test<kittens::bf16>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        sweep_size_2d<test<kittens::half>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        #ifdef KITTENS_HOPPER
        sweep_size_2d<test<kittens::fp8e4m3>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        sweep_size_2d<test<kittens::fp8e5m2>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        #endif
    }
};
template<template<typename> typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_gmem_type_2d_warp = sweep_gmem_type_2d<test, MAX_H, MAX_W, 1, args...>;

template<typename T> concept gl_t = kittens::ducks::gl::all<T>;

// ----- Multi-GPU Test Wrappers -----

template<typename T, int NUM_DEVICES>
struct shared_layouts {
    // Do not initialize MC (we initialize with dummy GLs), and
    // make all dims runtime, so we can create one big PGL and share it across tests
    // The TMA type here is a dummy; we initialize TMA descriptors manually later if needed
    using PGL = kittens::pgl<kittens::gl<T, -1, -1, -1, -1>, NUM_DEVICES, false, false, kittens::st<T, 32, 32>>; 
    inline static PGL *input_pgl = nullptr;
    inline static PGL *output_pgl = nullptr;
};

template<typename T, int NUM_DEVICES>
inline void shared_pgl_replace_gl(T *d_i_arr[NUM_DEVICES], T *d_o_arr[NUM_DEVICES], int B, int D, int R, int C) {
    using shared_layout = shared_layouts<T, NUM_DEVICES>;

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        // This is super hacky, but I think there is no clean way to do this, and 
        // this sort of situation with hundreds of pgls of different dims is specific to testing
        shared_layout::input_pgl->gls[dev_idx].raw_ptr = d_i_arr[dev_idx];
        shared_layout::output_pgl->gls[dev_idx].raw_ptr = d_o_arr[dev_idx];
        shared_layout::input_pgl->gls[dev_idx].batch_internal.v = B;
        shared_layout::output_pgl->gls[dev_idx].batch_internal.v = B;
        shared_layout::input_pgl->gls[dev_idx].depth_internal.v = D;
        shared_layout::output_pgl->gls[dev_idx].depth_internal.v = D;
        shared_layout::input_pgl->gls[dev_idx].rows_internal.v = R;
        shared_layout::output_pgl->gls[dev_idx].rows_internal.v = R;
        shared_layout::input_pgl->gls[dev_idx].cols_internal.v = C;
        shared_layout::output_pgl->gls[dev_idx].cols_internal.v = C;
    }
}

inline int check_multi_gpus() {
#if NUM_GPUS > 1
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices < 2) {
        std::cerr << "Multi-GPU tests require at least 2 GPUs, found " << num_devices << "." << std::endl;
        std::cerr << "Please set the testing flag NUM_GPUS to 1 if your system does not have multiple GPUs." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return 1;
#else
    std::cout << "Skipping because the testing flag NUM_GPUS is less than 2." << std::endl;
    return 0;
#endif
}
