#pragma once

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <type_traits>

#include <cuda/pipeline>
#include <cooperative_groups.h>

#include "utils.cuh"
#include "kittens.cuh"

/* ---------- STRUCTS ---------- */

/*
General Unit Test Struct
struct test {
    template<typename... args> using valid = true; // Set this invalid if you don't want the test to be compiled and run.
    static inline const std::string test_identifier; ("block::load" as an example.)
    template<typename... args> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref);
    template<typename... args> __global__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output);
};
*/
enum test_result {
    PASSED = 0,
    FAILED = 1,
    INVALID = 2 // This is a useful one for tests that are only defined for certain template specializations, but we still want to sweep.
};
struct test_info {
    std::string label;
    test_result result;
};
using test_data = std::vector<test_info>;

/* ---------- LABELS AND TEST NAMES ---------- */

template<kittens::ducks::st_layout::all layout> std::string layout_name();

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
    std::string label = test_id+"_["+std::to_string(H)+"x"+std::to_string(K)+"x"+std::to_string(W)+"]";
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
template<int H, int W, int NW, kittens::ducks::st_layout::all L> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<H,W,NW>(test_id)+"_["+layout_name<L>()+"]";
    return label;
}
template<int H, int W, int NW, kittens::ducks::st_layout::all L1, kittens::ducks::st_layout::all L2> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<H,W,NW>(test_id) + "_["+layout_name<L2>()+"->"+layout_name<L1>()+"]";
    return label;
}
template<int H, int W, int NW, kittens::ducks::st_layout::all L, integral_wrapper _J, integral_wrapper _K> std::string generate_test_name(std::string test_id) {
    constexpr int J = _J::value, K = _K::value;
    std::string label = test_id+"_["+std::to_string(H)+"x"+std::to_string(W)+"_"+std::to_string(J)+"x"+std::to_string(K)+"]_["+layout_name<L>()+"]";
    if constexpr (NW > 1) {
        label += "_["+std::to_string(NW)+"warps]";
    }
    return label;
}
template<int H, int W, int NW, kittens::ducks::st_layout::all L, kittens::ducks::rt_layout::all RL> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<H,W,NW,L>(test_id);
    if constexpr (std::is_same_v<L, kittens::ducks::rt_layout::row>) label += "_[rt_row_layout]";
    else label += "_[rt_col_layout]";
    return label;
}
template<int H, int W, int NW, kittens::ducks::base_types::T2 T2, kittens::ducks::base_types::T2 U2> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<H,W,NW>(test_id);
    if constexpr (std::is_same_v<U2, float2>) label += "_[float2->";
    else label += "_[bf16_2->";
    if constexpr (std::is_same_v<T2, float2>) label += "float2]";
    else label += "bf16_2]";
    return label;
}


/* ---------- BASE HELPERS ---------- */

enum initializers {
    RANDOM = 0,
    ARANGE = 1,
    NONE   = 2
};
template<initializers initializer=initializers::RANDOM, int SEED=42>
void initialize(kittens::bf16 **d_i, kittens::bf16 **d_o, std::vector<float> &i_ref, std::vector<float> &o_ref) {
    using namespace kittens;

    const int input_size  = i_ref.size();
    const int output_size = o_ref.size();

    // Initialize matrices
    std::vector<bf16> i_bf(input_size);

    std::mt19937 gen(SEED); // Standard mersenne_twister_engine
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for(int idx = 0; idx < input_size; idx++) {
        float f;
        if constexpr (initializer == initializers::RANDOM) {
            f = dis(gen);
        }
        else if constexpr (initializer == initializers::ARANGE) {
            f = float(idx);
        }
        else {
            f = i_ref[idx];
        }
        i_bf[idx] = __float2bfloat16(f); // fill in for transfer to device
        i_ref[idx] = __bfloat162float(i_bf[idx]); // ensure lossiness of fp16 is captured on cpu
    }

    cudaMalloc(d_i, input_size  * sizeof(bf16));
    cudaMalloc(d_o, output_size * sizeof(bf16));
    CudaCheckError();

    cudaMemcpy(*d_i, i_bf.data(), input_size * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();
}
extern int should_write_outputs;
test_result validate(kittens::bf16 *d_i, kittens::bf16 *d_o, const std::vector<float> &i_ref, std::vector<float> &o_ref, std::string test_name, int cols, float eps=1e-4);

/* ---------- TEST WRAPPERS ---------- */

// 1D Wrappers

template<typename Ker, int S, int NW, typename... args>
static __global__ void global_wrapper_1d(const kittens::bf16 *input, kittens::bf16 *output) {
    Ker::template device_func<S, NW, args...>(input, output);
}
template<typename test, int S, int NUM_WORKERS, typename... args>
struct wrapper_1d {
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<S,NUM_WORKERS,args...>(test::test_identifier);
        if constexpr (test::template valid<S, NUM_WORKERS, args...>::value) {
            constexpr int SIZE = S*16;
            // initialize
            kittens::bf16 *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // run kernel
            cudaFuncSetAttribute(
                global_wrapper_1d<test, S, NUM_WORKERS, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            global_wrapper_1d<test, S, NUM_WORKERS, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(d_i, d_o);
            // fill in correct results on cpu
            test::template host_func<S, NUM_WORKERS, args...>(i_ref, o_ref);
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

template<template<typename,int,int,typename...> typename base, typename test, int MAX_S, int NUM_WORKERS, int S, typename... args>
struct loop_s {
    static void run(test_data& results) {
        base<test, S, NUM_WORKERS, args...>::run(results);
        if constexpr (S > 1) {
            loop_s<base, test, MAX_S, NUM_WORKERS, S-1, args...>::run(results);
        }
    }
};
template<typename test, int MAX_S=8, int NUM_WORKERS=1, typename... args> using sweep_size_1d = loop_s<wrapper_1d, test, MAX_S, NUM_WORKERS, MAX_S, args...>;
template<typename test, int MAX_S=8, typename... args> using sweep_size_1d_warp      = sweep_size_1d<test, MAX_S, 1, args...>;
template<typename test, int MAX_S=8, typename... args> using sweep_size_1d_warpgroup = sweep_size_1d<test, MAX_S, 4, args...>;


// 2D Wrappers

template<typename Ker, int H, int W, int NW, typename... args>
static __global__ void global_wrapper_2d(const kittens::bf16 *input, kittens::bf16 *output) {
    Ker::template device_func<H, W, NW, args...>(input, output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct wrapper_2d {
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int SIZE = H*W*256;
            // initialize
            kittens::bf16 *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // run kernel
            cudaFuncSetAttribute(
                global_wrapper_2d<test, H, W, NUM_WORKERS, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            global_wrapper_2d<test, H, W, NUM_WORKERS, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(d_i, d_o);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int H, int W, typename... args> using wrapper_2d_warp      = wrapper_2d<test, H, W, 1, args...>;
template<typename test, int H, int W, typename... args> using wrapper_2d_warpgroup = wrapper_2d<test, H, W, 4, args...>;

template<template<typename,int,int,int,typename...> typename base, typename test, int MAX_H, int MAX_W, int NUM_WORKERS, int H, int W, typename... args>
struct loop_w {
    static void run(test_data& results) {
        base<test, H, W, NUM_WORKERS, args...>::run(results);
        if constexpr (W > 1) {
            loop_w<base, test, MAX_H, MAX_W, NUM_WORKERS, H, W-1, args...>::run(results);
        }
    }
};
template<template<typename,int,int,int,typename...> typename base, typename test, int MAX_H, int MAX_W, int NUM_WORKERS, int H, typename... args>
struct loop_h {
    static void run(test_data& results) {
        loop_w<base, test, MAX_H, MAX_W, NUM_WORKERS, H, MAX_W, args...>::run(results);
        if constexpr (H > 1) {
            loop_h<base, test, MAX_H, MAX_W, NUM_WORKERS, H-1, args...>::run(results);
        }
    }
};
// template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
// struct sweep_size_2d {
//     static void run(test_data &results) {
//         loop_h<wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>::run(results);
//     }
// };
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using sweep_size_2d = loop_h<wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_size_2d_warp      = sweep_size_2d<test, MAX_H, MAX_W, 1, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_size_2d_warpgroup = sweep_size_2d<test, MAX_H, MAX_W, 4, args...>;


template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
struct sweep_st_layout_size_2d {
    static void run(test_data &results) {
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::naive, args...>::run(results);
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::tma_swizzle, args...>::run(results);
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::xor_swizzle, args...>::run(results);
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::wgmma_row_0b, args...>::run(results);
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::wgmma_row_32b, args...>::run(results);
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::wgmma_col_t_0b, args...>::run(results);
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::wgmma_col_t_32b, args...>::run(results);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_st_layout_size_2d_warp      = sweep_st_layout_size_2d<test, MAX_H, MAX_W, 1, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_st_layout_size_2d_warpgroup = sweep_st_layout_size_2d<test, MAX_H, MAX_W, 4, args...>;