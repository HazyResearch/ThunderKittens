#pragma once

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <type_traits>

#include <cuda/pipeline>
#include <cooperative_groups.h>

#include "testing_utils.cuh"
#include "../src/kittens.cuh"
using namespace kittens;


/* ---------- STRUCTS ---------- */

/*
struct test {
    template<typename... args> using valid = true; // Set this invalid if you don't want the test to be compiled and run.
    static inline const std::string test_identifier; ("block::load" as an example.)
    template<typename... args> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref);
    template<typename... args> __global__ static void device_func(const bf16 *input, bf16 *output);
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
template<> std::string layout_name<kittens::ducks::st_layout::naive          >() { return "naive";           }
template<> std::string layout_name<kittens::ducks::st_layout::tma_swizzle    >() { return "tma_swizzle";     }
template<> std::string layout_name<kittens::ducks::st_layout::xor_swizzle    >() { return "xor_swizzle";     }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_row_0b   >() { return "wgmma_row_0b";    }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_row_32b  >() { return "wgmma_row_32b";   }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_col_t_0b >() { return "wgmma_col_t_0b";  }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_col_t_32b>() { return "wgmma_col_t_32b"; }

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
template<int H, int W, int NW, ducks::rt_layout::all L> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<H,W,NW>(test_id);
    if constexpr (std::is_same_v<L, ducks::rt_layout::row>) label += "_[rt_row_layout]";
    else label += "_[rt_col_layout]";
    return label;
}
template<int H, int W, int NW, ducks::st_layout::all L> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<H,W,NW>(test_id)+"_["+layout_name<L>()+"]";
    return label;
}
template<int H, int W, int NW, ducks::st_layout::all L, ducks::rt_layout::all RL> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<H,W,NW,L>(test_id);
    if constexpr (std::is_same_v<L, ducks::rt_layout::row>) label += "_[rt_row_layout]";
    else label += "_[rt_col_layout]";
    return label;
}


/* ---------- BASE HELPERS ---------- */

enum initializers {
    RANDOM = 0,
    ARANGE = 1,
    NONE   = 2
};
template<initializers initializer=initializers::RANDOM, int SEED=42>
void initialize(bf16 **d_i, bf16 **d_o, std::vector<float> &i_ref, std::vector<float> &o_ref) {

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
int should_write_outputs;
test_result validate(bf16 *d_i, bf16 *d_o, const std::vector<float> &i_ref, std::vector<float> &o_ref, std::string test_name, int cols, float eps=1e-4) {
    const int input_size  = i_ref.size();
    const int output_size = o_ref.size();
    // copy back
    bf16* o_bf = new bf16[output_size];
    float *o = new float[output_size];
    cudaDeviceSynchronize();
    CudaCheckError();
    cudaMemcpy(o_bf, d_o, output_size * sizeof(bf16), cudaMemcpyDeviceToHost);
    CudaCheckError();
    for(int idx = 0; idx < output_size; idx++) {
        o[idx] = __bfloat162float(o_bf[idx]);
        o_ref[idx] = __bfloat162float(__float2bfloat16(o_ref[idx]));
    }
    // check
    std::cout << "test `" << test_name << "`";
    bool good = true;
    for(int i = 0; i < output_size; i++) {
        if(abs(o_ref[i] - o[i]) > eps) {
            good = false;
            break;
        }
    }
    if(good) std::cout << " -- PASSED" << std::endl;
    else std::cout << " ----- ALERT! FAILED test `" << test_name << "` -----" << std::endl;
    if(should_write_outputs) {
        std::ofstream reffile("outputs/"+test_name+"_ref.txt");
        std::ofstream outfile("outputs/"+test_name+"_out.txt");
        for(int i = 0; i < output_size; i++) {
            reffile << o_ref[i] << ' ';
            outfile << o[i] << ' ';
            if(i%cols == cols-1) {
                reffile << '\n';
                outfile << '\n';
            }
        }
        reffile << "\n\n\nINPUTS:\n\n";
        for(int i = 0; i < input_size; i++) {
            reffile << i_ref[i] << ' ';
            if(i%cols == cols-1) {
                reffile << '\n';
            }
        }
        reffile.close();
        outfile.close();
    }
    cudaFree(d_i);
    cudaFree(d_o);
    delete[] o_bf, o;
    CudaCheckError();
    return good ? test_result::PASSED : test_result::FAILED;
}


/* ---------- TEST WRAPPERS ---------- */
template<typename Ker, int H, int W, int NW, typename... args>
static __global__ void global_wrapper_2d(const bf16 *input, bf16 *output) {
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
            bf16 *d_i, *d_o;
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
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, ducks::st_layout::naive, args...>::run(results);
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, ducks::st_layout::tma_swizzle, args...>::run(results);
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, ducks::st_layout::xor_swizzle, args...>::run(results);
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, ducks::st_layout::wgmma_row_0b, args...>::run(results);
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, ducks::st_layout::wgmma_row_32b, args...>::run(results);
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, ducks::st_layout::wgmma_col_t_0b, args...>::run(results);
        sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, ducks::st_layout::wgmma_col_t_32b, args...>::run(results);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_st_layout_size_2d_warp      = sweep_st_layout_size_2d<test, MAX_H, MAX_W, 1, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_st_layout_size_2d_warpgroup = sweep_st_layout_size_2d<test, MAX_H, MAX_W, 4, args...>;