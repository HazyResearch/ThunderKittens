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

/* ---------- LABELS ---------- */

// map an st_layout to a string representing it.
template<kittens::ducks::st_layout::all layout> std::string layout_name();

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
template<int H, int W, int NW, integral_wrapper _K, kittens::ducks::st_layout::all L1, kittens::ducks::st_layout::all L2>
std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<H, W, NW, _K>(test_id)+"_["+layout_name<L1>()+"]_["+layout_name<L2>()+"]";
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

// ----- 1D Wrappers -----

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
template<typename test, int S, typename... args> using wrapper_1d_block     = wrapper_1d<test, S, 8, args...>;

template<typename test, int MAX_S=8, int NUM_WORKERS=1, typename... args> using sweep_size_1d = loop_s<wrapper_1d, test, MAX_S, NUM_WORKERS, MAX_S, args...>;
template<typename test, int MAX_S=8, typename... args> using sweep_size_1d_warp      = sweep_size_1d<test, MAX_S, 1, args...>;
template<typename test, int MAX_S=8, typename... args> using sweep_size_1d_warpgroup = sweep_size_1d<test, MAX_S, 4, args...>;
template<typename test, int MAX_S=8, typename... args> using sweep_size_1d_block     = sweep_size_1d<test, MAX_S, 8, args...>;


// ----- 2D Wrappers -----

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
template<typename test, int H, int W, typename... args> using wrapper_2d_block     = wrapper_2d<test, H, W, 8, args...>;

template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using sweep_size_2d = loop_h<wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_size_2d_warp      = sweep_size_2d<test, MAX_H, MAX_W, 1, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_size_2d_warpgroup = sweep_size_2d<test, MAX_H, MAX_W, 4, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_size_2d_block     = sweep_size_2d<test, MAX_H, MAX_W, 8, args...>;

// Loop over st_layouts too, since this is needed by a bunch of tests.
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
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_st_layout_size_2d_block     = sweep_st_layout_size_2d<test, MAX_H, MAX_W, 8, args...>;