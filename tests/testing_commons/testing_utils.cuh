#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <type_traits>
#include "kittens.cuh"

#define FIXED_FORMAT std::fixed << std::setprecision(6) << std::setw(10)

/* --------------------  CUDA ERROR UTILS  -------------------- */

#include <stdio.h>
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line ) {
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}

/* --------------------  TEST STRUCTS  -------------------- */

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


/* --------------------  TEMPLATE METAPROGRAMMING UTILS  -------------------- */

// 1D wrapper
template<template<typename,int,int,typename...> typename base, typename test, int MAX_S, int NUM_WORKERS, int S, typename... args>
struct loop_s {
    static void run(test_data& results) {
        if constexpr (S > 1) {
            loop_s<base, test, MAX_S, NUM_WORKERS, S-1, args...>::run(results);
        }
        base<test, S, NUM_WORKERS, args...>::run(results);
    }
};

// 2D wrappers
template<template<typename,int,int,int,typename...> typename base, typename test, int MAX_H, int MAX_W, int NUM_WORKERS, int H, int W, typename... args>
struct loop_w {
    static void run(test_data& results) {
        if constexpr (W > 1) {
            loop_w<base, test, MAX_H, MAX_W, NUM_WORKERS, H, W-1, args...>::run(results);
        }
        base<test, H, W, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename,int,int,int,typename...> typename base, typename test, int MAX_H, int MAX_W, int NUM_WORKERS, int H, typename... args>
struct loop_h {
    static void run(test_data& results) {
        if constexpr (H > 1) {
            loop_h<base, test, MAX_H, MAX_W, NUM_WORKERS, H-1, args...>::run(results);
        }
        loop_w<base, test, MAX_H, MAX_W, NUM_WORKERS, H, MAX_W, args...>::run(results);
    }
};

// 1D wrapper for multi-gpu tests
template<template<typename,int,int,int,typename...> typename base, typename test, int NUM_DEVICES, int MAX_S, int NUM_WORKERS, int S, typename... args>
struct mg_loop_s {
    static void run(test_data& results) {
        if constexpr (S > 1) {
            mg_loop_s<base, test, NUM_DEVICES, MAX_S, NUM_WORKERS, S-1, args...>::run(results);
        }
        base<test, NUM_DEVICES, S, NUM_WORKERS, args...>::run(results);
    }
};

// 2D wrappers for multi-gpu tests
template<template<typename,int,int,int,int,typename...> typename base, typename test, int NUM_DEVICES, int MAX_H, int MAX_W, int NUM_WORKERS, int H, int W, typename... args>
struct mg_loop_w {
    static void run(test_data& results) {
        if constexpr (W > 1) {
            mg_loop_w<base, test, NUM_DEVICES, MAX_H, MAX_W, NUM_WORKERS, H, W-1, args...>::run(results);
        }
        base<test, NUM_DEVICES, H, W, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename,int,int,int,int,typename...> typename base, typename test, int NUM_DEVICES, int MAX_H, int MAX_W, int NUM_WORKERS, int H, typename... args>
struct mg_loop_h {
    static void run(test_data& results) {
        if constexpr (H > 1) {
            mg_loop_h<base, test, NUM_DEVICES, MAX_H, MAX_W, NUM_WORKERS, H-1, args...>::run(results);
        }
        mg_loop_w<base, test, NUM_DEVICES, MAX_H, MAX_W, NUM_WORKERS, H, MAX_W, args...>::run(results);
    }
};

/* --------------------  TEST INITIALIZE+VALIDATE FUNCS  -------------------- */

enum initializers {
    RANDOM = 0, // uniform random init. useful for confirming correctness.
    ARANGE = 1, // write an increasing sequence into i_ref and d_i arrays. useful for debugging memory movement.
    NONE   = 2  // use whatever values were already in i_ref. useful for detailed debugging.
};
template<typename T, initializers initializer=initializers::RANDOM, int SEED=42>
void initialize(T **d_i, T **d_o, std::vector<float> &i_ref, std::vector<float> &o_ref) {
    using namespace kittens;

    const int input_size  = i_ref.size();
    const int output_size = o_ref.size();

    // Initialize matrices
    std::vector<T> i_t(input_size);

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
        if constexpr (std::is_same_v<T, bf16>) {
            i_t[idx] = __float2bfloat16(f); // fill in for transfer to device
            i_ref[idx] = __bfloat162float(i_t[idx]); // ensure lossiness of fp16 is captured on cpu
        }
        else if constexpr (std::is_same_v<T, float>) {
            i_t[idx] = f;
            i_ref[idx] = f;
        }
        else if constexpr (std::is_same_v<T, half>) {
            i_t[idx] = __float2half(f);
            i_ref[idx] = __half2float(i_t[idx]);
        }
        #ifdef KITTENS_HOPPER
        else if constexpr (std::is_same_v<T, fp8e4m3>) {
            i_t[idx] = __nv_fp8_e4m3(f); 
            i_ref[idx] = float(i_t[idx]); 
        } 
        else if constexpr (std::is_same_v<T, fp8e5m2>) {
            i_t[idx] = __nv_fp8_e5m2(f); 
            i_ref[idx] = float(i_t[idx]); 
        }
        #endif
        else {
            assert(false && "Unsupported data type");
        }
    }

    cudaMalloc(d_i, input_size  * sizeof(T));
    cudaMalloc(d_o, output_size * sizeof(T));
    CudaCheckError();

    cudaMemcpy(*d_i, i_t.data(), input_size * sizeof(T), cudaMemcpyHostToDevice);
    CudaCheckError();
}

// Initializer for multi-gpu tests
template<int NUM_DEVICES, typename T, initializers initializer=initializers::RANDOM, int SEED=42>
static void initialize(
    T **d_i_arr,
    T **d_o_arr,
    T **d_i_mc,
    T **d_o_mc,
    size_t *d_i_alloc_size,
    size_t *d_o_alloc_size,
    size_t *d_i_mc_alloc_size,
    size_t *d_o_mc_alloc_size,
    std::vector<std::vector<float>> &i_ref, 
    std::vector<std::vector<float>> &o_ref
) {
    
    const int input_size  = i_ref[0].size();
    const int output_size = o_ref[0].size();
    
    // Initialize matrices
    std::vector<T> i_t(input_size);
    
    std::mt19937 gen(SEED); // Standard mersenne_twister_engine
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    kittens::detail::vmm::handle d_i_mc_handle;
    kittens::detail::vmm::handle d_o_mc_handle;

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
        kittens::detail::vmm::vm_alloc_map_set_access((void **)&d_i_arr[dev_idx], d_i_alloc_size, input_size * sizeof(T), dev_idx, NUM_DEVICES);
        kittens::detail::vmm::vm_alloc_map_set_access((void **)&d_o_arr[dev_idx], d_o_alloc_size, output_size * sizeof(T), dev_idx, NUM_DEVICES);
        cudaMemcpy(d_i_arr[dev_idx], i_t.data(), input_size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemset(d_o_arr[dev_idx], 0, output_size * sizeof(T)); // for atomic tests

        if (dev_idx == 0) {
            kittens::detail::vmm::multicast_create_handle(&d_i_mc_handle, d_i_mc_alloc_size, *d_i_alloc_size, NUM_DEVICES);
            kittens::detail::vmm::multicast_create_handle(&d_o_mc_handle, d_o_mc_alloc_size, *d_o_alloc_size, NUM_DEVICES);        
        }
        kittens::detail::vmm::multicast_check(dev_idx);
        kittens::detail::vmm::multicast_bind_device(d_i_mc_handle, dev_idx);
        kittens::detail::vmm::multicast_bind_device(d_o_mc_handle, dev_idx);
        CudaCheckError();
    }

    // Binding memory to MC should be done after all device binding
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; dev_idx++) {
        kittens::detail::vmm::multicast_bind_address(d_i_mc_handle, d_i_arr[dev_idx], *d_i_alloc_size);
        kittens::detail::vmm::multicast_bind_address(d_o_mc_handle, d_o_arr[dev_idx], *d_o_alloc_size);
    }
    kittens::detail::vmm::vm_map((void **)d_i_mc, d_i_mc_handle, *d_i_mc_alloc_size);
    kittens::detail::vmm::vm_map((void **)d_o_mc, d_o_mc_handle, *d_o_mc_alloc_size);
    kittens::detail::vmm::vm_set_access((void *)*d_i_mc, *d_i_mc_alloc_size, NUM_DEVICES);
    kittens::detail::vmm::vm_set_access((void *)*d_o_mc, *d_o_mc_alloc_size, NUM_DEVICES);
    kittens::detail::vmm::vm_free(d_i_mc_handle);
    kittens::detail::vmm::vm_free(d_o_mc_handle);
}

extern int should_write_outputs;
template<typename T>
test_result validate(T *d_i, T *d_o, const std::vector<float> &i_ref, std::vector<float> &o_ref, std::string test_name, int cols, float eps=5e-2) { // default eps has to be fairly high due to lots of different types
    using namespace kittens;
    const int input_size  = i_ref.size();
    const int output_size = o_ref.size();
    // copy back
    T* o_t = new T[output_size];
    float *o = new float[output_size];
    cudaDeviceSynchronize();
    CudaCheckError();
    cudaMemcpy(o_t, d_o, output_size * sizeof(T), cudaMemcpyDeviceToHost);
    CudaCheckError();
    for(int idx = 0; idx < output_size; idx++) {
        if constexpr (std::is_same_v<T, bf16>) {
            o[idx] = __bfloat162float(o_t[idx]);
            o_ref[idx] = __bfloat162float(__float2bfloat16(o_ref[idx]));
        }
        else if constexpr (std::is_same_v<T, half>) {
            o[idx] = __half2float(o_t[idx]);
            o_ref[idx] = __half2float(__float2half(o_ref[idx]));
        }
        else if constexpr(std::is_same_v<T, float>) {
            o[idx] = o_t[idx];
            o_ref[idx] = o_ref[idx];
        }
        #ifdef KITTENS_HOPPER
        else if constexpr(std::is_same_v<T, fp8e4m3>) {
            o[idx] = float(o_t[idx]);
            o_ref[idx] = float(__nv_fp8_e4m3(o_ref[idx])); 
        }
        else if constexpr(std::is_same_v<T, fp8e5m2>) {
            o[idx] = float(o_t[idx]);
            o_ref[idx] = float(__nv_fp8_e5m2(o_ref[idx])); 
        }
        #endif
        else {
            assert(false && "Unsupported data type");
        }
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
    if(should_write_outputs && !good) {
        std::ofstream reffile("outputs/"+test_name+"_ref.txt");
        std::ofstream outfile("outputs/"+test_name+"_out.txt");
        for(int i = 0; i < output_size; i++) {
            reffile << FIXED_FORMAT << o_ref[i] << ' ';
            outfile << FIXED_FORMAT << o[i] << ' ';
            if(i%cols == cols-1) {
                reffile << '\n';
                outfile << '\n';
            }
        }
        reffile << "\n\n\nINPUTS:\n\n";
        for(int i = 0; i < input_size; i++) {
            reffile << FIXED_FORMAT << i_ref[i] << ' ';
            if(i%cols == cols-1) {
                reffile << '\n';
            }
        }
        reffile.close();
        outfile.close();
    }
    cudaFree(d_i);
    cudaFree(d_o);
    delete[] o_t, o;
    CudaCheckError();
    return good ? test_result::PASSED : test_result::FAILED;
}

// Validation for multi-gpu tests
template<int NUM_DEVICES, kittens::ducks::pgl::all PGL, typename T>
test_result validate(
    PGL &input,
    PGL &output,
    size_t d_i_alloc_size,
    size_t d_o_alloc_size,
    size_t d_i_mc_alloc_size,
    size_t d_o_mc_alloc_size,
    const std::vector<std::vector<float>> &i_ref,
    std::vector<std::vector<float>> &o_ref,
    std::string test_name,
    int cols=16,
    float eps=1e-1
) { // default eps even higher due to multiple GPUs parallelizing
    const int input_size  = i_ref[0].size();
    const int output_size = o_ref[0].size();

    // copy back
    T* o_t = new T[NUM_DEVICES * output_size];
    float *o = new float[NUM_DEVICES * output_size];

    std::cout << "test `" << test_name << "`";
    bool good = true;

    // Wait for all devices to complete
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaSetDevice(dev_idx);
        cudaDeviceSynchronize();
        CudaCheckError();
    }

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        cudaMemcpy(&o_t[dev_idx * output_size], output[dev_idx].raw_ptr, output_size * sizeof(T), cudaMemcpyDeviceToHost);
        CudaCheckError();

        for(int idx = 0; idx < output_size; idx++) {
            int unit_idx = dev_idx * output_size + idx;
            if constexpr (std::is_same_v<T, kittens::bf16>) {
                o[unit_idx] = __bfloat162float(o_t[unit_idx]);
                o_ref[dev_idx][idx] = __bfloat162float(__float2bfloat16(o_ref[dev_idx][idx]));
            }
            else if constexpr (std::is_same_v<T, kittens::half>) {
                o[unit_idx] = __half2float(o_t[unit_idx]);
                o_ref[dev_idx][idx] = __half2float(__float2half(o_ref[dev_idx][idx]));
            }
            else if constexpr(std::is_same_v<T, float>) {
                o[unit_idx] = o_t[unit_idx];
                o_ref[dev_idx][idx] = o_ref[dev_idx][idx];
            }
            #ifdef KITTENS_HOPPER
            else if constexpr(std::is_same_v<T, kittens::fp8e4m3>) {
                o[unit_idx] = float(o_t[unit_idx]);
                o_ref[dev_idx][idx] = float(__nv_fp8_e4m3(o_ref[dev_idx][idx])); 
            }
            else if constexpr(std::is_same_v<T, kittens::fp8e5m2>) {
                o[unit_idx] = float(o_t[unit_idx]);
                o_ref[dev_idx][idx] = float(__nv_fp8_e5m2(o_ref[dev_idx][idx])); 
            }
            #endif
            else {
                assert(false && "Unsupported data type");
            }
        }

        // check
        for(int i = 0; i < output_size; i++) {
            if(abs(o_ref[dev_idx][i] - o[dev_idx * output_size + i]) > eps) {
                good = false;
                break;
            }
        }

        // Even if we failed, continue so we can print all the results in the output file
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
                outfile << o[dev_idx * output_size + i] << ' ';
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

    // Destroy multicast object
    kittens::detail::vmm::handle d_in_mc_handle;
    kittens::detail::vmm::handle d_out_mc_handle;
    kittens::detail::vmm::vm_retrieve_handle(&d_in_mc_handle, input.mc_ptr);
    kittens::detail::vmm::vm_retrieve_handle(&d_out_mc_handle, output.mc_ptr);        
    kittens::detail::vmm::vm_unmap(input.mc_ptr, d_i_mc_alloc_size);
    kittens::detail::vmm::vm_unmap(output.mc_ptr, d_o_mc_alloc_size);
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        kittens::detail::vmm::multicast_unbind_device(d_in_mc_handle, d_i_mc_alloc_size, dev_idx);
        kittens::detail::vmm::multicast_unbind_device(d_out_mc_handle, d_o_mc_alloc_size, dev_idx);
    }
    kittens::detail::vmm::vm_free(d_in_mc_handle);
    kittens::detail::vmm::vm_free(d_out_mc_handle);

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        kittens::detail::vmm::vm_unmap(input[dev_idx].raw_ptr, d_i_alloc_size);
        kittens::detail::vmm::vm_unmap(output[dev_idx].raw_ptr, d_o_alloc_size);
    }

    delete[] o_t, o;
    CudaCheckError();
    return good ? test_result::PASSED : test_result::FAILED;
}
