#pragma once

#include <iostream>
#include <fstream>
#include <random>
#include <vector>

#include <cuda/pipeline>
#include <cooperative_groups.h>

#include "../src/kittens.cuh"

template<kittens::st_layout layout> std::string layout_name();
template<> std::string layout_name<kittens::st_naive_row_0b_layout      >() { return "st_naive_row_0b_layout";       }
template<> std::string layout_name<kittens::st_naive_row_32b_layout  >() { return "st_naive_row_32b_layout";   }
template<> std::string layout_name<kittens::st_naive_row_64b_layout  >() { return "st_naive_row_64b_layout";   }
template<> std::string layout_name<kittens::st_naive_row_128b_layout >() { return "st_naive_row_128b_layout";  }
template<> std::string layout_name<kittens::st_xor_row_layout        >() { return "st_xor_row_layout";         }
template<> std::string layout_name<kittens::st_wgmma_row_0b_layout   >() { return "st_wgmma_row_0b_layout";    }
template<> std::string layout_name<kittens::st_wgmma_row_32b_layout  >() { return "st_wgmma_row_32b_layout";   }
template<> std::string layout_name<kittens::st_wgmma_col_t_0b_layout >() { return "st_wgmma_col_t_0b_layout";  }
template<> std::string layout_name<kittens::st_wgmma_col_t_32b_layout>() { return "st_wgmma_col_t_32b_layout"; }

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

using namespace kittens;

#define ROWS (HEIGHT*16)
#define COLS (WIDTH*16)
#define SIZE (WIDTH*HEIGHT*256)
#define SEED 42

template<bool autofill=true>
void initialize(bf16 **d_i, bf16 **d_o, std::vector<float> &i_ref, std::vector<float> &o_ref) {

    const int input_size  = i_ref.size();
    const int output_size = o_ref.size();

    // Initialize matrices
    std::vector<bf16> i_bf(input_size);

    std::mt19937 gen(SEED); // Standard mersenne_twister_engine
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for(int idx = 0; idx < input_size; idx++) {
        float f = autofill ? dis(gen) : i_ref[idx];
        i_bf[idx] = __float2bfloat16(f); // fill in for transfer to device
        i_ref[idx] = __bfloat162float(i_bf[idx]); // ensure lossiness of fp16 is captured on cpu
    }

    cudaMalloc(d_i, input_size  * sizeof(bf16));
    cudaMalloc(d_o, output_size * sizeof(bf16));
    CudaCheckError();

    cudaMemcpy(*d_i, i_bf.data(), input_size * sizeof(bf16), cudaMemcpyHostToDevice);
    CudaCheckError();
}
bool validate(bf16 *d_i, bf16 *d_o, const std::vector<float> &i_ref, std::vector<float> &o_ref, std::string test_name, int cols=COLS, float eps=1e-4) {
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
    std::cout << "Test `" << test_name << "`\n";
    bool good = true;
    for(int i = 0; i < output_size; i++) {
        if(abs(o_ref[i] - o[i]) > eps) {
            good = false;
            break;
        }
    }
    if(good) std::cout << "PASSED" << std::endl;
    else std::cout << " ----- ALERT! FAILED TEST `" << test_name << "` -----" << std::endl;
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
    cudaFree(d_i);
    cudaFree(d_o);
    delete[] o_bf, o;
    CudaCheckError();
    return good;
}