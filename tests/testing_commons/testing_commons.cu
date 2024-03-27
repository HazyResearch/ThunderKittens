#include "testing_commons.cuh"

// Explicit specializations

template<> std::string layout_name<kittens::ducks::st_layout::naive          >() { return "naive";           }
template<> std::string layout_name<kittens::ducks::st_layout::tma_swizzle    >() { return "tma_swizzle";     }
template<> std::string layout_name<kittens::ducks::st_layout::xor_swizzle    >() { return "xor_swizzle";     }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_row_0b   >() { return "wgmma_row_0b";    }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_row_32b  >() { return "wgmma_row_32b";   }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_col_t_0b >() { return "wgmma_col_t_0b";  }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_col_t_32b>() { return "wgmma_col_t_32b"; }

int should_write_outputs;
test_result validate(kittens::bf16 *d_i, kittens::bf16 *d_o, const std::vector<float> &i_ref, std::vector<float> &o_ref, std::string test_name, int cols, float eps) {
    using namespace kittens;
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
    if(should_write_outputs && !good) {
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