#include "kittens.cuh"

#include <random>
#include <cuda_bf16.h>

using namespace kittens;

constexpr int HEAD_DIM = 128;
// Changed float to bf16 throughout the layouts
using global_layout   =  gl<bf16, 1, 1, -1, -1>;
// using rt_tile = kittens::rt<bf16, 16, 64>;
using q_st = kittens::st<bf16, 16, HEAD_DIM>;
using q_sv = kittens::sv<bf16, HEAD_DIM>;
using q_rt = kittens::rt<bf16, 16, HEAD_DIM>;

template <ducks::sv::all SV, ducks::rt::all RT>
__device__ static inline void store_8_rows(SV (&dst)[8], const RT &src, int row4idx)
{
    static_assert(RT::rows == 16, "src rows must be 16.");
    static_assert(SV::length == src.cols, "dst length must match src cols.");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;

    uint32_t dst_ptr[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst_ptr[i] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[i].data[0]));
    }

    int laneid = kittens::laneid();
    int local_row_idx = (laneid % 32) / 4;
    int local_col_idx = laneid % 4;


    for (int j = 0; j < src.width; j++) {
        U2 tmp[2];
        tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
        tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);
        int col_idx = local_col_idx * 2 + j * 16;
        move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
        move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx+8), tmp[1]);
    }
}

__global__ void test_kernel(global_layout a_gl, global_layout b_gl) {
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);
    q_sv (&q_shared_vec)[8] = al.allocate<q_sv[8]>();
        
    q_rt q_register_tile;
    kittens::load(q_register_tile, a_gl, {0, 0});
    
    store_8_rows(q_shared_vec, q_register_tile, 0);
    __syncwarp();

    for (int i = 0; i < 8; ++i) {
        kittens::store(b_gl, q_shared_vec[i], {i, 0}); 
    }
}

int main() {
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    size_t a_size = 16 * HEAD_DIM * sizeof(bf16);
    size_t b_size = 16 * HEAD_DIM * sizeof(bf16);

    // Use float for host arrays and convert to/from bf16 during transfer
    float *host_mat_1_float = new float[16 * HEAD_DIM];
    // for (int i = 0; i < 16 * HEAD_DIM; ++i) host_mat_1_float[i] = 1.0f;
    // for (int i = 0; i < 16; ++i) {
    //     for (int j = 0; j < HEAD_DIM; ++j) {
    //         if (j < 16) {
    //             host_mat_1_float[i * HEAD_DIM + j] = float((j % 16) - i);
    //         } else {
    //             host_mat_1_float[i * HEAD_DIM + j] = float((j % 16) - (2 * i));
    //         }
    //         // if (i >= 4) {
    //         // } 
    //         // // else if (i < 4) 
    //         // else
    //         // {
    //         //     // host_mat_1_float[i * HEAD_DIM + j] = float((j% 16) - 4);
    //         //     host_mat_1_float[i * HEAD_DIM + j] = float(j % 16);
    //         // } 
    //     }
    // }
    for (int i = 0; i < 16 * HEAD_DIM; ++i) host_mat_1_float[i] = dis(gen);

    float *host_mat_2_float = new float[16 * HEAD_DIM];
    for (int i = 0; i < 16 * HEAD_DIM; ++i) host_mat_2_float[i] = 0.0f;
    // for (int i = 0; i < 16 * 64; ++i) host_mat_2_float[i] = float(i % 16);
    // for (int i = 0; i < 16 * 64; ++i) host_mat_2_float[i] = float(i % 64);
    // for (int i = 0; i < 16 * HEAD_DIM; ++i) host_mat_2_float[i] = dis(gen);

    // for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 64; j++) {
    //         printf("%f ", host_mat_2_float[i * 64 + j]);
    //     }
    //     printf("\n");
    // }

    // Allocate host bf16 arrays for data transfer
    bf16 *host_mat_1 = new bf16[16 * HEAD_DIM];
    bf16 *host_mat_2 = new bf16[16 * HEAD_DIM];

    // Convert from float to bf16 for device transfer
    for (int i = 0; i < 16 * HEAD_DIM; ++i) {
        host_mat_1[i] = __float2bfloat16(host_mat_1_float[i]);
        host_mat_2[i] = __float2bfloat16(host_mat_2_float[i]);
    }


    /*
    Compute Reference
    */
    float cpu_reference[16 * HEAD_DIM];
    for (int i = 0; i < 16; ++i) {  
        for (int j = 0; j < HEAD_DIM; ++j) {
            cpu_reference[i] = host_mat_1_float[i * HEAD_DIM + j];
        }
    }
    
    
    bf16 *d_host_mat_1, *d_host_mat_2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_host_mat_1, a_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_host_mat_2, b_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_host_mat_1, host_mat_1, a_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_host_mat_2, host_mat_2, b_size, cudaMemcpyHostToDevice));

    global_layout a_gl(d_host_mat_1, nullptr, nullptr, 16, HEAD_DIM);
    global_layout b_gl(d_host_mat_2, nullptr, nullptr, 16, HEAD_DIM);

    unsigned long smem = 2 * 16 * HEAD_DIM * sizeof(bf16);


    dim3 grid(1);
    dim3 block(32);
    cudaSetDevice(0);
    test_kernel<<<grid, block, smem>>>(a_gl, b_gl);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Bring back data
    cudaMemcpy(host_mat_1, d_host_mat_1, a_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_mat_2, d_host_mat_2, b_size, cudaMemcpyDeviceToHost);
    
    // Convert from bf16 to float for printing results
    for (int i = 0; i < 16 * HEAD_DIM; ++i) {
        host_mat_1_float[i] = __bfloat162float(host_mat_1[i]);
        host_mat_2_float[i] = __bfloat162float(host_mat_2[i]);
    }
    
    // Print results
    printf("\n\n-------------------------------------------------------------------\n");
    // for (int i = 0; i < 16; ++i) {
    //     // if (i != 0) continue;
    //     for (int j = 0; j < HEAD_DIM; ++j) {
    //         // if (j % 16 == 0 && j != 0) printf("\n");

    //         if (j % 16 == 0 && j != 0) printf("  ");
    //         // if (j % 32 == 0 && j != 0) printf("\n");
    //         printf("%f ", host_mat_2_float[i * HEAD_DIM + j]);
    //     }
    //     printf("\n\n");
    // }   
    // printf("\n\n");

    // Check correctness, should be all ones
    bool passed = true;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < HEAD_DIM; ++j) {
            // if (i == 0 && j == 0) continue;
            if (abs(host_mat_2_float[i * HEAD_DIM + j] - host_mat_1_float[i * HEAD_DIM + j]) > 1e-2) {
                std::cout << "Mismatch at index " << i * HEAD_DIM + j << ": " << host_mat_2_float[i * HEAD_DIM + j] << " != " << host_mat_1_float[i + HEAD_DIM + j] << std::endl;
                passed = false;
            }
        }
    }
    if (!passed) {
        std::cerr << "Error: Mismatch found!" << std::endl;
        return -1;
    }
    std::cout << "All checks passed!" << std::endl;

    // Cleanup and exit
    delete[] host_mat_1;
    delete[] host_mat_2;
    delete[] host_mat_1_float;
    delete[] host_mat_2_float;

    std::cout << "Done!" << std::endl;
    return 0;
}