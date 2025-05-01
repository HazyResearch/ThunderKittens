#include "kittens.cuh"

#include <random>
#include <cuda_bf16.h>

using namespace kittens;

// Changed float to bf16 throughout the layouts
using global_layout   =  gl<bf16, 1, 1, -1, -1>;
// using rt_tile = kittens::rt<bf16, 16, 64>;
using st_tile = kittens::st<bf16, 16, 64>;
using ab_rv_vec = kittens::rv<bf16, 64>;
using c_rv_vec = kittens::rv<bf16, 16>;

template<ducks::rv::naive_layout RV, ducks::st::all ST>
__device__ static inline void load_shared_to_vec(RV &dst, const ST &src, int row_idx) {
    using U2 = ST::dtype; 
    using T2 = RV::dtype;
    using U = base_types::packing<U2>::unpacked_type;
    using T = base_types::packing<T2>::unpacked_type;

    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    #pragma unroll
    for (auto w = 0; w < dst.outer_dim; ++w) {
        if (w < dst.outer_dim - 1 || dst.length %32 == 0 || laneid() < 16) {
            U tmp;
            move<T>::lds(tmp, src.idx(src_ptr, {row_idx, laneid() + (w * 32)}));
            dst[w][0] = base_types::convertor<T, U>::convert(tmp);
        }
    }
}
template<ducks::rv::naive_layout RV, typename T>
__device__ static inline void add_val_to_col(RV &dst, T val, int col_idx) {
    using U2 = typename RV::dtype;
    using U = base_types::packing<U2>::unpacked_type;

    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    if (laneid() == col_idx) 
    {   
        dst[0][0] += base_types::convertor<U, T>::convert(val);
    }
}
template<ducks::rv::naive_layout DST_RV, ducks::rv::naive_layout RV>
__device__ static inline void vec_mul(DST_RV &dst, const RV &a, const RV &b) {
    #pragma unroll
    for(int i = 0; i < dst.outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            float tmp = a[i][j] * b[i][j];
            dst[i][j] = tmp;
        }
    }
}
template<ducks::rv::naive_layout DST_RV, ducks::rv::naive_layout SRC_RV, ducks::st::all ST>
__device__ static inline void fma_ABt(DST_RV &d, const SRC_RV &a, const ST &b_smem, const DST_RV &c) {
    using T2 = SRC_RV::dtype;
    using T = base_types::packing<T2>::unpacked_type;

    SRC_RV b;
    rv_fl<64> buf; 
    for (int i = 0; i < 16; ++i) 
    {
        load_shared_to_vec(b, b_smem, i);
        __syncwarp();
        vec_mul(buf, a, b); // increase precision for bf16
        T tmp = kittens::sum(buf); // increase precision here too 
        add_val_to_col(d, tmp, i);
    }
}

__global__ void test_kernel(global_layout c_gl, global_layout a_gl, global_layout b_gl) {
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);
    st_tile (&b_smem) = al.allocate<st_tile>();
    kittens::load(b_smem, b_gl, {0, 0});
    __syncwarp();

    ab_rv_vec a;
    c_rv_vec c;
    kittens::load(a, a_gl, {0, 0});
    kittens::zero(c);

    fma_ABt(c, a, b_smem, c);

    kittens::store(c_gl, c, {0, 0}); 
}

int main() {
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    size_t a_size = 64 * sizeof(bf16);
    size_t b_size = 16 * 64 * sizeof(bf16);
    size_t c_size = 16 * sizeof(bf16);

    // Use float for host arrays and convert to/from bf16 during transfer
    float *host_mat_1_float = new float[64];
    // for (int i = 0; i < 64; ++i) host_mat_1_float[i] = 1.0f;
    // for (int i = 0; i < 64; ++i) host_mat_1_float[i] = float(i % 16);
    for (int i = 0; i < 64; ++i) host_mat_1_float[i] = dis(gen);

    float *host_mat_2_float = new float[16 * 64];
    // for (int i = 0; i < 16 * 64; ++i) host_mat_2_float[i] = 1.0f;
    // for (int i = 0; i < 16 * 64; ++i) host_mat_2_float[i] = float(i % 16);
    // for (int i = 0; i < 16 * 64; ++i) host_mat_2_float[i] = float(i % 64);
    for (int i = 0; i < 16 * 64; ++i) host_mat_2_float[i] = dis(gen);

    // for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 64; j++) {
    //         printf("%f ", host_mat_2_float[i * 64 + j]);
    //     }
    //     printf("\n");
    // }

    float *host_mat_3_float = new float[16];
    for (int i = 0; i < 16; ++i) host_mat_3_float[i] = 0.0f;

    // Allocate host bf16 arrays for data transfer
    bf16 *host_mat_1 = new bf16[64];
    bf16 *host_mat_2 = new bf16[16 * 64];
    bf16 *host_mat_3 = new bf16[16];

    // Convert from float to bf16 for device transfer
    for (int i = 0; i < 16 * 64; ++i) {
        if (i < 64) host_mat_1[i] = __float2bfloat16(host_mat_1_float[i]);
        host_mat_2[i] = __float2bfloat16(host_mat_2_float[i]);
    }


    /*
    Compute Reference
    */
    float cpu_reference[16];
    for (int i = 0; i < 16; ++i) {
        float sum = 0.0f;        
        for (int j = 0; j < 64; ++j) {
            sum += host_mat_1_float[j] * host_mat_2_float[i * 64 + j];
        }
        cpu_reference[i] = sum;
    }
    
    
    bf16 *d_host_mat_1, *d_host_mat_2, *d_host_mat_3;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_host_mat_1, a_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_host_mat_2, b_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_host_mat_3, c_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_host_mat_1, host_mat_1, a_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_host_mat_2, host_mat_2, b_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_host_mat_3, host_mat_3, c_size, cudaMemcpyHostToDevice));

    global_layout a_gl(d_host_mat_1, nullptr, nullptr, 1, 64);
    global_layout b_gl(d_host_mat_2, nullptr, nullptr, 16, 64);
    global_layout c_gl(d_host_mat_3, nullptr, nullptr, 1, 16);

    unsigned long smem = 2 * 16 * 64 * sizeof(bf16);


    dim3 grid(1);
    dim3 block(32);
    cudaSetDevice(0);
    test_kernel<<<grid, block, smem>>>(c_gl, a_gl, b_gl);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Bring back data
    cudaMemcpy(host_mat_3, d_host_mat_3, c_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_mat_2, d_host_mat_2, b_size, cudaMemcpyDeviceToHost);
    
    // Convert from bf16 to float for printing results
    for (int i = 0; i < 16 * 64; ++i) {
        if (i < 16) {
            host_mat_3_float[i] = __bfloat162float(host_mat_3[i]);
        }
        host_mat_2_float[i] = __bfloat162float(host_mat_2[i]);
    }
    
    // Print results
    printf("\n\n-------------------------------------------------------------------\n");
    for (int i = 0; i < 16; ++i) {
        printf("%f ", host_mat_3_float[i]);
    }   
    printf("\n");
    // for (int i = 0; i < 16; ++i) {
    //     if (i != 0) continue;
    //     for (int j = 0; j < 64; ++j) {
    //         if (j % 16 == 0 && j != 0) printf("\n");

    //         // if (j % 16 == 0 && j != 0) printf("  ");
    //         // if (j % 32 == 0 && j != 0) printf("\n");
    //         printf("%f ", host_mat_2_float[i * 64 + j]);
    //     }
    //     printf("\n\n");
    // }   
    // printf("\n\n");

    // Check correctness, should be all ones
    bool passed = true;
    for (int i = 0; i < 16; ++i) {
        if (abs(host_mat_3_float[i] - cpu_reference[i]) > 1e-2) {
            std::cout << "Mismatch at index " << i << ": " << host_mat_3_float[i] << " != " << cpu_reference[i] << std::endl;
            passed = false;
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