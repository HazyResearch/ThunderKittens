#include "kittens.cuh"

#include <random>
#include <cuda_bf16.h>

constexpr int NUM_DEVICES = 2;
constexpr size_t N = 32;

using namespace kittens;

// Changed float to bf16 throughout the layouts
using global_layout   =  gl<bf16, 1, 1, -1, -1>;    
using st_tile = kittens::st<bf16, 16, 128>;
using rt_tile = kittens::rt<bf16, 16, 128>;
using sv_vec = kittens::sv<bf16, 128>;

constexpr int LLAMA_8B_HEAD_DIM = 128;
constexpr int GQA_RATIO = 4;

__device__ static inline void load_Q_async(st_tile &dst, global_layout src, int q_head_start_idx)
{
    using T = typename st_tile::dtype;
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(typename st_tile::dtype); // 8
    constexpr int memcpy_per_row = LLAMA_8B_HEAD_DIM / elem_per_memcpy;            // 16

    bf16 *src_ptr = &src.raw_ptr[q_head_start_idx * LLAMA_8B_HEAD_DIM];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));

    int laneid = kittens::laneid();
    int col = (laneid % memcpy_per_row) * elem_per_memcpy; // (0...15) * 8
    int base_row_in_group = (laneid < memcpy_per_row) ? 0 : 1;

    #pragma unroll
    for (int i = 0; i < (GQA_RATIO / 2); ++i)
    {
        int row = base_row_in_group + i * 2;
        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
            "r"(dst.idx(dst_ptr, {row, col})),
            "l"(&src_ptr[row * LLAMA_8B_HEAD_DIM + col])
            : "memory");
    }

    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <ducks::sv::all SV, ducks::rt::all RT>
__device__ static inline void store_4_rows(SV (&dst)[4], const RT &src)
{
    static_assert(RT::rows == 16, "src rows must be 16.");
    static_assert(SV::length == src.cols, "dst length must match src cols.");

    using T2 = RT::dtype;
    using T = base_types::packing<T2>::unpacked_type;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;

    uint32_t dst_ptr[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        dst_ptr[i] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[i].data[0]));
    }

    
    int laneid = kittens::laneid();

    
    if (laneid < 16)
    {
        int local_row_idx = laneid / 4;
        int local_col_idx = laneid % 4;

        for (int j = 0; j < src.width; j++)
        {
            U2 tmp[2];

            tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
            tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);

            int col_idx = local_col_idx * 2 + j * 16;

            move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
            move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
        }
    }
}

__global__ void test_kernel(global_layout a, global_layout b) {
    // extern __shared__ kittens::alignment_dummy __shm[];
    // kittens::shared_allocator al((int*)&__shm[0]);
    // st_tile (&s_tile) = al.allocate<st_tile>();
    // load_Q_async(s_tile, a, 0);
    // load_async_wait();

    // kittens::store(b, s_tile, {});

    // __syncwarp();

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);
    sv_vec (&s_vec)[4] = al.allocate<sv_vec, 4>();

    rt_tile r_tile;
    kittens::load(r_tile, a, {0, 0});
    store_4_rows(s_vec, r_tile);

    kittens::store(b, s_vec[0], {0, 0});
    kittens::store(b, s_vec[1], {1, 0});
    kittens::store(b, s_vec[2], {2, 0});
    kittens::store(b, s_vec[3], {3, 0});

    __syncwarp();
}

int main() {
    // Setup
    int nelem = 16 * 128;
    size_t size = nelem * sizeof(bf16);

    std::mt19937 gen(345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float *h_a_float = new float[nelem];
    // for (int i = 0; i < nelem; ++i) h_a_float[i] = 0.0f;
    for (int i = 0; i < nelem; ++i) h_a_float[i] = dist(gen);

    float *h_b_float = new float[nelem];
    for (int i = 0; i < nelem; ++i) h_b_float[i] = 0.0f;

    // Allocate host bf16 arrays for data transfer
    bf16 *h_a = new bf16[nelem];
    bf16 *h_b = new bf16[nelem];

    // Convert from float to bf16 for device transfer
    for (int i = 0; i < nelem; ++i) {
        h_a[i] = __float2bfloat16(h_a_float[i]);
        h_b[i] = __float2bfloat16(h_b_float[i]);
    }
    
    // Allocate and copy data to device
    bf16 *d_a, *d_b;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    global_layout a(d_a, nullptr, nullptr, 16, 128);
    global_layout b(d_b, nullptr, nullptr, 16, 128);

    // Perform the reduction
    unsigned long smem = kittens::MAX_SHARED_MEMORY;

    cudaFuncSetAttribute(test_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    dim3 grid(1);
    dim3 block(32);
    test_kernel<<<grid, block, smem>>>(a, b);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Bring back data
    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nelem; ++i) {
        h_a_float[i] = __bfloat162float(h_a[i]);
        h_b_float[i] = __bfloat162float(h_b[i]);
    }

    // Check correctness, comparing d_b with h_a_float
    bool correct = true;
    const float tolerance = 1e-3f; // Adjust based on bf16 precision requirements
    int errors = 0;
    float max_error = 0.0f;

    
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 128; ++j) {
            float error = std::abs(h_b_float[i * 128 + j] - h_a_float[i * 128 + j]);
            max_error = std::max(max_error, error);
            if (error > tolerance) {
                if (errors < 10) { // Print first 10 errors
                    printf("Mismatch at index (%d, %d): h_a_float = %f, h_b_float = %f, error = %f\n", 
                        i, j, h_a_float[i * 128 + j], h_b_float[i * 128 + j], error);
                }
                errors++;
                correct = false;
            }
        }
    }
    
    if (correct) {
        printf("Results are correct! Max error: %e\n", max_error);
    } else {
        printf("Results are INCORRECT! %d errors found. Max error: %e\n", errors, max_error);
    }

    // Cleanup and exit
    delete[] h_a;
    delete[] h_b;
    delete[] h_a_float;
    delete[] h_b_float;

    std::cout << "Done!" << std::endl;
    return 0;
}