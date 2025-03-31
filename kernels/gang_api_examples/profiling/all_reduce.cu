#include "kittens.cuh"

#include <random>
#include <cuda_bf16.h>

constexpr int NUM_DEVICES = 2;
constexpr int NUM_WARPS = 8;
constexpr size_t N = 16384;

/*
From multimem.cuh
*/
constexpr int WARPSIZE = 32;
constexpr int STRIDE = 32;
constexpr int MAX_VEC_SIZE = 16;
constexpr int THREADS_PER_BLOCK = 256;

using namespace kittens;

// Changed float to bf16 throughout the layouts
using global_layout   =  gl<bf16, 1, 1, -1, -1>;
using kittens_pgl = kittens::pgl<global_layout, 2, true>;
using rt_tile = kittens::rt<bf16, 64, 64>;
using st_tile = kittens::st<bf16, 64, 256>;

__global__ void kittens_all_reduce(kittens_pgl p_o, int dev_id) {
    int row = blockIdx.x;
    int col = (blockIdx.y * NUM_WARPS) + kittens::warpid();

    if ((row * rt_tile::rows) >= N || (col * rt_tile::cols) >= N) return;

    rt_tile tile;
    kittens::all_reduce_add(tile, p_o, dev_id, {row, col});
    kittens::broadcast(p_o, tile, dev_id, {row, col});
}

__global__ void shared_kittens_all_reduce(kittens_pgl p_o, int dev_id) {
    extern __shared__ kittens::alignment_dummy __shm[]; 
    kittens::shared_allocator al((int*)&__shm[0]);
    st_tile (&s_tile)[2] = al.allocate<st_tile, 2>();

    int row = blockIdx.x;
    int col = (blockIdx.y * 2) + warpgroup::groupid();
    if ((row * st_tile::rows) >= N || (col * st_tile::cols) >= N) return;

    warpgroup::all_reduce_add(s_tile[warpgroup::groupid()], p_o, dev_id, {row, col});
    warpgroup::sync(warpgroup::groupid());
    warpgroup::broadcast(p_o, s_tile[warpgroup::groupid()], dev_id, {row, col});
}

__global__ void all_reduce(__nv_bfloat16 *data, const int N) {

    if (blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) {
        printf("mc: only 1D grids and blocks should be passed in\n");
        return;
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARPSIZE;
    int lane_id = threadIdx.x % WARPSIZE;

    constexpr int N_per_iter = MAX_VEC_SIZE / sizeof(__nv_bfloat16);
    constexpr int N_per_warp_per_iter = N_per_iter * WARPSIZE;
    constexpr int N_per_warp = STRIDE * N_per_warp_per_iter;
    int start_idx = N_per_warp * warp_id;

    for (int i = 0; i < STRIDE; ++i) {
        int idx = start_idx + i * N_per_warp_per_iter + lane_id * N_per_iter;
        if (idx < N) {
            volatile float x, y, z, w; // hacking type to hold 2 bfloat16s
            __nv_bfloat16 *ptr = data + idx;
            asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(ptr) : "memory");
            asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
        }
        __syncthreads();
    }
}

int main() {
    // Setup
    int nelem = N * N;
    size_t size = nelem * sizeof(bf16);

    // Allocate and copy data to device
    bf16 **dev_mats = new bf16*[NUM_DEVICES];
    CUmemGenericAllocationHandle *dev_handles = new CUmemGenericAllocationHandle[NUM_DEVICES];
    
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;

    float *host_mat_1_float = new float[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_1_float[i] = 1.0f;
    bf16 *host_mat_1 = new bf16[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_1[i] = __float2bfloat16(host_mat_1_float[i]);

    float *host_mat_2_float = new float[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_2_float[i] = 0.0f;
    bf16 *host_mat_2 = new bf16[nelem];
    for (int i = 0; i < nelem; ++i) host_mat_2[i] = __float2bfloat16(host_mat_2_float[i]);
    
    cudaSetDevice(0);
    pglCudaMalloc<true>(NUM_DEVICES, device_ids, 0, &dev_mats[0], size);
    CHECK_CUDA_ERROR(cudaMemcpy(dev_mats[0], host_mat_1, size, cudaMemcpyHostToDevice));

    cudaSetDevice(1);
    pglCudaMalloc<true>(NUM_DEVICES, device_ids, 1, &dev_mats[1], size);
    CHECK_CUDA_ERROR(cudaMemcpy(dev_mats[1], host_mat_2, size, cudaMemcpyHostToDevice));


    // Initialize parallel global layout
    kittens_pgl dev_mat_pgl{device_ids, dev_mats, nullptr, nullptr, N, N};

    // Perform the reduction
    KittensClub club(device_ids, NUM_DEVICES);

    dim3 block(NUM_WARPS * kittens::WARP_THREADS);
    dim3 grid_rt(N / rt_tile::rows, N / rt_tile::cols);
    printf("Grid RT: %d %d, Block: %d %d\n", grid_rt.x, grid_rt.y, block.x, block.y);
    dim3 grid_st(N / st_tile::rows, N / (st_tile::cols * 2));
    printf("Grid ST: %d %d, Block: %d %d\n", grid_st.x, grid_st.y, block.x, block.y);

    cudaSetDevice(0);
    int NUM_PROFILE_ITERS = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_PROFILE_ITERS; ++i) {
        kittens_all_reduce<<<grid_rt, block>>>(dev_mat_pgl, 0);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / NUM_PROFILE_ITERS;
    printf("Average time per kittens kernel: %f seconds\n", avg_time);

    start = std::chrono::high_resolution_clock::now();
    size_t smem = 2 * st_tile::rows * st_tile::cols * sizeof(bf16);
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        shared_kittens_all_reduce, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    for (int i = 0; i < NUM_PROFILE_ITERS; ++i) {
        shared_kittens_all_reduce<<<grid_st, block, smem>>>(dev_mat_pgl, 0);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    avg_time = elapsed.count() / NUM_PROFILE_ITERS;
    printf("Average time per shared kittens kernel: %f seconds\n", avg_time);

    int N_per_block = THREADS_PER_BLOCK * STRIDE * (MAX_VEC_SIZE / sizeof(__nv_bfloat16));
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_PROFILE_ITERS; ++i) {
        all_reduce<<<(nelem + N_per_block - 1) / N_per_block, THREADS_PER_BLOCK>>>(dev_mat_pgl.mc_vas[0], nelem);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    avg_time = elapsed.count() / NUM_PROFILE_ITERS;
    printf("Average time per og kernel: %f seconds\n", avg_time);

    // Bring back data
    cudaSetDevice(0);
    cudaMemcpy(host_mat_1, dev_mats[0], size, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(host_mat_2, dev_mats[1], size, cudaMemcpyDeviceToHost);
    
    // Convert from bf16 to float for printing results
    for (int i = 0; i < nelem; ++i) {
        host_mat_1_float[i] = __bfloat162float(host_mat_1[i]);
        host_mat_2_float[i] = __bfloat162float(host_mat_2[i]);
    }

    // Check correctness, should be all ones
    for (int i = 0; i < nelem; ++i) {
        if (host_mat_1_float[i] != 1.0f) {
            std::cerr << "Error: Device 1 element " << i << " is " << host_mat_1_float[i] << std::endl;
            return 1;
        }
        if (host_mat_2_float[i] != 1.0f) {
            std::cerr << "Error: Device 2 element " << i << " is " << host_mat_2_float[i] << std::endl;
            return 1;
        }
    }
    printf("Results are correct!\n");

    // Cleanup and exit
    delete[] dev_mats;
    delete[] dev_handles;
    delete[] host_mat_1;
    delete[] host_mat_2;
    delete[] host_mat_1_float;
    delete[] host_mat_2_float;

    std::cout << "Done!" << std::endl;
    return 0;
}