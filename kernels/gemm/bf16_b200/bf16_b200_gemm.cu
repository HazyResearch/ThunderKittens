#include "kittens.cuh"
#include <iostream>

using namespace kittens;

constexpr int NUM_CONSUMERS = 1;
constexpr int NUM_PRODUCERS = 1;
constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
constexpr int NUM_THREADS = NUM_WORKERS * WARP_THREADS;

static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 64;

constexpr int SMEM_PIPE_DEPTH = 5;
constexpr int MMA_PIPE_DEPTH = 2;
constexpr int TMEM_PIPE_DEPTH = 8;
constexpr int CLC_PIPE_DEPTH = 2;

constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

constexpr int CLUSTER_SIZE = 4;

struct matmul_globals {
    using a_tile = st_bf<Mb, Kb, true, Kb/2*sizeof(bf16)>;
    using a_subtile = st_bf<Mb, Kb/2, true, Kb/2*sizeof(bf16)>;
    using b_tile = st_bf<Nb/2, Kb>;
    using b_subtile = st_bf<Nb/2, Kb>; // TODO
    using d_tile = st_bf<Mb, Nb/TMEM_PIPE_DEPTH>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_subtile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_subtile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;

    __host__ __inline__ dim3 grid() {
        return dim3(d.rows()/Mb * d.cols()/Nb);
    }
};

template<int SUPER_M=4> __device__ static inline int get_task_idx(const matmul_globals &g, int block_idx) {
    constexpr int CLUSTER_M = 2*Mb, CLUSTER_N = 2*Nb; // depends on cluster grid layout
    int cluster_idx = block_idx/CLUSTER_SIZE;
    int Rblocks = g.d.rows() / CLUSTER_M, Cblocks = g.d.cols() / CLUSTER_N;
    int super_rows = (Rblocks/SUPER_M)*SUPER_M,
        final_rows = Rblocks - super_rows,
        super_repeat = SUPER_M*Cblocks;
    if (cluster_idx < super_rows * Cblocks) {
        return ((SUPER_M*(cluster_idx/super_repeat) + cluster_idx%SUPER_M) << 16) | (cluster_idx%super_repeat)/SUPER_M;
    } else {
        int remainder_id = cluster_idx - super_rows*Cblocks;
        return ((super_rows + remainder_id%final_rows) << 16) | (remainder_id/final_rows);
    }
}

__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1) __launch_bounds__(NUM_THREADS, 1)
void matmul(const __grid_constant__ matmul_globals g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int warpid = kittens::warpid();
    const int warpgroupid = warpgroup::groupid();
    const int cta_rank = cluster_ctarank();
    const int iters_per_task = g.a.cols() / Kb;

    using a_tile = matmul_globals::a_tile;
    using b_tile = matmul_globals::b_tile;
    using d_tile = matmul_globals::d_tile;
    
    static_assert(sizeof(a_tile) * SMEM_PIPE_DEPTH +
                  sizeof(b_tile) * SMEM_PIPE_DEPTH +
                  sizeof(d_tile) * 2 <= DYNAMIC_SHARED_MEMORY);
    a_tile (&a_smem)[SMEM_PIPE_DEPTH] = al.allocate<a_tile, SMEM_PIPE_DEPTH>();
    b_tile (&b_smem)[SMEM_PIPE_DEPTH] = al.allocate<b_tile, SMEM_PIPE_DEPTH>();
    d_tile (&d_smem)[2]               = al.allocate<d_tile, 2>();

    everyone::tma::cluster::sync();
    tensor_allocator<1, 2> tm_alloc{};
    using d_tt_t = tt<float, Mb, Nb>;

    __shared__ int next_tile_idx[CLC_PIPE_DEPTH];
    __shared__ clc::handle clc_handle;
    __shared__ semaphore clc_arrived, job_arrived[CLC_PIPE_DEPTH], job_finished[CLC_PIPE_DEPTH];
    __shared__ semaphore inputs_arrived[SMEM_PIPE_DEPTH], inputs_finished[SMEM_PIPE_DEPTH], outputs_arrived, outputs_finished[MMA_PIPE_DEPTH];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 0) { 
        #pragma unroll
        for (int i = 0; i < SMEM_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, CLUSTER_SIZE);
            init_semaphore(inputs_finished[i], 0, CLUSTER_SIZE/2); // even CTA only
        }
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_finished[i], 0, 2); // CTA pair
        }
        init_semaphore(clc_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < CLC_PIPE_DEPTH; i++) {
            init_semaphore(job_arrived[i], 0, 1);
            init_semaphore(job_finished[i], 0, 1);
        }
    }
    everyone::tma::cluster::sync();

    if(warpgroupid == NUM_CONSUMERS) {
        warpgroup::increase_registers<256>();
        if (warp::laneid() == 0 && warpgroup::warpid() == 0) {
            const uint16_t a_mask = 0b101<<(cta_rank&0b1); // depends on cluster grid layout
            const uint16_t b_mask = 1<<cta_rank; // depends on cluster grid layout
            const uint16_t sem_mask = cta_rank&(~1); // to even CTA
            int input_ring = 0; // tracking which input block is being loaded
            for (int task_iter = 0; true; task_iter++) {
                wait(job_arrived[task_iter%CLC_PIPE_DEPTH], (task_iter/CLC_PIPE_DEPTH)%2);
                int rowcol_packed = next_tile_idx[task_iter%CLC_PIPE_DEPTH];
                if(rowcol_packed == -1) {
                    for (int idx = 0; idx < (SMEM_PIPE_DEPTH); idx++) {
                        tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                        input_ring=ring_advance<SMEM_PIPE_DEPTH>(input_ring);
                    }
                    arrive(outputs_arrived);
                    break;
                }
                int2 rowcol = {rowcol_packed >> 16, rowcol_packed & 0xFFFF};
                for (int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    update_phasebit<1>(bitfield, input_ring);
                    if(task_iter>0 && idx==SMEM_PIPE_DEPTH-1) arrive(outputs_arrived);
                    auto &a_subtile = a_smem[input_ring].subtile<Kb/2>(cta_rank>>1);
                    auto &b_subtile = b_smem[input_ring];
                    tma::cluster::expect_bytes(inputs_arrived[input_ring], sizeof(a_subtile) + sizeof(b_subtile), sem_mask);
                    tma::cluster::expect_bytes(inputs_arrived[input_ring], sizeof(a_subtile), sem_mask ^ (0b10));
                    tma::cluster::load_async(a_subtile, g.a, {rowcol.x*2+(cta_rank&1), idx*2+(cta_rank>>1)}, inputs_arrived[input_ring], a_mask, sem_mask);
                    tma::cluster::load_async(b_subtile, g.b, {rowcol.y*4+cta_rank,     idx}                , inputs_arrived[input_ring], b_mask, sem_mask);
                    input_ring=ring_advance<SMEM_PIPE_DEPTH>(input_ring);
                }
            }
        }
        else if((cta_rank&1) == 0 && warp::laneid() == 0 && warpgroup::warpid() == 1) { // launch the MMA's
            d_tt_t d_tt[2] = {tm_alloc.allocate<d_tt_t>(0*Nb), tm_alloc.allocate<d_tt_t>(1*Nb)};
            int input_ring = 0; // tracking which input block is being loaded
            constexpr uint16_t cta_mask = (1<<CLUSTER_SIZE) - 1;
            for(int task_iter = 0; true; task_iter++) {
                wait(job_arrived[task_iter%CLC_PIPE_DEPTH], (task_iter/CLC_PIPE_DEPTH)%2);
                if(next_tile_idx[task_iter%CLC_PIPE_DEPTH] == -1) break;
                tma::cluster::wait(outputs_finished[task_iter%MMA_PIPE_DEPTH], (task_iter/MMA_PIPE_DEPTH+1)%2); // make sure tensor memory is ready to be written to.
                tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                update_phasebit<0>(bitfield, input_ring);
                mm2_ABt(d_tt[task_iter%MMA_PIPE_DEPTH], a_smem[input_ring], b_smem[input_ring]);
                detail::tcgen05::commit<2>(inputs_finished[input_ring], cta_mask);
                input_ring=ring_advance<SMEM_PIPE_DEPTH>(input_ring);
                for(int idx = 1; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    update_phasebit<0>(bitfield, input_ring);
                    mma2_ABt(d_tt[task_iter%MMA_PIPE_DEPTH], a_smem[input_ring], b_smem[input_ring]);
                    detail::tcgen05::commit<2>(inputs_finished[input_ring], cta_mask);
                    input_ring=ring_advance<SMEM_PIPE_DEPTH>(input_ring);
                }
            }
        }
        else if(warp::laneid() == 0 && warpgroup::warpid() == 2) { // fetch next block idx
            update_phasebit<1>(bitfield, 0);
            next_tile_idx[0] = get_task_idx(g, blockIdx.x);
            arrive(job_arrived[0]);
            for(int task_iter = 1; true; task_iter++) {
                tma::cluster::expect_bytes(clc_arrived, sizeof(clc_handle), cta_rank);
                if (cta_rank == 0)
                    clc::schedule(clc_handle, clc_arrived);
                tma::cluster::wait(clc_arrived, get_phasebit<0>(bitfield, 0));
                update_phasebit<0>(bitfield, 0);
                clc::result clc_result;
                clc::query(clc_result, clc_handle);
                wait(job_finished[task_iter%CLC_PIPE_DEPTH], get_phasebit<1>(bitfield, task_iter%CLC_PIPE_DEPTH));
                update_phasebit<1>(bitfield, task_iter%CLC_PIPE_DEPTH);
                next_tile_idx[task_iter%CLC_PIPE_DEPTH] = clc_result.success ? get_task_idx(g, clc_result.x + cta_rank) : -1;
                arrive(job_arrived[task_iter%CLC_PIPE_DEPTH]);
                if (!clc_result.success) break;
            }
        }
    }
    else {
        warpgroup::increase_registers<256>();
        d_tt_t d_tt[2] = {tm_alloc.allocate<d_tt_t>(0*Nb), tm_alloc.allocate<d_tt_t>(1*Nb)};
        for(int task_iter = 0; true; task_iter++) {
            wait(job_arrived[task_iter%CLC_PIPE_DEPTH], (task_iter/CLC_PIPE_DEPTH)%2);
            int rowcol_packed = next_tile_idx[task_iter%CLC_PIPE_DEPTH];
            if(rowcol_packed == -1) break;
            int2 rowcol = {rowcol_packed >> 16, rowcol_packed & 0xFFFF};
            wait(outputs_arrived, task_iter%2);
            #pragma unroll
            for (int i = 0; i < TMEM_PIPE_DEPTH; i++) {
                rt_bf<Mb/WARPGROUP_WARPS, Nb/TMEM_PIPE_DEPTH> d_partial_reg;
                warpgroup::load_async(d_partial_reg, d_tt[task_iter%MMA_PIPE_DEPTH].subtile<tt<float, Mb, Nb/TMEM_PIPE_DEPTH>>(0, i*Nb/TMEM_PIPE_DEPTH));
                tensor_load_wait();
                if (warpgroup::laneid()==0) tma::store_async_read_wait<1>();
                warpgroup::sync(1);
                if (i==TMEM_PIPE_DEPTH-1 && warpgroup::laneid()==0) tma::cluster::arrive(outputs_finished[task_iter%MMA_PIPE_DEPTH], cta_rank&(~1));
                warpgroup::store(d_smem[i % 2], d_partial_reg);
                warpgroup::sync(1);
                if (warpgroup::laneid()==0) tma::store_async(g.d, d_smem[i % 2], {2*rowcol.x+(cta_rank&1), TMEM_PIPE_DEPTH*(rowcol.y*2+(cta_rank>>1))+i});
            }
            warpgroup::arrive(job_finished[task_iter%CLC_PIPE_DEPTH]);
        }
    }
}


constexpr bool NCU = false;
constexpr bool CHECK_CORRECTNESS = false;
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>

void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[j * K + k];
            }
            c[i * N + j] = sum;
        }
    }
}

void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K) {
    using globals  = matmul_globals;
    typename globals::a_gl Ag{d_A, nullptr, nullptr, M, K};
    typename globals::b_gl Bg{d_B, nullptr, nullptr, N, K};
    typename globals::d_gl Dg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Dg};
    matmul<<<G.grid(), NUM_THREADS, MAX_SHARED_MEMORY-1024>>>(G);
}

int run_benchmark(size_t M, size_t N, size_t K) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << Mb*2 << "x" << Nb << "x" << Kb << "\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    if (CHECK_CORRECTNESS) {
        cpu_gemm(h_A, h_B, h_C_ref, M, N, K);
        std::cout << "Performed CPU matrix multiplication" << std::endl;
    }

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    CUDACHECK(cudaMalloc(&d_A, M*K*sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_B, K*N*sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_C, M*N*sizeof(__nv_bfloat16)));
    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);
    CUDACHECK(cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice));
    std::cout << "Copied matrices to device" << std::endl;

    // Set kernel attributes
    cudaFuncSetAttribute(matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SHARED_MEMORY);
    cudaFuncSetAttribute(matmul, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); // enable 16-cluster

    // Warmup
    for(int i = 0; i < (NCU ? 0 : 500); i++)
        inner_run(d_A, d_B, d_C, M, N, K);

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(start));
    constexpr int ITERS = (NCU ? 1 : 100);
    for(int i = 0; i < ITERS; i++)
        inner_run(d_A, d_B, d_C, M, N, K);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double useconds = milliseconds * 1000.0 / ITERS;
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;
    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    if (CHECK_CORRECTNESS) {
        // Copy result back to host
        CUDACHECK(cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost));
        std::cout << "Copied result back to host" << std::endl;

        // Convert result back to float for comparison
        for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);
        std::cout << "Converted result back to float" << std::endl;

        // Check result
        float max_error = 0.0f;
        float average_error = 0.0f;
        int error_count = 0;
        for (int i = 0; i < M * N; ++i) {
            float error = std::abs(h_C[i] - h_C_ref[i]);
            if(error > .2f) { // large because of bf16 vs fp32 numerics
                if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
                else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
                error_count++;
            }
            max_error = std::max(max_error, error);
            average_error += error;
        }
        average_error /= M*N;

        std::cout << "Max error: " << max_error << std::endl;
        std::cout << "Average error: " << average_error << std::endl;
        std::cout << "Error count: " << error_count << std::endl;
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

int main() {
    int N;
    N = 1024;
    run_benchmark(N, N, N);
    N = 2048;
    run_benchmark(N, N, N);
    N = 4096;
    run_benchmark(N, N, N);
    N = 8192;
    run_benchmark(N, N, N);
    N = 16384;
    run_benchmark(N, N, N);
    return 0;
}
