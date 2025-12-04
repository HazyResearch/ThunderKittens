#include "kittens.cuh"

using namespace kittens;

static constexpr int CLUSTER_SIZE = 2;
static constexpr int NUM_CONSUMERS = 1;
static constexpr int NUM_PRODUCERS = 1;
static constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
static constexpr int NUM_THREADS = NUM_WORKERS * WARP_THREADS;
static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

template <int _SUPER_M, int _Mb, int _Nb, int _Kb, int _SMEM_PIPE_DEPTH, int _MMA_PIPE_DEPTH, int _TMEM_PIPE_DEPTH>
struct globals {
    static constexpr int SUPER_M = _SUPER_M;

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = _Kb;
    
    static constexpr int CLUSTER_M = 2*Mb;
    static constexpr int CLUSTER_N = Nb;

    static constexpr int SMEM_PIPE_DEPTH = _SMEM_PIPE_DEPTH;
    static constexpr int MMA_PIPE_DEPTH = _MMA_PIPE_DEPTH;
    static constexpr int TMEM_PIPE_DEPTH = _TMEM_PIPE_DEPTH;

    static constexpr int NUM_D_TILES = TMEM_PIPE_DEPTH > 1 ? 2 : 1;

    using a_tile = st_bf<Mb, Kb>;
    using b_tile = st_bf<Nb/2, Kb>;
    using d_tile = st_bf<Mb, Nb/TMEM_PIPE_DEPTH>;
    
    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;

    __host__ __inline__ dim3 grid() { return dim3(148); }
    __host__ __inline__ dim3 block() { return dim3(NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename G>
__cluster_dims__(CLUSTER_SIZE, 1, 1) __launch_bounds__(NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ G g) {
    const int cta_rank = cluster_ctarank();
    const int iters_per_task = g.a.cols() / G::Kb;

    auto get_task_idx = [&](int task_iter) -> int2 {
        int task_id = task_iter * (gridDim.x/CLUSTER_SIZE) + blockIdx.x/CLUSTER_SIZE;
        int Rblocks = g.d.rows() / G::CLUSTER_M, Cblocks = g.d.cols() / G::CLUSTER_N;
        int super_rows = (Rblocks/G::SUPER_M)*G::SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = G::SUPER_M*Cblocks;
        if (task_id < super_rows * Cblocks) {
            return { G::SUPER_M*(task_id/super_repeat) + task_id%G::SUPER_M, (task_id%super_repeat)/G::SUPER_M };
        }
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            return { super_rows + remainder_id%final_rows, remainder_id/final_rows };
        }
        else {
            return { -1, -1 };
        }
    };

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);

    static_assert(sizeof(G::a_tile) * G::SMEM_PIPE_DEPTH +
                  sizeof(G::b_tile) * G::SMEM_PIPE_DEPTH +
                  sizeof(G::d_tile) * G::NUM_D_TILES <= DYNAMIC_SHARED_MEMORY);
    typename G::a_tile (&a_smem)[G::SMEM_PIPE_DEPTH] = al.allocate<G::a_tile, G::SMEM_PIPE_DEPTH>();
    typename G::b_tile (&b_smem)[G::SMEM_PIPE_DEPTH] = al.allocate<G::b_tile, G::SMEM_PIPE_DEPTH>();
    typename G::d_tile (&d_smem)[G::NUM_D_TILES]     = al.allocate<G::d_tile, G::NUM_D_TILES>();

    tensor_allocator<1, 2> tm_alloc{};
    using d_tt_t = tt<float, G::Mb, G::Nb>;

    __shared__ semaphore inputs_arrived[G::SMEM_PIPE_DEPTH], inputs_finished[G::SMEM_PIPE_DEPTH], outputs_arrived, outputs_finished[G::MMA_PIPE_DEPTH];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 0) { 
        #pragma unroll
        for (int i = 0; i < G::SMEM_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < G::MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_finished[i], 0, CLUSTER_SIZE);
        }
    }
    everyone::tma::cluster::sync();

    if (warpgroup::groupid() == NUM_CONSUMERS) {
        warpgroup::increase_registers<256>();
        if (warp::laneid() == 0 && warpgroup::warpid() == 3) {
            int input_ring = 0;
            for (int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx(task_iter);
                if (rowcol.x == -1) {
                    for(int idx = 0; idx < (G::SMEM_PIPE_DEPTH); idx++) {
                        tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                        input_ring=ring_advance<G::SMEM_PIPE_DEPTH>(input_ring);
                    }
                    arrive(outputs_arrived);
                    break;
                }
                for (int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    update_phasebit<1>(bitfield, input_ring);
                    if(task_iter>0 && idx==G::SMEM_PIPE_DEPTH-1 && laneid() == 0) arrive(outputs_arrived); // TODO REVIEW 
                    tma::cluster::load_async(a_smem[input_ring], g.a, {rowcol.x*2+cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], g.b, {rowcol.y*2+cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    input_ring=ring_advance<G::SMEM_PIPE_DEPTH>(input_ring);
                }
            }
        }
        else if (cta_rank == 0 && warp::laneid() == 0 && warpgroup::warpid() == 0) {
            d_tt_t d_tt[G::MMA_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < G::MMA_PIPE_DEPTH; i++) {
                if constexpr(G::Mb == 128)
                    d_tt[i] = tm_alloc.allocate<d_tt_t>(i*G::Nb);
                else
                    d_tt[i] = tm_alloc.allocate<d_tt_t>(0, i*G::Nb);
            }
            int input_ring = 0;
            for(int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx(task_iter);
                if(rowcol.x == -1) break;
                tma::cluster::wait(outputs_finished[task_iter%G::MMA_PIPE_DEPTH], ((task_iter+G::MMA_PIPE_DEPTH)/G::MMA_PIPE_DEPTH)%2);
                tma::cluster::expect_bytes(inputs_arrived[input_ring], 2*sizeof(G::a_tile) + 2*sizeof(G::b_tile));
                tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                update_phasebit<0>(bitfield, input_ring);
                mm2_ABt(d_tt[task_iter%G::MMA_PIPE_DEPTH], a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                input_ring=ring_advance<G::SMEM_PIPE_DEPTH>(input_ring);
                for(int idx = 1; idx < iters_per_task; idx++) {
                    tma::cluster::expect_bytes(inputs_arrived[input_ring], 2*sizeof(G::a_tile) + 2*sizeof(G::b_tile));
                    tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    update_phasebit<0>(bitfield, input_ring);
                    mma2_ABt(d_tt[task_iter%G::MMA_PIPE_DEPTH], a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                    input_ring=ring_advance<G::SMEM_PIPE_DEPTH>(input_ring);
                }
            }
        }
    }
    else {
        warpgroup::increase_registers<256>();
        d_tt_t d_tt[G::MMA_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < G::MMA_PIPE_DEPTH; i++) {
            if constexpr(G::Mb == 128)
                d_tt[i] = tm_alloc.allocate<d_tt_t>(i*G::Nb);
            else
                d_tt[i] = tm_alloc.allocate<d_tt_t>(0, i*G::Nb);
        }
        for(int task_iter = 0; true; task_iter++) {
            int2 rowcol = get_task_idx(task_iter);
            if(rowcol.x == -1) break;
            wait(outputs_arrived, task_iter%2);
            rt_bf<G::Mb/4, G::Nb/G::TMEM_PIPE_DEPTH> d_reg[G::TMEM_PIPE_DEPTH];
            #pragma unroll
            for(int i = 0; i < G::TMEM_PIPE_DEPTH; i++) {
                warpgroup::load_async(d_reg[i], d_tt[task_iter%G::MMA_PIPE_DEPTH].template subtile<tt<float, G::Mb, G::Nb/G::TMEM_PIPE_DEPTH>>(0, G::Nb/G::TMEM_PIPE_DEPTH*i));
                tensor_load_wait();
                warpgroup::tma::store_async_read_wait<1>();
                warpgroup::sync(1);
                warpgroup::store(d_smem[i%2], d_reg[i]);
                warpgroup::sync(1);
                warpgroup::tma::store_async(g.d, d_smem[i%2], {2*rowcol.x+cta_rank, G::TMEM_PIPE_DEPTH*rowcol.y+i});
            }
            warpgroup::tma::store_async_read_wait();
            warpgroup::tma::cluster::arrive(outputs_finished[task_iter%G::MMA_PIPE_DEPTH], 0);
        }
    }
}

#include <random>
#include <omp.h>

template <typename G>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool check_correctness = false, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: SUPER_M=" << G::SUPER_M << " Mb=" << G::Mb << " Nb=" << G::Nb << " Kb=" << G::Kb << 
                 " SMEM_PIPE_DEPTH=" << G::SMEM_PIPE_DEPTH << " MMA_PIPE_DEPTH=" << G::MMA_PIPE_DEPTH << " TMEM_PIPE_DEPTH=" << G::TMEM_PIPE_DEPTH << "\n";
    std::cout << "Total number of tasks: " << (M / G::Mb * N / G::Nb) << "\n";
    std::cout << "Number of iterations per task: " << (K / G::Kb) << "\n";

    // Sleep for 50 ms to limit power consumption and thermals
    usleep(50000);

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
    if (check_correctness) {
        #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += h_A[i * K + k] * h_B[j * K + k];
                }
                h_C_ref[i * N + j] = sum;
            }
        }
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

    // Prepare kernel inputs
    typename G::a_gl Ag{d_A, nullptr, nullptr, M, K};
    typename G::b_gl Bg{d_B, nullptr, nullptr, N, K};
    typename G::d_gl Dg{d_C, nullptr, nullptr, M, N};
    G g{Ag, Bg, Dg};

    // Set kernel attributes
    cudaFuncSetAttribute(kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory());

    // Number of iterations
    int num_warmups = ncu ? 0 : 500;
    int num_iters = ncu ? 1 : 100;

    // Warmup
    for(int i = 0; i < num_warmups; i++)
        kernel<G><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(start));
    for(int i = 0; i < num_iters; i++)
        kernel<G><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double useconds = milliseconds * 1000.0 / num_iters;
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;
    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    if (check_correctness) {
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
            if(error > .5f) { // large because of bf16 vs fp32 numerics
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

    return tflops;
}

__host__ int main() {
    int N;
    bool check_correctness = false;
    bool ncu = false;

    // Template parameters: SUPER_M, Mb, Nb, Kb, SMEM_PIPE_DEPTH, MMA_PIPE_DEPTH, TMEM_PIPE_DEPTH
    N = 1024;
    run_benchmark<globals<4, 64, 128, 128, 4, 2, 2>>(N, N, N, check_correctness, ncu);\
    N = 2048;
    run_benchmark<globals<4, 128, 256, 64, 4, 2, 8>>(N, N, N, check_correctness, ncu);
    N = 4096;
    run_benchmark<globals<4, 128, 256, 64, 5, 2, 2>>(N, N, N, check_correctness, ncu);
    N = 8192;
    run_benchmark<globals<8, 128, 256, 64, 6, 2, 8>>(N, N, N, check_correctness, ncu);
    N = 16384;
    run_benchmark<globals<8, 128, 256, 64, 4, 2, 8>>(N, N, N, check_correctness, ncu);

    return 0;
}
