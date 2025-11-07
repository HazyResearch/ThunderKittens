#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

static constexpr int SM_COUNT = 132;
static constexpr int NUM_MAIN_THREADS = 384;
static constexpr int NUM_EPILOGUE_THREADS = 256;
static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - 1024;

struct globals {
    // Multi-GPU parameters
    static constexpr int NUM_DEVICES = 8;

    // Model parameters
    static constexpr int H = 7168;
    static constexpr int I = 2048;
    static constexpr int TOP_K = 8;
    static constexpr int NUM_EXPERTS = 256;
    static constexpr int NUM_EXPERTS_PER_DEV = NUM_EXPERTS / NUM_DEVICES;

    // GEMM parameters
    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    // Dispatch tiles
    using token_vec = sv_bf<H>;
    static constexpr int TOKENS_PER_BLOCK = 16;

    // GEMM tiles
    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>; // warpgroup distributed
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>; // warpgroup distributed

    // Input/output tensors
    using pre_tokens_pgl = pgl<gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>; // local tokens before dispatch
    using post_tokens_gl = gl<bf16, 1, 1, -1, H, token_vec, A_tile>; // local tokens after dispatch
    using weights_gl = gl<bf16, 1, NUM_EXPERTS_PER_DEV, H, I, B_tile>;
    using outputs_gl = gl<bf16, 1, 1, -1, I, C_tile>;
    using padded_tokens_per_expert_gl = gl<int, 1, 1, 1, NUM_EXPERTS>;
    using pull_dispatch_indices_gl = gl<int, 1, 1, -1, 2>;
    using barrier_pgl = device<NUM_DEVICES>::barrier_t;

    pre_tokens_pgl pre_tokens;
    post_tokens_gl post_tokens;
    weights_gl weights;
    outputs_gl outputs;
    padded_tokens_per_expert_gl padded_tokens_per_expert;
    pull_dispatch_indices_gl pull_dispatch_indices;
    barrier_pgl barrier;

    const int dev_idx;
    const int num_padded_local_tokens;
    const int num_comm_sms;
    const int num_comp_sms;

    struct pipeline_inputs {
        A_tile A[2];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C[2];
    };
};

__device__ inline void dispatch(const globals &G, const int sm_idx) {
    // Declare and allocate shared memory
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] = al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];

    // Start the dispatch
    const int lane_id = threadIdx.x;
    if (lane_id < globals::TOKENS_PER_BLOCK) {
        const int token_idx = sm_idx * globals::TOKENS_PER_BLOCK + lane_id;

        if (token_idx < G.num_padded_local_tokens) {
            int src_dev_idx = G.pull_dispatch_indices[{token_idx, 0}];
            int src_token_idx = G.pull_dispatch_indices[{token_idx, 1}];

            if (src_dev_idx >= 0 && src_token_idx >= 0) {
                init_semaphore(token_arrived[lane_id], 0, 1);
                tma::expect_bytes(token_arrived[lane_id], sizeof(globals::token_vec));
                tma::load_async(token[lane_id], G.pre_tokens[src_dev_idx], {src_token_idx, 0}, token_arrived[lane_id]);
    
                wait(token_arrived[lane_id], 0);
                tma::store_async(G.post_tokens, token[lane_id], {token_idx, 0});
                tma::store_async_wait();
            }

            asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&G.barrier[G.dev_idx][{token_idx / globals::ROW_BLOCK}]), "r"(1) : "memory");
        }
    }
}

__device__ inline void group_gemm(const globals &G, const int sm_idx, const int num_sms) {
    // Declare shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    // Allocate shared memory
    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES] = allocator.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
    globals::pipeline_outputs &outputs = *reinterpret_cast<globals::pipeline_outputs *>(&inputs[globals::PIPELINE_STAGES - 1]);
    
    // Load padded_tokens_per_expert
    const int expert_offset = G.dev_idx * globals::NUM_EXPERTS_PER_DEV;
    __shared__ int padded_tokens_per_expert[globals::NUM_EXPERTS_PER_DEV];
    if (threadIdx.x < globals::NUM_EXPERTS_PER_DEV)
        padded_tokens_per_expert[threadIdx.x] = G.padded_tokens_per_expert[{expert_offset + static_cast<int>(threadIdx.x)}];

    // Set up mbarriers
    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();

    // Common variables
    const int warpgroup_id = warpgroup::groupid();
    const int warp_id = warpgroup::warpid();
    const int lane_id = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    constexpr int num_iters = globals::H / globals::RED_BLOCK;
    constexpr int col_blocks = globals::I / globals::COL_BLOCK;

    // Main divergence
    if (warpgroup_id == 2) {
        warpgroup::decrease_registers<40>();

        if (warp_id == 0 && lane_id == 0) {
            #pragma unroll
            for (int task_id = sm_idx, num_tokens_per_expert_cum = 0, expert_id = 0; expert_id < globals::NUM_EXPERTS_PER_DEV; expert_id++) {
                const int row_block_start = num_tokens_per_expert_cum / globals::ROW_BLOCK; // inclusive
                num_tokens_per_expert_cum += padded_tokens_per_expert[expert_id];
                const int row_block_end = (num_tokens_per_expert_cum + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK; // exclusive
                const int row_blocks = row_block_end - row_block_start;
                const int num_blocks = row_blocks * col_blocks;

                for (; task_id < num_blocks; task_id += num_sms) {
                    const int row_idx = task_id / col_blocks + row_block_start;
                    const int col_idx = task_id % col_blocks;

                    int bar_val;
                    asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(bar_val) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
                    while (bar_val != globals::ROW_BLOCK) {
                        __nanosleep(16);
                        asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(bar_val) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
                    }

                    for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        update_phasebit<1>(phasebits, stage);
                        tma::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs));
                        if (red_idx == globals::PIPELINE_STAGES - 1) { // assumption: K is always a multiple of 4*64
                            wait(outputs_finished, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                            update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);
                        }
                        #pragma unroll
                        for (int i = 0; i < 2; i++)
                            tma::load_async(inputs[stage].A[i], G.post_tokens, {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                        tma::load_async(inputs[stage].B, G.weights, {expert_id, red_idx, col_idx}, inputs_arrived[stage]);
                        stage = (stage + 1) % globals::PIPELINE_STAGES;
                    }
                }
                task_id -= num_blocks;
            }
        } else if (warp_id == 1 && lane_id == 0) {
            #pragma unroll
            for (int task_id = sm_idx, num_tokens_per_expert_cum = 0, expert_id = 0; expert_id < globals::NUM_EXPERTS_PER_DEV; expert_id++) {
                const int row_block_start = num_tokens_per_expert_cum / globals::ROW_BLOCK; // inclusive
                num_tokens_per_expert_cum += padded_tokens_per_expert[expert_id];
                const int row_block_end = (num_tokens_per_expert_cum + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK; // exclusive
                const int row_blocks = row_block_end - row_block_start;
                const int num_blocks = row_blocks * col_blocks;

                for (; task_id < num_blocks; task_id += num_sms) {
                    const int row_idx = task_id / col_blocks + row_block_start;
                    const int col_idx = task_id % col_blocks;

                    wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                    update_phasebit<0>(phasebits, 0);
                    #pragma unroll
                    for (int i = 0; i < 2; i++)
                        tma::store_async(G.outputs, outputs.C[i], {row_idx * 2 + i, col_idx});
                    tma::store_async_read_wait();
                    arrive(outputs_finished);
                }
                task_id -= num_blocks;
            }
        }
    } else {
        warpgroup::increase_registers<232>();

        #pragma unroll
        for (int task_id = sm_idx, num_tokens_per_expert_cum = 0, expert_id = 0; expert_id < globals::NUM_EXPERTS_PER_DEV; expert_id++) {
            const int row_block_start = num_tokens_per_expert_cum / globals::ROW_BLOCK; // inclusive
            num_tokens_per_expert_cum += padded_tokens_per_expert[expert_id];
            const int row_block_end = (num_tokens_per_expert_cum + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK; // exclusive
            const int row_blocks = row_block_end - row_block_start;
            const int num_blocks = row_blocks * col_blocks;

            for (; task_id < num_blocks; task_id += num_sms) {
                rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK> C_accum;
                warp::zero(C_accum);

                for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                    wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    warpgroup::mma_AB(C_accum, inputs[stage].A[warpgroup_id], inputs[stage].B);
                    warpgroup::mma_async_wait();
                    warp::arrive(inputs_finished[stage]);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }

                group<8>::sync(3);
                warpgroup::store(outputs.C[warpgroup_id], C_accum);
                warpgroup::sync(warpgroup_id + 1);
                warpgroup::arrive(outputs_arrived);
            }
            task_id -= num_blocks;
        }
    }
}

__global__ __launch_bounds__(NUM_MAIN_THREADS, 1)
void dispatch_group_gemm_kernel(const __grid_constant__ globals G) {
    if (blockIdx.x < G.num_comp_sms)
        group_gemm(G, blockIdx.x, G.num_comp_sms);
    else
        dispatch(G, blockIdx.x - G.num_comp_sms);
}

__global__ __launch_bounds__(NUM_EPILOGUE_THREADS)
void epilogue_kernel(const __grid_constant__ globals G) {
    // Reset the barrier
    const int num_blocks = (G.num_padded_local_tokens + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = offset; i < num_blocks; i += stride)
        G.barrier[G.dev_idx][{i}] = 0;
}

void entrypoint(
    kittens::py::TKParallelTensor &pre_tokens,
    at::Tensor &post_tokens,
    at::Tensor &weights,
    at::Tensor &outputs,
    at::Tensor &padded_tokens_per_expert,
    at::Tensor &pull_dispatch_indices,
    kittens::py::TKParallelTensor &barrier,
    const int num_comm_sms,
    const int num_padded_local_tokens
) {
    globals G {
        .pre_tokens=kittens::py::parallel_tensor_to_pgl<globals::pre_tokens_pgl>(pre_tokens),
        .post_tokens=kittens::py::tensor_to_gl<globals::post_tokens_gl>(post_tokens),
        .weights=kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs=kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert=kittens::py::tensor_to_gl<globals::padded_tokens_per_expert_gl>(padded_tokens_per_expert),
        .pull_dispatch_indices=kittens::py::tensor_to_gl<globals::pull_dispatch_indices_gl>(pull_dispatch_indices),
        .barrier=kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = barrier.local_rank_,
        .num_padded_local_tokens = num_padded_local_tokens,
        .num_comm_sms = num_comm_sms,
        .num_comp_sms = SM_COUNT - num_comm_sms
    };

    cudaStream_t current_stream = at::cuda::getCurrentCUDAStream();

    const int dispatch_group_gemm_blocks = (G.num_padded_local_tokens + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK + G.num_comp_sms;
    cudaFuncSetAttribute(dispatch_group_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SHARED_MEMORY);
    dispatch_group_gemm_kernel<<<dispatch_group_gemm_blocks, NUM_MAIN_THREADS, DYNAMIC_SHARED_MEMORY, current_stream>>>(G);

    const int epilogue_blocks = ((G.num_padded_local_tokens + globals::ROW_BLOCK - 1) / globals::ROW_BLOCK + NUM_EPILOGUE_THREADS - 1) / NUM_EPILOGUE_THREADS;
    epilogue_kernel<<<epilogue_blocks, NUM_EPILOGUE_THREADS, 0, current_stream>>>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("moe_dispatch_gemm", &entrypoint);
}
