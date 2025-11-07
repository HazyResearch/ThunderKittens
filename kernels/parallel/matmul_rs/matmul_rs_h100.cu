#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace matmul_reduce_scatter {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 132;

    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 2;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;

};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>; // warpgroup distributed
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>; // warpgroup distributed

    using A_gl = gl<bf16, 1, 1, -1, -1, A_tile>;
    using B_gl = gl<bf16, 1, 1, -1, -1, B_tile>;
    using C_pgl = pgl<gl<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, false>;

    A_gl A;
    B_gl B;
    C_pgl C;
    const int dev_idx;

    struct pipeline_inputs {
        A_tile A[2];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C[2];
    };
};

__device__ inline void kernel(const globals &G) {
    // Declare shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    // Allocate shared memory
    static_assert(sizeof(globals::pipeline_inputs) * (globals::PIPELINE_STAGES - 1) + sizeof(globals::pipeline_outputs) <= config::DYNAMIC_SHARED_MEMORY); // overlap last stage
    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES] = allocator.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
    globals::pipeline_outputs &outputs = *reinterpret_cast<globals::pipeline_outputs *>(&inputs[globals::PIPELINE_STAGES - 1]);

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
    int warpgroup_id = warpgroup::groupid();
    int warp_id = warpgroup::warpid();
    int lane_id = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    int row_blocks = G.A.rows() / globals::ROW_BLOCK;
    int col_blocks = G.B.cols() / globals::COL_BLOCK;
    int super_rows = (row_blocks / globals::SUPER_M) * globals::SUPER_M;
    int final_rows = row_blocks - super_rows;
    int super_blocks = globals::SUPER_M * col_blocks;
    int num_blocks = row_blocks * col_blocks;
    int num_iters = G.A.cols() / globals::RED_BLOCK;

    // Multi-GPU variables
    int row_blocks_per_dev = row_blocks / globals::NUM_DEVICES;
    int dev_task_offset = ((G.dev_idx + 1) * (num_blocks / globals::NUM_DEVICES)) % num_blocks;

    // Main divergence
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        if (warp_id == 0 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < num_blocks; task_id += gridDim.x) {
                // Calculate indices
                int real_task_id = (task_id + dev_task_offset) % num_blocks;
                int row_idx, col_idx;
                if (real_task_id < super_rows * col_blocks) {
                    row_idx = globals::SUPER_M * (real_task_id / super_blocks) + real_task_id % globals::SUPER_M;
                    col_idx = (real_task_id % super_blocks) / globals::SUPER_M;
                } else {
                    int remainder_id = real_task_id - super_rows * col_blocks;
                    row_idx = super_rows + remainder_id % final_rows;
                    col_idx = remainder_id / final_rows;
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
                        tma::load_async(inputs[stage].A[i], G.A, {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                    tma::load_async(inputs[stage].B, G.B, {red_idx, col_idx}, inputs_arrived[stage]);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        } else if (warp_id == 1 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < num_blocks; task_id += gridDim.x) {
                // Calculate indices
                int real_task_id = (task_id + dev_task_offset) % num_blocks;
                int row_idx, col_idx;
                if (real_task_id < super_rows * col_blocks) {
                    row_idx = globals::SUPER_M * (real_task_id / super_blocks) + real_task_id % globals::SUPER_M;
                    col_idx = (real_task_id % super_blocks) / globals::SUPER_M;
                } else {
                    int remainder_id = real_task_id - super_rows * col_blocks;
                    row_idx = super_rows + remainder_id % final_rows;
                    col_idx = remainder_id / final_rows;
                }

                int dev_idx = row_idx / row_blocks_per_dev;
                row_idx %= row_blocks_per_dev;

                wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                update_phasebit<0>(phasebits, 0);
                #pragma unroll
                for (int i = 0; i < 2; i++)
                    tma::store_add_async(G.C[dev_idx], outputs.C[i], {row_idx * 2 + i, col_idx});
                tma::store_async_read_wait();
                arrive(outputs_finished);
            }
        }
    } else {
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        for (int task_id = blockIdx.x; task_id < num_blocks; task_id += gridDim.x) {
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
    }
}

} // namespace matmul_reduce_scatter

namespace matmul_reduce_scatter_barrier {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    device<NUM_DEVICES>::barrier_t barrier;
    const int dev_idx;
};

__device__ inline void kernel(const globals &G) {
    device<globals::NUM_DEVICES>::barrier(G.barrier, {0}, G.dev_idx);
}

}

void entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &barrier
) {
    matmul_reduce_scatter::globals matmul_reduce_scatter_G {
        .A = kittens::py::tensor_to_gl<matmul_reduce_scatter::globals::A_gl>(A),
        .B = kittens::py::tensor_to_gl<matmul_reduce_scatter::globals::B_gl>(B),
        .C = kittens::py::parallel_tensor_to_pgl<matmul_reduce_scatter::globals::C_pgl>(C),
        .dev_idx = barrier.local_rank_
    };

    matmul_reduce_scatter_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<device<matmul_reduce_scatter_barrier::globals::NUM_DEVICES>::barrier_t>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<matmul_reduce_scatter::config, matmul_reduce_scatter::globals, matmul_reduce_scatter::kernel>(matmul_reduce_scatter_G);
    kittens::py::launch_kernel<matmul_reduce_scatter_barrier::config, matmul_reduce_scatter_barrier::globals, matmul_reduce_scatter_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("matmul_reduce_scatter", &entrypoint);
}
