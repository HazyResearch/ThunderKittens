#include <kittens.cuh>
#include <prototype.cuh>
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace tp_matmul {

// Kernel configuration
struct config {
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 2;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 56;
    static constexpr int CONSUMER_REGISTERS = 224;

    static constexpr int PIPELINE_STAGES = 4;
};

// Kernel globals
struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int SUPERGROUP_BLOCKS = 8;
    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 256;
    static constexpr int REDUCTION_BLOCK = 64;

    using A_tile = st_bf<ROW_BLOCK / 2, REDUCTION_BLOCK>; // cluster distributed
    using B_tile = st_bf<COL_BLOCK / 2, REDUCTION_BLOCK>; // cluster distributed
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;       // cluster distributed

    using A_gl = gl<bf16, 1, 1, -1, -1, A_tile>;
    using B_gl = gl<bf16, 1, 1, -1, -1, B_tile>;
    using C_pgl = pgl<gl<bf16, 1, 1, -1, -1>, NUM_DEVICES, true, C_tile>;

    A_gl A;
    B_gl B;
    C_pgl C;
    const int dev_idx;

    struct pipeline_inputs {
        A_tile A;
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };
};

// Kernel implementation
__device__ inline void kernel(const globals &G) {
    // Shared memory declaration
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    // Warpgroup configuration
    using consumer = group<config::CONSUMER_WARPGROUPS * WARPGROUP_WARPS>;
    int warpgroup_id = warpgroup::groupid();
    int warp_id = warpgroup::warpid();
    int lane_id = warp::laneid();

    // Allocate shared and tensor memory
    static_assert(sizeof(globals::pipeline_inputs) * config::PIPELINE_STAGES + sizeof(globals::pipeline_outputs) <= config::DYNAMIC_SHARED_MEMORY);
    globals::pipeline_inputs (&inputs)[config::PIPELINE_STAGES] = allocator.allocate<globals::pipeline_inputs, config::PIPELINE_STAGES>();
    globals::pipeline_outputs &outputs = allocator.allocate<globals::pipeline_outputs>();
    tensor_allocator<1, 2> tm_allocator {};

    // Set up mbarriers
    __shared__ semaphore inputs_arrived[config::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[config::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore tensors_finished;
    __shared__ semaphore C_arrived;
    __shared__ semaphore C_finished;
    if (threadIdx.x == 0) {
        for (int i = 0; i < config::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 2);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(tensors_finished, 0, 2);
        init_semaphore(C_arrived, 0, 1);
        init_semaphore(C_finished, 0, 1);
    }
    everyone::tma::cluster::sync();

    // Pipeline configuration
    int num_blocks_per_row = G.C.cols() / globals::COL_BLOCK;
    int num_blocks_per_col = G.C.rows() / globals::ROW_BLOCK;
    int num_blocks = num_blocks_per_row * num_blocks_per_col;
    int num_iters_per_block = G.A.cols() / globals::REDUCTION_BLOCK;
    int num_blocks_per_supergroup = globals::SUPERGROUP_BLOCKS * num_blocks_per_row;

    // Declare stage and phasebits for semaphore waits
    int stage = 0;
    int last_stage = -1;
    uint32_t phasebits = 0xFFFF0000;

    // Main divergence
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        // Producer group
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();
        int ctarank = cluster_ctarank();

        // Sub divergence
        if (warp_id == 3 && lane_id == 0) {
            // Loader
            for (int task_idx = clusterIdx().x; task_idx < num_blocks; task_idx += gridDim.x / config::CLUSTER_SIZE) {
                // Compute block indices
                int block_idx = (task_idx + G.dev_idx * (num_blocks / globals::NUM_DEVICES)) % num_blocks;
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(globals::SUPERGROUP_BLOCKS, num_blocks_per_col - supergroup_idx * globals::SUPERGROUP_BLOCKS);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * globals::SUPERGROUP_BLOCKS + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                for (int i = 0; i < num_iters_per_block; ++i) {
                    tma::cluster::wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    if (stage == last_stage) {
                        arrive(outputs_arrived);
                        last_stage = -1;
                    }
                    tma::cluster::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs), 0);
                    tma::cluster::load_async(inputs[stage].A, G.A, {row_block_idx * 2 + ctarank, i}, inputs_arrived[stage], (uint16_t)(1 << ctarank), 0);
                    tma::cluster::load_async(inputs[stage].B, G.B, {col_block_idx * 2 + ctarank, i}, inputs_arrived[stage], (uint16_t)(1 << ctarank), 0);
                    update_phasebit<1>(phasebits, stage);
                    if (i == num_iters_per_block - 1) {
                        last_stage = stage;
                    }
                    stage = (stage + 1) % config::PIPELINE_STAGES;
                }
            }
            if (clusterIdx().x < num_blocks) {
                tma::cluster::wait(inputs_finished[last_stage], get_phasebit<1>(phasebits, last_stage));
                arrive(outputs_arrived);
            }
        } else if (lane_id == 0 && warp_id == 1) {
            // Storer
            for (int task_idx = clusterIdx().x; task_idx < num_blocks; task_idx += gridDim.x / config::CLUSTER_SIZE) {
                // Compute block indices
                int block_idx = (task_idx + G.dev_idx * (num_blocks / globals::NUM_DEVICES)) % num_blocks;
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(globals::SUPERGROUP_BLOCKS, num_blocks_per_col - supergroup_idx * globals::SUPERGROUP_BLOCKS);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * globals::SUPERGROUP_BLOCKS + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                wait(C_arrived, get_phasebit<0>(phasebits, 0));
                update_phasebit<0>(phasebits, 0);

                tma::store_add_async(G.C, outputs.C, {row_block_idx * 2 + ctarank, col_block_idx});
                tma::store_async_read_wait();
                arrive(C_finished);
            }
        } else if (lane_id == 0 && ctarank == 0 && warp_id == 0) {
            // Launcher
            auto tm = tm_allocator.allocate<tt<float, globals::ROW_BLOCK / 2, globals::COL_BLOCK>>(0);
            for (int task_idx = clusterIdx().x; task_idx < num_blocks; task_idx += gridDim.x / config::CLUSTER_SIZE) {
                tma::cluster::wait(tensors_finished, get_phasebit<1>(phasebits, config::PIPELINE_STAGES));
                update_phasebit<1>(phasebits, config::PIPELINE_STAGES);
                {
                    tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    mm2_ABt(tm, inputs[stage].A, inputs[stage].B, inputs_finished[stage]);
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % config::PIPELINE_STAGES;
                }
                for (int i = 1; i < num_iters_per_block; ++i) {
                    tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    mma2_ABt(tm, inputs[stage].A, inputs[stage].B, inputs_finished[stage]);
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % config::PIPELINE_STAGES;
                }
            }
        }
    } else {
        // Consumer group
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
        int ctarank = cluster_ctarank();
        auto tm = tm_allocator.allocate<tt<float, globals::ROW_BLOCK / 2, globals::COL_BLOCK>>(0);

        for (int task_idx = clusterIdx().x; task_idx < num_blocks; task_idx += gridDim.x / config::CLUSTER_SIZE) {
            // Wait for the last matmul to complete
            wait(outputs_arrived, get_phasebit<0>(phasebits, config::PIPELINE_STAGES));
            update_phasebit<0>(phasebits, config::PIPELINE_STAGES);

            // Load the output from tensor memory into registers
            rt_bf<globals::ROW_BLOCK / 16, globals::COL_BLOCK> C;
            consumer::load_async(C, tm);
            tensor_load_wait();
            consumer::sync(1);
            if (consumer::laneid() == 0)
                tma::cluster::arrive(tensors_finished, 0);

            // Store to shared memory
            wait(C_finished, get_phasebit<1>(phasebits, 0));
            update_phasebit<1>(phasebits, 0);
            consumer::store(outputs.C, C);
            consumer::sync(1);
            if (consumer::laneid() == 0)
                arrive(C_arrived);
        }
    }
}

} // namespace tp_matmul

namespace tp_matmul_barrier {

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
    device<globals::NUM_DEVICES>::sync_on_exit(G.barrier, G.dev_idx);
}

} // namespace tp_matmul_barrier

void entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::device_check(A, B);
    kittens::py::parallel_tensor_check(C, barrier);

    tp_matmul::globals tp_matmul_G {
        .A = kittens::py::tensor_to_gl<typename tp_matmul::globals::A_gl>(A),
        .B = kittens::py::tensor_to_gl<typename tp_matmul::globals::B_gl>(B),
        .C = kittens::py::parallel_tensor_to_pgl<typename tp_matmul::globals::C_pgl>(C),
        .dev_idx = barrier.local_rank_
    };

    tp_matmul_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<device<tp_matmul_barrier::globals::NUM_DEVICES>::barrier_t>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<tp_matmul::config, tp_matmul::globals, tp_matmul::kernel>(tp_matmul_G);
    kittens::py::launch_kernel<tp_matmul_barrier::config, tp_matmul_barrier::globals, tp_matmul_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_tp_matmul", &entrypoint);
}
