#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

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
    using A_comm_tile = st_bf<ROW_BLOCK * 2, RED_BLOCK * 2>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>; // warpgroup distributed

    static constexpr int NUM_CHUNKS = config::DYNAMIC_SHARED_MEMORY / sizeof(globals::A_comm_tile);

    using A_pgl = pgl<gl<bf16, 1, 1, -1, -1, A_tile, A_comm_tile>, NUM_DEVICES, true, A_comm_tile>;
    using B_gl = gl<bf16, 1, 1, -1, -1, B_tile>;
    using C_gl = gl<bf16, 1, 1, -1, -1, C_tile>;
    using barrier_pgl = device<NUM_DEVICES>::barrier_t;

    A_pgl A;
    B_gl B;
    C_gl C;
    barrier_pgl barrier;

    const int dev_idx;
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

__device__ inline void comm_sm(const globals &G) {
    // Declare and allocate shared memory
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);

    static_assert(globals::NUM_CHUNKS < config::NUM_WARPS);
    typename globals::A_comm_tile (&A_smem)[globals::NUM_CHUNKS] = al.allocate<typename globals::A_comm_tile, globals::NUM_CHUNKS>();
    __shared__ kittens::semaphore inputs_arrived[globals::NUM_CHUNKS];

    // Common variables
    const int comm_sm_id = blockIdx.x - G.num_comp_sms;
    const int warp_id = warp::groupid();
    const int lane_id = warp::laneid();
    const int global_row_blocks = G.A.rows() / (globals::ROW_BLOCK * 2);
    const int local_row_blocks = global_row_blocks / globals::NUM_DEVICES;
    const int col_blocks = G.A.cols() / (globals::RED_BLOCK * 2); // different than comp_sm!
    const int num_local_blocks = local_row_blocks * col_blocks;
    uint32_t phasebits = 0xFFFF0000;

    if (warp_id < globals::NUM_CHUNKS && lane_id == 0) {
        init_semaphore(inputs_arrived[warp_id], 0, 1);

        for (int task_id = comm_sm_id * globals::NUM_CHUNKS + warp_id; task_id < num_local_blocks; task_id += G.num_comm_sms * globals::NUM_CHUNKS) {
            // Calculate indices
            const int row_idx = task_id / col_blocks;
            const int global_row_idx = row_idx + G.dev_idx * local_row_blocks;
            const int col_idx = task_id % col_blocks;

            // Load
            tma::expect_bytes(inputs_arrived[warp_id], sizeof(globals::A_comm_tile));
            tma::load_async(A_smem[warp_id], G.A[G.dev_idx], {global_row_idx, col_idx}, inputs_arrived[warp_id]);

            // Store
            wait(inputs_arrived[warp_id], get_phasebit<0>(phasebits, warp_id));
            update_phasebit<0>(phasebits, warp_id);
            tma::store_async(G.A, A_smem[warp_id], {global_row_idx, col_idx});
            tma::store_async_wait();

            // Signal
            if (col_idx + G.num_comm_sms * globals::NUM_CHUNKS >= col_blocks)
                device<globals::NUM_DEVICES>::signal_all(G.barrier, {global_row_idx}, 1);
        }
    }
}

__device__ inline void comp_sm(const globals &G) {
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
    const int warpgroup_id = warpgroup::groupid();
    const int warp_id = warpgroup::warpid();
    const int lane_id = warp::laneid();
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    const int global_row_blocks = G.A.rows() / globals::ROW_BLOCK;
    const int local_row_blocks = global_row_blocks / globals::NUM_DEVICES;
    const int col_blocks = G.B.cols() / globals::COL_BLOCK;
    const int super_rows = (local_row_blocks / globals::SUPER_M) * globals::SUPER_M;
    const int final_rows = local_row_blocks - super_rows;
    const int super_blocks = globals::SUPER_M * col_blocks;
    const int num_global_blocks = global_row_blocks * col_blocks;
    const int num_local_blocks = local_row_blocks * col_blocks;
    const int num_iters = G.A.cols() / globals::RED_BLOCK;
    const int num_comm_workers_per_stage = min(G.num_comm_sms * globals::NUM_CHUNKS, num_iters / 2);

    // Main divergence
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        if (warp_id == 0 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < num_global_blocks; task_id += G.num_comp_sms) {
                // Calculate indices
                int row_idx, col_idx;
                if (task_id < num_local_blocks) {
                    int local_row_idx;
                    if (task_id < super_rows * col_blocks) {
                        local_row_idx = globals::SUPER_M * (task_id / super_blocks) + task_id % globals::SUPER_M;
                        col_idx = (task_id % super_blocks) / globals::SUPER_M;
                    } else {
                        int remainder_id = task_id - super_rows * col_blocks;
                        local_row_idx = super_rows + remainder_id % final_rows;
                        col_idx = remainder_id / final_rows;
                    }
                    row_idx = local_row_idx + G.dev_idx * local_row_blocks;
                } else {
                    const int num_peer_devices = globals::NUM_DEVICES - 1;
                    const int amortized_task_id = task_id - num_local_blocks;
                    const int target_shard = amortized_task_id / (num_peer_devices * col_blocks);
                    const int idx_in_shard = amortized_task_id % (num_peer_devices * col_blocks);
                    const int shard_row_idx = idx_in_shard % num_peer_devices;
                    row_idx = (shard_row_idx >= G.dev_idx ? shard_row_idx + 1 : shard_row_idx) * local_row_blocks + target_shard;
                    col_idx = idx_in_shard / num_peer_devices;
                    device<globals::NUM_DEVICES>::wait(G.barrier, {row_idx / 2}, G.dev_idx, num_comm_workers_per_stage);
                }

                for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    tma::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs));
                    if (red_idx == globals::PIPELINE_STAGES - 1) { // ASSUMPTION: K is always a multiple of 4*64
                        wait(outputs_finished, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                        update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);
                    }
                    #pragma unroll
                    for (int i = 0; i < 2; i++)
                        tma::load_async(inputs[stage].A[i], G.A[G.dev_idx], {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                    tma::load_async(inputs[stage].B, G.B, {red_idx, col_idx}, inputs_arrived[stage]);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        } else if (warp_id == 1 && lane_id == 0) {
            for (int task_id = blockIdx.x; task_id < num_global_blocks; task_id += G.num_comp_sms) {
                // Calculate indices
                int row_idx, col_idx;
                if (task_id < num_local_blocks) {
                    int local_row_idx;
                    if (task_id < super_rows * col_blocks) {
                        local_row_idx = globals::SUPER_M * (task_id / super_blocks) + task_id % globals::SUPER_M;
                        col_idx = (task_id % super_blocks) / globals::SUPER_M;
                    } else {
                        int remainder_id = task_id - super_rows * col_blocks;
                        local_row_idx = super_rows + remainder_id % final_rows;
                        col_idx = remainder_id / final_rows;
                    }
                    row_idx = local_row_idx + G.dev_idx * local_row_blocks;
                } else {
                    const int num_peer_devices = globals::NUM_DEVICES - 1;
                    const int amortized_task_id = task_id - num_local_blocks;
                    const int target_shard = amortized_task_id / (num_peer_devices * col_blocks);
                    const int idx_in_shard = amortized_task_id % (num_peer_devices * col_blocks);
                    const int shard_row_idx = idx_in_shard % num_peer_devices;
                    row_idx = (shard_row_idx >= G.dev_idx ? shard_row_idx + 1 : shard_row_idx) * local_row_blocks + target_shard;
                    col_idx = idx_in_shard / num_peer_devices;
                }

                wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                update_phasebit<0>(phasebits, 0);
                #pragma unroll
                for (int i = 0; i < 2; i++)
                    tma::store_async(G.C, outputs.C[i], {row_idx * 2 + i, col_idx});
                tma::store_async_read_wait();
                arrive(outputs_finished);
            }
        }
    } else {
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        for (int task_id = blockIdx.x; task_id < num_global_blocks; task_id += G.num_comp_sms) {
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

__device__ inline void main_kernel(const globals &G) {
    if (blockIdx.x < G.num_comp_sms)
        comp_sm(G);
    else
        comm_sm(G);
}

__device__ inline void epilogue_kernel(const globals &G) {
    // Reset the barrier
    const int num_blocks = G.A.rows() / (globals::ROW_BLOCK * 2);
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = offset; i < num_blocks; i += stride)
        G.barrier[G.dev_idx][{i}] = 0;

    // Ensure all devices exit together
    if (blockIdx.x == 0 && threadIdx.x == 0)
        device<globals::NUM_DEVICES>::barrier(G.barrier, {1, 0, 0}, G.dev_idx);
}

void entrypoint(
    kittens::py::TKParallelTensor &A,
    const at::Tensor &B,
    at::Tensor &C,
    kittens::py::TKParallelTensor &barrier,
    const int num_comm_sms
) {
    globals G {
        .A = kittens::py::parallel_tensor_to_pgl<globals::A_pgl>(A),
        .B = kittens::py::tensor_to_gl<globals::B_gl>(B),
        .C = kittens::py::tensor_to_gl<globals::C_gl>(C),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = barrier.local_rank_,
        .num_comm_sms = num_comm_sms,
        .num_comp_sms = config::NUM_BLOCKS - num_comm_sms
    };
    kittens::py::launch_kernel<config, globals, main_kernel>(G);
    kittens::py::launch_kernel<config, globals, epilogue_kernel>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("all_gather_matmul", &entrypoint);
}
