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
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>; // warpgroup distributed

    using A_gl = gl<bf16, 1, 1, -1, -1, A_tile>;
    using B_gl = gl<bf16, 1, 1, -1, -1, B_tile>;
    using C_pgl = pgl<gl<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, true>;
    using barrier_pgl = device<NUM_DEVICES>::barrier_t;

    A_gl A;
    B_gl B;
    C_pgl C;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_comm_sms;
    const int num_comp_sms;
};

struct lcsct {
    struct comp_sem {
        semaphore inputs_arrived[globals::PIPELINE_STAGES];
        semaphore inputs_finished[globals::PIPELINE_STAGES];
        semaphore outputs_arrived;
        semaphore outputs_finished;
    };

    struct comm_sem { };
    
    struct comp_smem {
        struct pipeline_inputs {
            globals::A_tile A[2];
            globals::B_tile B;
        };
        pipeline_inputs inputs[globals::PIPELINE_STAGES];
    };

    struct comm_smem { };

    struct comp_regs {
        int worker_id;
        int comm_workers;
        int comp_workers;

        int num_tasks;
        int task_offset;
        int task_stride;
        int task_id;

        int warpgroup_id;
        int warp_id;
        int lane_id;
        int stage;
        uint32_t phasebits;
        int row_blocks;
        int col_blocks;
        int super_rows;
        int final_rows;
        int super_blocks;
        int num_iters;

        globals::C_tile *C;
    };

    struct comm_regs {
        int worker_id;
        int comm_workers;
        int comp_workers;

        int num_tasks;
        int task_offset;
        int task_stride;
        int task_id;

        int warpgroup_id;
        int lane_in_warpgroup;
        int row_blocks;
        int col_blocks;
        int super_rows;
        int final_rows;
        int super_blocks;
    };

    __device__ static inline void comp_setup(const globals &G, comp_sem &sem, comp_smem &smem, comp_regs &regs) {
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
                init_semaphore(sem.inputs_arrived[i], 0, 1);
                init_semaphore(sem.inputs_finished[i], 0, 8);
            }
            init_semaphore(sem.outputs_arrived, 0, 2);
            init_semaphore(sem.outputs_finished, 0, 1);
        }
        __syncthreads();
        regs.warpgroup_id = warpgroup::groupid();
        regs.warp_id = warpgroup::warpid();
        regs.lane_id = warp::laneid();
        regs.stage = 0;
        regs.phasebits = 0xFFFF0000;
        regs.row_blocks = G.A.rows() / globals::ROW_BLOCK;
        regs.col_blocks = G.B.cols() / globals::COL_BLOCK;
        regs.super_rows = (regs.row_blocks / globals::SUPER_M) * globals::SUPER_M;
        regs.final_rows = regs.row_blocks - regs.super_rows;
        regs.super_blocks = globals::SUPER_M * regs.col_blocks;
        regs.num_tasks = regs.row_blocks * regs.col_blocks;
        regs.task_offset = regs.worker_id;
        regs.task_stride = regs.comp_workers;
        regs.num_iters = G.A.cols() / globals::RED_BLOCK;
        regs.C = reinterpret_cast<globals::C_tile *>(&smem.inputs[globals::PIPELINE_STAGES - 1]);
    }

    __device__ static inline void comm_setup(const globals &G, comm_sem &sem, comm_smem &smem, comm_regs &regs) {
        regs.warpgroup_id = warpgroup::groupid();
        regs.lane_in_warpgroup = warpgroup::laneid();
        regs.row_blocks = G.A.rows() / globals::ROW_BLOCK;
        regs.col_blocks = G.B.cols() / globals::COL_BLOCK;
        regs.super_rows = (regs.row_blocks / globals::SUPER_M) * globals::SUPER_M;
        regs.final_rows = regs.row_blocks - regs.super_rows;
        regs.super_blocks = globals::SUPER_M * regs.col_blocks;
        regs.num_tasks = regs.row_blocks * regs.col_blocks;
        regs.task_offset = globals::NUM_DEVICES * regs.worker_id + G.dev_idx;
        regs.task_stride = globals::NUM_DEVICES * regs.comm_workers;
    }

    __device__ static inline void loader(const globals &G, comp_sem &sem, comp_smem &smem, comp_regs &regs) { 
        int real_task_id = regs.task_id;
        int row_idx, col_idx;
        if (real_task_id < regs.super_rows * regs.col_blocks) {
            row_idx = globals::SUPER_M * (real_task_id / regs.super_blocks) + real_task_id % globals::SUPER_M;
            col_idx = (real_task_id % regs.super_blocks) / globals::SUPER_M;
        } else {
            int remainder_id = real_task_id - regs.super_rows * regs.col_blocks;
            row_idx = regs.super_rows + remainder_id % regs.final_rows;
            col_idx = remainder_id / regs.final_rows;
        }

        for (int red_idx = 0; red_idx < regs.num_iters; red_idx++) {
            wait(sem.inputs_finished[regs.stage], get_phasebit<1>(regs.phasebits, regs.stage));
            update_phasebit<1>(regs.phasebits, regs.stage);
            tma::expect_bytes(sem.inputs_arrived[regs.stage], sizeof(globals::A_tile) * 2 + sizeof(globals::B_tile));
            if (red_idx == globals::PIPELINE_STAGES - 1) { // ASSUMPTION: K is always a multiple of 4*64
                wait(sem.outputs_finished, get_phasebit<1>(regs.phasebits, globals::PIPELINE_STAGES));
                update_phasebit<1>(regs.phasebits, globals::PIPELINE_STAGES);
            }
            #pragma unroll
            for (int i = 0; i < 2; i++)
                tma::load_async(smem.inputs[regs.stage].A[i], G.A, {row_idx * 2 + i, red_idx}, sem.inputs_arrived[regs.stage]);
            tma::load_async(smem.inputs[regs.stage].B, G.B, {red_idx, col_idx}, sem.inputs_arrived[regs.stage]);
            regs.stage = (regs.stage + 1) % globals::PIPELINE_STAGES;
        }
    }

    __device__ static inline void storer(const globals &G, comp_sem &sem, comp_smem &smem, comp_regs &regs) { 
        int real_task_id = regs.task_id;
        int row_idx, col_idx;
        if (real_task_id < regs.super_rows * regs.col_blocks) {
            row_idx = globals::SUPER_M * (real_task_id / regs.super_blocks) + real_task_id % globals::SUPER_M;
            col_idx = (real_task_id % regs.super_blocks) / globals::SUPER_M;
        } else {
            int remainder_id = real_task_id - regs.super_rows * regs.col_blocks;
            row_idx = regs.super_rows + remainder_id % regs.final_rows;
            col_idx = remainder_id / regs.final_rows;
        }

        wait(sem.outputs_arrived, get_phasebit<0>(regs.phasebits, 0));
        update_phasebit<0>(regs.phasebits, 0);
        #pragma unroll
        for (int i = 0; i < 2; i++)
            tma::store_async(G.C[G.dev_idx], regs.C[i], {row_idx * 2 + i, col_idx});
        tma::store_async_read_wait();
        arrive(sem.outputs_finished);

        int signal_dev_idx = regs.task_id % globals::NUM_DEVICES; // static assignment for now. Ordering must match with comm SM signaling
        device<globals::NUM_DEVICES>::signal(G.barrier, {row_idx, col_idx}, signal_dev_idx, 1);
    }

    __device__ static inline void consumer(const globals &G, comp_sem &sem, comp_smem &smem, comp_regs &regs) {
        rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK> C_accum;
        warp::zero(C_accum);

        for (int red_idx = 0; red_idx < regs.num_iters; red_idx++) {
            wait(sem.inputs_arrived[regs.stage], get_phasebit<0>(regs.phasebits, regs.stage));
            update_phasebit<0>(regs.phasebits, regs.stage);
            warpgroup::mma_AB(C_accum, smem.inputs[regs.stage].A[regs.warpgroup_id], smem.inputs[regs.stage].B);
            warpgroup::mma_async_wait();
            warp::arrive(sem.inputs_finished[regs.stage]);
            regs.stage = (regs.stage + 1) % globals::PIPELINE_STAGES;
        }

        group<8>::sync(3);
        warpgroup::store(regs.C[regs.warpgroup_id], C_accum);
        warpgroup::sync(regs.warpgroup_id + 1);
        warpgroup::arrive(sem.outputs_arrived);
     }

    __device__ static inline void communicator(const globals &G, comm_sem &sem, comm_smem &smem, comm_regs &regs) {
        // Calculate indices
        int real_task_id = regs.task_id;
        int row_idx, col_idx;
        if (real_task_id < regs.super_rows * regs.col_blocks) {
            row_idx = globals::SUPER_M * (real_task_id / regs.super_blocks) + real_task_id % globals::SUPER_M;
            col_idx = (real_task_id % regs.super_blocks) / globals::SUPER_M;
        } else {
            int remainder_id = real_task_id - regs.super_rows * regs.col_blocks;
            row_idx = regs.super_rows + remainder_id % regs.final_rows;
            col_idx = remainder_id / regs.final_rows;
        }

        // Cross-GPU barrier
        if (threadIdx.x == 0)
            device<globals::NUM_DEVICES>::wait(G.barrier, {row_idx, col_idx}, G.dev_idx, globals::NUM_DEVICES);
        __syncthreads();

        // Do in-network all-reduce
        group<config::NUM_WARPS>::all_reduce<globals::ROW_BLOCK, globals::COL_BLOCK, reduce_op::ADD>(G.C, {row_idx, col_idx});
    }
};

__device__ inline void epilogue_kernel(const globals &G) {
    // Reset the barrier
    const int row_blocks = G.A.rows() / globals::ROW_BLOCK;
    const int col_blocks = G.B.cols() / globals::COL_BLOCK;
    const int num_blocks = row_blocks * col_blocks;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = offset; i < num_blocks; i += stride)
        G.barrier[G.dev_idx][{i / col_blocks, i % col_blocks}] = 0;

    // Ensure all devices exit together
    if (blockIdx.x == 0 && threadIdx.x == 0)
        device<globals::NUM_DEVICES>::barrier(G.barrier, {1, 0, 0}, G.dev_idx);
}

void entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &barrier,
    const int num_comm_sms
) {
    globals G {
        .A = kittens::py::tensor_to_gl<globals::A_gl>(A),
        .B = kittens::py::tensor_to_gl<globals::B_gl>(B),
        .C = kittens::py::parallel_tensor_to_pgl<globals::C_pgl>(C),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = barrier.local_rank_,
        .num_comm_sms = num_comm_sms,
        .num_comp_sms = config::NUM_BLOCKS - num_comm_sms
    };
    lcsc::launch_kernel<config, globals, lcsct>(G, at::cuda::getCurrentCUDAStream());
    kittens::py::launch_kernel<config, globals, epilogue_kernel>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("matmul_all_reduce", &entrypoint);
}
