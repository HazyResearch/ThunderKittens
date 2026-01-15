#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct config {
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_BLOCKS = 148;

    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int PIPELINE_STAGES = 5;
    static constexpr int MMA_PIPE_DEPTH = 2;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 128;

    using A_tile = st_fp8e4m3<ROW_BLOCK / 2, RED_BLOCK>;
    using A_comm_tile = st_fp8e4m3<ROW_BLOCK, RED_BLOCK * 2>;
    using B_tile = st_fp8e4m3<COL_BLOCK / 2, RED_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK / EPI_PIPE_DEPTH>;

    static constexpr int NUM_CHUNKS = config::DYNAMIC_SHARED_MEMORY / sizeof(globals::A_comm_tile);

    using A_pgl = pgl<gl<fp8e4m3, 1, 1, -1, -1, A_tile, A_comm_tile>, NUM_DEVICES, true, A_comm_tile>;
    using B_gl = gl<fp8e4m3, 1, 1, -1, -1, B_tile>;
    using C_gl = gl<bf16, 1, 1, -1, -1, C_tile>;
    using barrier_pgl = barrier_t<NUM_DEVICES>;

    A_pgl A;
    B_gl B;
    C_gl C;
    barrier_pgl barrier;

    const int dev_idx;
    const int num_comm_sms;
    const int num_comp_sms;
};

__device__ inline void comm_sm(const globals &g) {
    // Declare and allocate shared memory
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);

    static_assert(globals::NUM_CHUNKS < config::NUM_WARPS);
    typename globals::A_comm_tile (&A_smem)[globals::NUM_CHUNKS] = al.allocate<typename globals::A_comm_tile, globals::NUM_CHUNKS>();
    __shared__ kittens::semaphore inputs_arrived[globals::NUM_CHUNKS];

    // Common variables
    const int comm_sm_id = blockIdx.x - g.num_comp_sms;
    const int warp_id = warp::groupid();
    const int lane_id = warp::laneid();
    const int global_row_blocks = g.A.rows() / globals::ROW_BLOCK;
    const int local_row_blocks = global_row_blocks / globals::NUM_DEVICES;
    const int col_blocks = g.A.cols() / (globals::RED_BLOCK * 2); // different from comp_sm!
    const int num_local_blocks = local_row_blocks * col_blocks;
    uint32_t phasebits = 0xFFFF0000;

    if (warp_id < globals::NUM_CHUNKS && lane_id == 0) {
        init_semaphore(inputs_arrived[warp_id], 0, 1);

        for (int task_id = comm_sm_id * globals::NUM_CHUNKS + warp_id; task_id < num_local_blocks; task_id += g.num_comm_sms * globals::NUM_CHUNKS) {
            // Calculate indices
            const int row_idx = task_id / col_blocks;
            const int global_row_idx = row_idx + g.dev_idx * local_row_blocks;
            const int col_idx = task_id % col_blocks;

            // Load
            tma::expect_bytes(inputs_arrived[warp_id], sizeof(globals::A_comm_tile));
            tma::load_async(A_smem[warp_id], g.A[g.dev_idx], {global_row_idx, col_idx}, inputs_arrived[warp_id]);

            // Store
            wait(inputs_arrived[warp_id], get_phasebit<0>(phasebits, warp_id));
            update_phasebit<0>(phasebits, warp_id);
            tma::store_async(g.A, A_smem[warp_id], {global_row_idx, col_idx});
            tma::store_async_wait();

            // Signal
            if (col_idx + g.num_comm_sms * globals::NUM_CHUNKS >= col_blocks)
                signal_all(g.barrier, {global_row_idx}, 1);
        }
    }
}

__device__ inline void comp_sm(const globals &g) {
    using C = config;
    using G = globals;

    const int cta_rank = cluster_ctarank();
    const int cluster_idx = blockIdx.x / C::CLUSTER_SIZE;
    const int iters_per_task = g.A.cols() / G::RED_BLOCK;

    const int global_row_blocks = g.A.rows() / globals::ROW_BLOCK;
    const int local_row_blocks = global_row_blocks / globals::NUM_DEVICES;
    const int col_blocks = g.B.rows() / globals::COL_BLOCK;
    const int super_rows = (local_row_blocks / globals::SUPER_M) * globals::SUPER_M;
    const int final_rows = local_row_blocks - super_rows;
    const int super_blocks = globals::SUPER_M * col_blocks;
    const int num_global_blocks = global_row_blocks * col_blocks;
    const int num_local_blocks = local_row_blocks * col_blocks;
    const int num_comm_workers_per_stage = min(g.num_comm_sms * globals::NUM_CHUNKS, iters_per_task / 2);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    static_assert(sizeof(G::A_tile) * G::PIPELINE_STAGES +
                  sizeof(G::B_tile) * G::PIPELINE_STAGES +
                  sizeof(G::C_tile) * 2 <= C::DYNAMIC_SHARED_MEMORY);
    typename G::A_tile (&A_smem)[G::PIPELINE_STAGES] = al.allocate<G::A_tile, G::PIPELINE_STAGES>();
    typename G::B_tile (&B_smem)[G::PIPELINE_STAGES] = al.allocate<G::B_tile, G::PIPELINE_STAGES>();
    typename G::C_tile (&C_smem)[2]                  = al.allocate<G::C_tile, 2>();

    tensor_allocator<1, 2> tm_alloc{};
    using d_tt_t = tt<float, G::ROW_BLOCK/2, G::COL_BLOCK>;

    __shared__ semaphore inputs_arrived[G::PIPELINE_STAGES], 
                         inputs_finished[G::PIPELINE_STAGES], 
                         outputs_arrived, 
                         outputs_finished[G::MMA_PIPE_DEPTH];
    int input_ring = 0;
    int mma_ring = 0;
    uint32_t bitfield = 0xFFFF0000;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < G::PIPELINE_STAGES; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < G::MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_finished[i], 0, C::CLUSTER_SIZE);
        }
    }
    everyone::tma::cluster::sync();

    if (warpgroup::groupid() == 1) {
        warpgroup::increase_registers<256>();

        if (warp::laneid() == 0 && warpgroup::warpid() == 3) {
            for (int task_id = cluster_idx; task_id < num_global_blocks; task_id += g.num_comp_sms/2) {
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
                    row_idx = local_row_idx + g.dev_idx * local_row_blocks;
                } else {
                    const int num_peer_devices = globals::NUM_DEVICES - 1;
                    const int amortized_task_id = task_id - num_local_blocks;
                    const int target_shard = amortized_task_id / (num_peer_devices * col_blocks);
                    const int idx_in_shard = amortized_task_id % (num_peer_devices * col_blocks);
                    const int shard_row_idx = idx_in_shard % num_peer_devices;
                    row_idx = (shard_row_idx >= g.dev_idx ? shard_row_idx + 1 : shard_row_idx) * local_row_blocks + target_shard;
                    col_idx = idx_in_shard / num_peer_devices;
                    wait(g.barrier, {row_idx}, g.dev_idx, num_comm_workers_per_stage);
                }
                for (int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    update_phasebit<1>(bitfield, input_ring);
                    tma::cluster::load_async(A_smem[input_ring], g.A[g.dev_idx], {row_idx*2+cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    tma::cluster::load_async(B_smem[input_ring], g.B,            {col_idx*2+cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    input_ring=ring_advance<G::PIPELINE_STAGES>(input_ring);
                }
            }
        } else if (cta_rank == 0 && warp::laneid() == 0 && warpgroup::warpid() == 0) {
            d_tt_t d_tt[G::MMA_PIPE_DEPTH] = {
                tm_alloc.allocate<d_tt_t>(G::COL_BLOCK*0),
                tm_alloc.allocate<d_tt_t>(G::COL_BLOCK*1)
            };
            for (int task_id = cluster_idx; task_id < num_global_blocks; task_id += g.num_comp_sms/2) {
                tma::cluster::wait(outputs_finished[mma_ring], get_phasebit<1>(bitfield, mma_ring));
                update_phasebit<1>(bitfield, mma_ring);
                for(int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::expect_bytes(inputs_arrived[input_ring], 2*sizeof(G::A_tile) + 2*sizeof(G::B_tile));
                    tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    update_phasebit<0>(bitfield, input_ring);
                    if (idx == 0) mm2_ABt (d_tt[mma_ring], A_smem[input_ring], B_smem[input_ring], inputs_finished[input_ring]);
                    else          mma2_ABt(d_tt[mma_ring], A_smem[input_ring], B_smem[input_ring], inputs_finished[input_ring]);
                    input_ring=ring_advance<G::PIPELINE_STAGES>(input_ring);
                }
                kittens::detail::tcgen05::commit<C::CLUSTER_SIZE>(outputs_arrived);
                mma_ring=ring_advance<G::MMA_PIPE_DEPTH>(mma_ring);
            }
        }
    }
    else {
        warpgroup::increase_registers<256>();

        d_tt_t d_tt[G::MMA_PIPE_DEPTH] = {
            tm_alloc.allocate<d_tt_t>(G::COL_BLOCK*0),
            tm_alloc.allocate<d_tt_t>(G::COL_BLOCK*1)
        };

        for (int task_id = cluster_idx; task_id < num_global_blocks; task_id += g.num_comp_sms/2) {
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
                row_idx = local_row_idx + g.dev_idx * local_row_blocks;
            } else {
                const int num_peer_devices = globals::NUM_DEVICES - 1;
                const int amortized_task_id = task_id - num_local_blocks;
                const int target_shard = amortized_task_id / (num_peer_devices * col_blocks);
                const int idx_in_shard = amortized_task_id % (num_peer_devices * col_blocks);
                const int shard_row_idx = idx_in_shard % num_peer_devices;
                row_idx = (shard_row_idx >= g.dev_idx ? shard_row_idx + 1 : shard_row_idx) * local_row_blocks + target_shard;
                col_idx = idx_in_shard / num_peer_devices;
            }

            wait(outputs_arrived, mma_ring);
            rt_bf<G::ROW_BLOCK/8, G::COL_BLOCK/G::EPI_PIPE_DEPTH> C_reg;
            #pragma unroll
            for(int i = 0; i < G::EPI_PIPE_DEPTH; i++) {
                warpgroup::load_async(C_reg, d_tt[mma_ring].template subtile<tt<float, G::ROW_BLOCK/2, G::COL_BLOCK/G::EPI_PIPE_DEPTH>>(0, G::COL_BLOCK/G::EPI_PIPE_DEPTH*i));
                tensor_load_wait();
                if (i == G::EPI_PIPE_DEPTH - 1) {
                    warpgroup::sync(1);
                    warpgroup::tma::cluster::arrive(outputs_finished[mma_ring], 0);
                }
                warpgroup::tma::store_async_read_wait<1>();
                warpgroup::sync(1);
                warpgroup::store(C_smem[i%2], C_reg);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.C, C_smem[i%2], {row_idx*2+cta_rank, G::EPI_PIPE_DEPTH*col_idx+i});
            }
            mma_ring=ring_advance<G::MMA_PIPE_DEPTH>(mma_ring);
        }
    }
}

__device__ inline void main_kernel(const globals &g) {
    if (blockIdx.x < g.num_comp_sms)
        comp_sm(g);
    else
        comm_sm(g);
}

__device__ inline void epilogue_kernel(const globals &g) {
    // Reset the barrier
    const int num_blocks = g.A.rows() / globals::ROW_BLOCK;
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = offset; i < num_blocks; i += stride)
        g.barrier[g.dev_idx][{i}] = 0;

    // Ensure all devices exit together
    if (blockIdx.x == 0 && threadIdx.x == 0)
        barrier_all(g.barrier, {1, 0, 0}, g.dev_idx);
}

void entrypoint(
    kittens::py::TKParallelTensor &A,
    const at::Tensor &B,
    at::Tensor &C,
    kittens::py::TKParallelTensor &barrier,
    const int num_comm_sms
) {
    globals g {
        .A = kittens::py::parallel_tensor_to_pgl<globals::A_pgl>(A),
        .B = kittens::py::tensor_to_gl<globals::B_gl>(B),
        .C = kittens::py::tensor_to_gl<globals::C_gl>(C),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = barrier.local_rank_,
        .num_comm_sms = num_comm_sms,
        .num_comp_sms = config::NUM_BLOCKS - num_comm_sms
    };
    kittens::py::launch_kernel<config, globals, main_kernel>(g);
    kittens::py::launch_kernel<config, globals, epilogue_kernel>(g);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("all_gather_matmul", &entrypoint);
}
