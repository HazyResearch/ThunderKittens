#include <kittens.cuh>
#include <prototype.cuh>
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace matmul_reduce_scatter {

constexpr int NUM_CONSUMERS = (2); 
constexpr int NUM_PRODUCERS = (1);

static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 64;

static constexpr int NUM_DEVICES = 8;

struct globals {
    using A_tile = st_bf<Mb, Kb>;
    using B_tile = st_bf<Nb/2, Kb>;
    using C_tile = st_bf<Mb, 64>;

    using A_gl = gl<bf16, 1, 1, -1, -1, A_tile>;
    using B_gl = gl<bf16, 1, 1, -1, -1, B_tile>;
    using C_pgl = pgl<gl<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, false>;

    A_gl A;
    B_gl B;
    C_pgl C;
    const int dev_idx;
};

constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
constexpr int CLUSTER_M = 4*Mb, CLUSTER_N = Nb;

struct config {
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_BLOCKS = 148;

    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int NUM_THREADS = NUM_WORKERS * WARP_THREADS;
};

__device__ static inline int get_iters_per_task(const globals &G) {
    return G.A.cols() / Kb;
}
template<int SUPER_M=8> __device__ static inline int2 get_task_idx(const globals &G, int task_iter, bool is_consumer) {
    int cluster_x = clusterIdx().x, ctarank = cluster_ctarank();
    int task_id = task_iter * (gridDim.x/2) + cluster_x;
    int Rblocks = G.A.rows() / CLUSTER_M, Cblocks = G.B.rows() / CLUSTER_N;
    int super_rows = (Rblocks/SUPER_M)*SUPER_M,
        final_rows = Rblocks - super_rows,
        super_repeat = SUPER_M*Cblocks;
    int total_blocks = Rblocks * Cblocks;
    if (task_id < total_blocks) {
        int real_task_id = (task_id + (G.dev_idx + 1) * (total_blocks / NUM_DEVICES)) % total_blocks;
        if (real_task_id < super_rows * Cblocks) {
            return { 
                (SUPER_M*(real_task_id/super_repeat) + real_task_id%SUPER_M)*4 + ctarank*2 + is_consumer*(warpgroup::groupid()),
                is_consumer ? (real_task_id%super_repeat)/SUPER_M : 2*((real_task_id%super_repeat)/SUPER_M) + ctarank
            };
        }
        else {
            int remainder_id = real_task_id - super_rows*Cblocks;
            return {
                (super_rows + remainder_id%final_rows)*4 + ctarank*2 + is_consumer*(warpgroup::groupid()),
                is_consumer ? remainder_id/final_rows : 2*(remainder_id/final_rows) + ctarank
            };
        }
    }
    else {
        return { -1, -1 };
    }
}

__device__ inline void kernel(const globals &G) {

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
    int iters_per_task = get_iters_per_task(G);

    constexpr int PIPE_DEPTH = 4;

    using A_tile = globals::A_tile;
    using B_tile = globals::B_tile;
    using C_tile = globals::C_tile;
    
    A_tile (&a_smem)[PIPE_DEPTH][NUM_CONSUMERS] = al.allocate<A_tile, PIPE_DEPTH, NUM_CONSUMERS>();
    B_tile (&b_smem)[PIPE_DEPTH]                = al.allocate<B_tile, PIPE_DEPTH>();
    C_tile (&d_smem)                            = al.allocate<C_tile>();

    everyone::tma::cluster::sync();
    tensor_allocator<1, 2> tm_alloc{};
    using d_tt_t = tt<float, Mb, Nb>;

    __shared__ kittens::semaphore inputs_arrived[PIPE_DEPTH], inputs_finished[PIPE_DEPTH], outputs_arrived, outputs_finished[NUM_CONSUMERS];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 0) { 
        for(int i = 0; i < PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 2); 
            init_semaphore(inputs_finished[i], 0, NUM_CONSUMERS); 
        }
        init_semaphore(outputs_arrived, 0, 1);
        for(int i = 0; i < NUM_CONSUMERS; i++) {
            init_semaphore(outputs_finished[i], 0, 2);
        }
    }

    everyone::tma::cluster::sync();
    
    if(warpgroupid == NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();
        int ctarank = cluster_ctarank(); 
        if(warpgroup::warpid() == 3 && warp::laneid() == 0) {
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx(G, task_iter, false);
                if(rowcol.x == -1) {
                    for(int idx = 0; idx < (PIPE_DEPTH); idx++) {
                        tma::cluster::wait(inputs_finished[input_ring], prototype::get_phasebit<1>(bitfield, input_ring));
                        input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring);
                    }
                    if(laneid() == 0) arrive(outputs_arrived); // TODO REVIEW
                    break;
                }
                for (int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_finished[input_ring], prototype::get_phasebit<1>(bitfield, input_ring));
                    prototype::update_phasebit<1>(bitfield, input_ring);
                    if(task_iter>0 && idx==PIPE_DEPTH-1 && laneid() == 0) arrive(outputs_arrived); // TODO REVIEW 
                    tma::cluster::expect(inputs_arrived[input_ring], 0, a_smem[0][0], a_smem[0][1], b_smem[0]);
                    tma::cluster::load_async(a_smem[input_ring][0], G.A, {(rowcol.x+0), idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    tma::cluster::load_async(a_smem[input_ring][1], G.A, {(rowcol.x+1), idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    tma::cluster::load_async(b_smem[input_ring],    G.B, { rowcol.y,    idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring);
                }
            }
        }
        else if(ctarank == 0 && warp::laneid() == 0 && (warpgroup::warpid() == 0 || warpgroup::warpid() == 1)) { // launch the MMA's
            d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(warpgroup::warpid()*Nb);
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx(G, task_iter, false);
                if(rowcol.x == -1) break;
                tma::cluster::wait(outputs_finished[warpgroup::warpid()], (task_iter+1)%2); // make sure tensor memory is ready to be written to.
                tma::cluster::wait(inputs_arrived[input_ring], prototype::get_phasebit<0>(bitfield, input_ring));
                prototype::update_phasebit<0>(bitfield, input_ring);
                mm2_ABt(d_tt, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring);
                for(int idx = 1; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_arrived[input_ring], prototype::get_phasebit<0>(bitfield, input_ring));
                    prototype::update_phasebit<0>(bitfield, input_ring);
                    mma2_ABt(d_tt, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring);
                }
            }
        }
    }
    else {
        warpgroup::increase_registers<224>();
        d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(warpgroupid*Nb);
        int fine_Rblocks_per_dev = ((G.A.rows() / CLUSTER_M) * 4) / NUM_DEVICES;
        for(int task_iter = 0; true; task_iter++) {
            int2 rowcol = get_task_idx(G, task_iter, true);
            if(rowcol.x == -1) break;
            int dst_dev_idx = rowcol.x / fine_Rblocks_per_dev;
            rowcol.x %= fine_Rblocks_per_dev;
            kittens::wait(outputs_arrived, task_iter%2);
            rt_bf<Mb/4, C_tile::cols> d_reg[4];
            if(warpgroupid == 1) group<8>::sync(15);
            #pragma unroll
            for(int i = 0; i < Nb/C_tile::cols; i++) {
                warpgroup::load_async(d_reg[i], d_tt.subtile<tt<float, 128, 64>>(0, 64*i));
            }
            tensor_load_wait();
            warpgroup::sync(warpgroupid);
            if(warpgroup::laneid() == 0) kittens::warp::tma::cluster::arrive(outputs_finished[warpgroupid], 0); // Tensor memory for warpgroup 0 is now free.
            if(warpgroupid == 0) group<8>::sync(15);
            if(warpgroupid == 1) group<8>::sync(14);
            warpgroup::store(d_smem, d_reg[0]);
            warpgroup::sync(warpgroupid);
            warpgroup::tma::store_add_async(G.C[dst_dev_idx], d_smem, {rowcol.x, 4*rowcol.y+0});
            #pragma unroll
            for(int i = 1; i < Nb/C_tile::cols; i++) {
                tma::store_async_read_wait();
                warpgroup::sync(warpgroupid);
                warpgroup::store(d_smem, d_reg[i]);
                warpgroup::sync(warpgroupid);
                warpgroup::tma::store_add_async(G.C[dst_dev_idx], d_smem, {rowcol.x, 4*rowcol.y+i});
            }
            tma::store_async_read_wait();
            if(warpgroupid == 0) group<8>::sync(14);
            group<8>::sync(15); // All consumers sync here.
        }
    }
    everyone::tma::cluster::sync();
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

} // namespace matmul_reduce_scatter_barrier

void entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &barrier
) {
    matmul_reduce_scatter::globals matmul_reduce_scatter_G {
        .A = kittens::py::tensor_to_gl<typename matmul_reduce_scatter::globals::A_gl>(A),
        .B = kittens::py::tensor_to_gl<typename matmul_reduce_scatter::globals::B_gl>(B),
        .C = kittens::py::parallel_tensor_to_pgl<typename matmul_reduce_scatter::globals::C_pgl>(C),
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
