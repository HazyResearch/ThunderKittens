#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace all_gather {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 6;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int BLOCK_SIZE = 128;

    using shared_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;
    using output_layout = pgl<gl<bf16, 1, 1, -1, -1>, NUM_DEVICES, true, shared_tile>;
    using input_layout = pgl<gl<bf16, 1, 1, -1, -1, shared_tile>, NUM_DEVICES, false>;

    output_layout output;
    input_layout input;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3((input.cols() / BLOCK_SIZE), (input.rows() / BLOCK_SIZE));
    }

    __host__ inline int dynamic_shared_memory() const {
        return sizeof(shared_tile) + 1024;
    }
};

__device__ inline void kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::shared_tile &tile = allocator.allocate<globals::shared_tile>();

    const int row_block_idx = blockIdx.y;
    const int col_block_idx = blockIdx.x;
    const int col_blocks_per_dev = G.output.cols() / globals::BLOCK_SIZE / globals::NUM_DEVICES;

    // Assume a single-threaded block
    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, sizeof(tile));
    tma::load_async(tile, G.input[G.dev_idx], {row_block_idx, col_block_idx}, arrived);
    wait(arrived, 0);
    tma::store_async(G.output, tile, {row_block_idx, col_blocks_per_dev * G.dev_idx + col_block_idx});
}

} // namespace all_gather

namespace all_gather_barrier {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    barrier_t<NUM_DEVICES> barrier;
    const int dev_idx;
};

__device__ inline void kernel(const globals &G) {
    barrier_all(G.barrier, {0}, G.dev_idx);
}

} // namespace all_gather_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(output, input, barrier);

    all_gather::globals all_gather_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename all_gather::globals::output_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename all_gather::globals::input_layout>(input),
        .dev_idx = input.local_rank_
    };

    all_gather_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_gather_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_gather_barrier::config, all_gather_barrier::globals, all_gather_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<all_gather::config, all_gather::globals, all_gather::kernel>(all_gather_G);
    kittens::py::launch_kernel<all_gather_barrier::config, all_gather_barrier::globals, all_gather_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_gather", &entrypoint);
}
