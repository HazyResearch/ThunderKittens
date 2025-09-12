#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace reduce_scatter {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int NUM_ELEMS_PER_INST = 2;
    static constexpr int NUM_ELEMS_PER_BLOCK = config::NUM_THREADS * NUM_ELEMS_PER_INST;

    using input_layout = pgl<gl<bf16, 1, -1, -1, -1>, NUM_DEVICES, true>;
    using output_layout = pgl<gl<bf16, 1, -1, -1, -1>, NUM_DEVICES, false>;

    output_layout output;
    input_layout input;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3(output.cols() / NUM_ELEMS_PER_BLOCK * output.rows(), output.depth());
    }
};

__device__ inline void kernel(const globals &G) {
    // Intentionally verbose for demonstration
    const size_t row_col_idx = static_cast<size_t>(blockIdx.x) * globals::NUM_ELEMS_PER_BLOCK;
    const size_t col_idx = row_col_idx % G.output.cols();
    const size_t row_idx = row_col_idx / G.output.cols();
    const size_t depth_idx = blockIdx.y;
    const size_t num_cols_per_dev = G.output.cols();

    const size_t input_idx = depth_idx * G.input.rows() * G.input.cols() +
                             row_idx * G.input.cols() + 
                             col_idx +
                             num_cols_per_dev * G.dev_idx +
                             threadIdx.x * globals::NUM_ELEMS_PER_INST;
    const size_t output_idx = depth_idx * G.output.rows() * G.output.cols() +
                              row_idx * G.output.cols() + 
                              col_idx +
                              threadIdx.x * globals::NUM_ELEMS_PER_INST;

    bf16_2 tmp;
    multimem<bf16_2>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<bf16_2 *>(&G.input.mc_ptr[input_idx]));
    move<bf16_2>::stg(reinterpret_cast<bf16_2 *>(&G.output[G.dev_idx].raw_ptr[output_idx]), tmp);
}

} // namespace reduce_scatter

namespace reduce_scatter_barrier {

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

} // namespace reduce_scatter_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(output, input, barrier);

    reduce_scatter::globals reduce_scatter_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename reduce_scatter::globals::output_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename reduce_scatter::globals::input_layout>(input),
        .dev_idx = input.local_rank_
    };

    reduce_scatter_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<device<reduce_scatter_barrier::globals::NUM_DEVICES>::barrier_t>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<reduce_scatter_barrier::config, reduce_scatter_barrier::globals, reduce_scatter_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<reduce_scatter::config, reduce_scatter::globals, reduce_scatter::kernel>(reduce_scatter_G);
    kittens::py::launch_kernel<reduce_scatter_barrier::config, reduce_scatter_barrier::globals, reduce_scatter_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_reduce_scatter", &entrypoint);
}
