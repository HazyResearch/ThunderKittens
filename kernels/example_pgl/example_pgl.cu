#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

struct globals {
    static constexpr int num_devices = 4;
    using matrix_layout = pgl<gl<float, -1, -1, -1, -1>, num_devices>;

    matrix_layout in_mat;
    matrix_layout out_mat;
    int n;

    int dev_idx;
    dim3 grid() { return dim3(4); }
    dim3 block() { return dim3(kittens::WARP_THREADS); }
    int dynamic_shared_memory() { return kittens::MAX_SHARED_MEMORY; }
};

__global__ void example_pgl_kernel(const __grid_constant__ globals g) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = tid + bid * blockDim.x;

    if (tid == 0 && bid == 0) printf("g.dev_idx = %d\n", g.dev_idx);

    if (index < g.n) {
        g.out_mat[0].raw_ptr[index] = g.in_mat[0].raw_ptr[index];
    }
}

PYBIND11_MODULE(example_pgl, m) {
    m.doc() = "example_pgl python module";
    kittens::py::bind_multigpu_kernel<example_pgl_kernel>(
        m,
        "example_pgl",
        &globals::in_mat,
        &globals::out_mat,
        &globals::n
    );
}
