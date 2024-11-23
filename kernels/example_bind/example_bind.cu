#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

using my_layout = gl<float, -1, -1, -1, 64, st_fl<64,64>>; // An example layout that also instantiates a TMA descriptor on Hopper.
struct globals {
    my_layout in, out;
    dim3 grid()  { return dim3(in.batch, in.depth, in.rows); }
    dim3 block() { return dim3(in.cols); }
};
__global__ void copy_kernel(const __grid_constant__ globals g) {
    if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) printf("Hello, from inside the kernel!\n");
    g.out[{blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x}] = g.in[{blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x}];
}
void run_copy_kernel(globals g) {
    printf("I am calling the kernel from the host.\n");
    copy_kernel<<<g.grid(), g.block()>>>(g);
}

PYBIND11_MODULE(example_bind, m) {
    m.doc() = "example_bind python module";
    BIND_KERNEL(m, "copy_kernel", copy_kernel, globals, in, out); // For wrapping kernels directly.
    BIND_FUNCTION(m, "wrapped_copy_kernel", run_copy_kernel, globals, in, out); // For host functions that wrap the kernel.
}
