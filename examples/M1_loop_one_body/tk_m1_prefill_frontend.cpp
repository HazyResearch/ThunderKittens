#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>


extern void  prefill(torch::Tensor W1,
                     torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                     torch::Tensor Out,
                     cudaStream_t stream);

extern void  prefill_ref(torch::Tensor W1,
                         torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                         torch::Tensor Out)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    prefill(W1, XA, XB, XC, Out, stream);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("prefill", &prefill_ref);
}