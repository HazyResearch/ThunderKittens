#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>


extern void  batch_add(
                     torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                     cudaStream_t stream);

extern void  batch_add_stream(
                         torch::Tensor XA, torch::Tensor XB, torch::Tensor XC)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    batch_add(XA, XB, XC, stream);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_add", &batch_add_stream);
}