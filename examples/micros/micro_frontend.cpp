#include <torch/extension.h>
#include <vector>


extern void  micro(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("micro", micro);
}