#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>
#include "src/common/pyutils/torch_helpers.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);


void
a012_compute(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    
    auto batch = q.size(0);
    auto head  = q.size(1);
    auto n     = q.size(2);
    auto d     = q.size(3);
    auto dv    = v.size(3);
    bool k_same = true, o_same = true;
    for(auto i = 0; i < 4; i++) { 
        k_same &= q.size(i) == k.size(i);
        o_same &= v.size(i) == o.size(i);
    }
    // This is just a restriction of what we're doing now...
    TORCH_CHECK(k_same, "Q and K should be same size");
    TORCH_CHECK(o_same, "V and O should be same size");

    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "Q is a Bfloat");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "K is a Bfloat");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "V is a Bfloat");
    TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "O is a Bfloat");

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    constexpr bool _debug_build = false;
    const int NUM_WORKERS = 8;

    // q,k,v, and o are all double buffered
    unsigned long mem_size  =  2*2*NUM_WORKERS*sizeof(st_bf_1x1<ducks::st_layout::xor_swizzle>); // q, k and v are double buffered.
                  mem_size +=    2*NUM_WORKERS*sizeof(st_bf_1x4<ducks::st_layout::xor_swizzle>);
                  mem_size += (NUM_WORKERS+NUM_WORKERS)*sizeof(st_bf_1x4<ducks::st_layout::xor_swizzle>);
                  mem_size += 2*NUM_WORKERS*sizeof(st_bf_1x4<ducks::st_layout::xor_swizzle>); // a0 and a1y

    TORCH_CHECK(n % (NUM_WORKERS*kittens::TILE_DIM) == 0, "The number of elements should be divisible the number of NUM_WORKERS times stored fragments");
    auto threads = NUM_WORKERS * kittens::WARP_THREADS;
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
             a012_compute_ker<H, T, _debug_build>,
             cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size));
    
    a012_compute_ker<H,T,false><<<batch*head,threads,mem_size>>>(n, d, dv, q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(),
          o.data_ptr<T>(), NULL, NULL, NULL);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// extern void a2_compute(const torch::Tensor q, const torch::Tensor k, const torch::Tensor v, torch::Tensor y);
extern void a012_compute(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    // m.def("a2_compute", a2_compute);
    m.def("a012_compute", a012_compute);
}
 