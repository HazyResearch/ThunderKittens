#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// using namespace nvcuda;

// *******
// ** BROADCAST API
// Constant broadcasts.
__device__ __nv_bfloat16* device_cast(c10::BFloat16* x) { return reinterpret_cast<__nv_bfloat16*>(x);}
__device__ half* device_cast(at::Half *x) { return reinterpret_cast<half*>(x); }
__device__ float* device_cast(float* x) { return x;}
__device__ const __nv_bfloat16* device_cast(const c10::BFloat16* x) { return reinterpret_cast<const __nv_bfloat16*>(x);}
__device__ const half* device_cast(const at::Half *x) { return reinterpret_cast<const half*>(x); }
__device__ const float* device_cast(const float* x) { return x;}

// This is a dispatch helper macro for us, to match the device and the pytorch style.
// It's modeled after the AT_DISPATCH macros.
// Note: that when you pass in FUNC you usually write it with parens to help the preprocessor parse it.
// It will give you errors about more parameters than the two it was expecting, if not.
#define DISPATCH(t, FUNC)\
    switch (t.scalar_type()) {\
        case c10::ScalarType::BFloat16: {\
            using H = __nv_bfloat16;\
            using D = __nv_bfloat162;\
            using T = c10::BFloat16;\
            using ACCUM = wmma_accum;\
            FUNC;\
        }\
            break;\
        case c10::ScalarType::Half: {\
            using H = half;\
            using D = __half2;\
            using T = at::Half;\
            using ACCUM = wmma_accum;\
            FUNC;\
        }\
            break;\
        case c10::ScalarType::Float: {\
            using H = float;\
            using D = float2;\
            using T = float;\
            using ACCUM = wmma_accum_tf32;\
            FUNC;\
        }\
            break;\
        default:\
            TORCH_CHECK(false, "Unsupported type!");\
    }

// copied from online tutorial.
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, char const* const func, char const* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        //std::exit(EXIT_FAILURE);
    }
}

bool is_tile(torch::Tensor t) {return t.size(0) == kittens::TILE_DIM && t.size(1) == kittens::TILE_DIM;}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_TILE(x) CHECK_INPUT(x); TORCH_CHECK(is_tile(x), #x " must be a 16x16 tile")