#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>

#include "kittens.cuh"
#include "pyutils/broker.cuh"

namespace kittens {
namespace py {

template <typename Config, typename Globals, auto Kernel>
__global__ __launch_bounds__(Config::NUM_THREADS, 1)
void global_kernel_unclustered(const __grid_constant__ Globals G) {
    Kernel(G);
}

template <typename Config, typename Globals, auto Kernel>
__global__ __launch_bounds__(Config::NUM_THREADS, 1) __cluster_dims__(Config::CLUSTER_SIZE)
void global_kernel_clustered(const __grid_constant__ Globals G) {
    Kernel(G);
}

template <typename Layout>
static inline void tensor_check(const at::Tensor &t) {
    TORCH_CHECK(t.is_cuda(), "Tensor must be on CUDA device")
    TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous")
    TORCH_CHECK(t.dim() <= 4, "Expected Tensor.dim() <= 4");

    if constexpr (std::is_same_v<typename Layout::dtype, char>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Char, "Tensor has invalid dtype (expected int8)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, short>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Short, "Tensor has invalid dtype (expected int16)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, int>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Int, "Tensor has invalid dtype (expected int32)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, long>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Long, "Tensor has invalid dtype (expected int64)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp8e4m3>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e4m3fn, "Tensor has invalid dtype (expected fp8e4m3)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp8e5m2>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e5m2, "Tensor has invalid dtype (expected fp8e5m2)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp8e8m0>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Byte, "Tensor has invalid dtype (expected fp8e8m0 represented as uint8)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::bf16>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::BFloat16, "Tensor has invalid dtype (expected bfloat16)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::half>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Half, "Tensor has invalid dtype (expected float16)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, float>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float, "Tensor has invalid dtype (expected float32)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, double>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Double, "Tensor has invalid dtype (expected float64)");
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

template <kittens::ducks::gl::all GL>
static inline GL tensor_to_gl(const at::Tensor &t) {
    tensor_check<GL>(t);

    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < static_cast<int>(t.dim()); ++i)
        shape[4 - t.dim() + i] = static_cast<int>(t.size(i));

    uint64_t data_ptr = reinterpret_cast<uint64_t>(t.data_ptr());

    return ::kittens::make_gl<GL>(data_ptr, shape[0], shape[1], shape[2], shape[3]);
}

template <kittens::ducks::pgl::all PGL>
static inline PGL tensor_to_pgl(const at::Tensor &t, KittensBroker &broker) {
    TORCH_CHECK(PGL::num_devices == broker.local_world_size_, "Number of devices mismatch between PGL and KittensBroker");
    TORCH_CHECK(!PGL::_INIT_MC, "PGL must be initialized with INIT_MC=false for multiprocess use");
    TORCH_CHECK(broker.local_rank_ == t.device().index(), "Current tensor device index mismatch with KittensBroker");

    tensor_check<PGL>(t);

    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < static_cast<int>(t.dim()); ++i)
        shape[4 - t.dim() + i] = static_cast<int>(t.size(i));

    KittensIPCPointerSet ptrs = broker.gather_ipc_ptrs(t);

    int device_ids[PGL::num_devices];
    uint64_t data_ptrs[PGL::num_devices];
    for (int i = 0; i < PGL::num_devices; i++) {
        device_ids[i] = i;
        data_ptrs[i] = reinterpret_cast<uint64_t>(ptrs.raw_ptrs_[i]);
    }

    TORCH_CHECK(data_ptrs[broker.local_rank_] == reinterpret_cast<uint64_t>(t.data_ptr()), 
                "Current tensor data pointer not found in KittensIPCPointerSet"); // sanity check

    return ::kittens::make_pgl<PGL>(device_ids, data_ptrs, shape[0], shape[1], shape[2], shape[3]);
}

template <kittens::ducks::pgl::all PGL>
static inline PGL tensor_to_pgl(const at::Tensor &t, const KittensIPCPointerSet &ptrs, const KittensBroker &broker) {
    TORCH_CHECK(PGL::num_devices == broker.local_world_size_, "Number of devices mismatch between PGL and KittensBroker");
    TORCH_CHECK(PGL::num_devices == ptrs.is_imported_.size(), "Number of devices mismatch between PGL and KittensIPCPointerSet");
    TORCH_CHECK(PGL::num_devices == ptrs.raw_ptrs_.size(), "Number of devices mismatch between PGL and KittensIPCPointerSet");
    TORCH_CHECK(!PGL::_INIT_MC, "PGL must be initialized with INIT_MC=false for multiprocess use");
    TORCH_CHECK(broker.local_rank_ == t.device().index(), "Current tensor device index mismatch with KittensBroker");

    tensor_check<PGL>(t);

    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < static_cast<int>(t.dim()); ++i)
        shape[4 - t.dim() + i] = static_cast<int>(t.size(i));

    int device_ids[PGL::num_devices];
    uint64_t data_ptrs[PGL::num_devices];
    for (int i = 0; i < PGL::num_devices; i++) {
        device_ids[i] = i;
        data_ptrs[i] = reinterpret_cast<uint64_t>(ptrs.raw_ptrs_[i]);
    }

    TORCH_CHECK(data_ptrs[broker.local_rank_] == reinterpret_cast<uint64_t>(t.data_ptr()), 
                "Current tensor data pointer not found in KittensIPCPointerSet"); // sanity check

    return ::kittens::make_pgl<PGL>(device_ids, data_ptrs, shape[0], shape[1], shape[2], shape[3]);
}

template <kittens::ducks::gl::all GL>
static inline GL make_fake_gl(const int batch, const int depth, const int rows, const int cols) {
    return ::kittens::make_gl<GL>(reinterpret_cast<uint64_t>(nullptr), batch, depth, rows, cols);
}

static inline void _device_check(const at::Tensor& first, const at::Tensor& second) {
    TORCH_CHECK(first.device() == second.device(), "All tensors must be on the same device");
}

template <typename T1, typename... Ts>
static inline void device_check(const T1& first, const Ts&... rest) {
    (_device_check(first, rest), ...);
}

template <typename Config>
concept static_grid = requires { Config::NUM_BLOCKS; };

template <typename Config>
concept static_block = requires { Config::NUM_THREADS; };

template <typename Config>
concept static_dynamic_shared_memory = requires { Config::DYNAMIC_SHARED_MEMORY; };

template <typename Config, typename Globals, auto Kernel>
static inline void launch_kernel(const Globals &G) {
    dim3 grid;
    if constexpr (static_grid<Config>)
        grid = dim3{Config::NUM_BLOCKS, 1, 1};
    else
        grid = G.grid();

    dim3 block;
    if constexpr (static_block<Config>)
        block = dim3{Config::NUM_THREADS, 1, 1};
    else
        block = G.block();

    int dynamic_shared_memory;
    if constexpr (static_dynamic_shared_memory<Config>)
        dynamic_shared_memory = static_cast<int>(Config::DYNAMIC_SHARED_MEMORY);
    else
        dynamic_shared_memory = G.dynamic_shared_memory();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if constexpr (Config::CLUSTER_SIZE <= 1) {
        cudaFuncSetAttribute(global_kernel_unclustered<Config, Globals, Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory);
        global_kernel_unclustered<Config, Globals, Kernel><<<grid, block, dynamic_shared_memory, stream>>>(G);
    } else {
        cudaFuncSetAttribute(global_kernel_clustered<Config, Globals, Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory);
        global_kernel_clustered<Config, Globals, Kernel><<<grid, block, dynamic_shared_memory, stream>>>(G);
    }
}

} // namespace py
} // namespace kittens
