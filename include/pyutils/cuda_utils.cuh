#pragma once

#include <stdexcept>
#include <string>

#include "kittens.cuh"

namespace kittens {
namespace detail {

__host__ inline void throw_if_cuda_error(cudaError_t error, const char *operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + " failed: " +
            cudaGetErrorName(error) + " (" + std::to_string(static_cast<int>(error)) +
            "): " + cudaGetErrorString(error));
}

template <typename Config>
concept has_min_blocks_per_sm = requires { std::integral_constant<int, int(Config::MIN_BLOCKS_PER_SM)>{}; };

template <typename Config>
__host__ consteval int min_blocks_per_sm() {
    if constexpr (has_min_blocks_per_sm<Config>)
        return Config::MIN_BLOCKS_PER_SM;
    else
        return 1;
}

template <typename Config, typename Globals, auto Kernel>
__global__
__launch_bounds__(Config::NUM_THREADS, min_blocks_per_sm<Config>())
void global_kernel(const __grid_constant__ Globals G) {
    Kernel(G);
}

template <typename Config>
concept static_grid = requires { Config::NUM_BLOCKS; };

template <typename Config>
concept static_block = requires { Config::NUM_THREADS; };

template <typename Config>
concept static_dynamic_shared_memory = requires { Config::DYNAMIC_SHARED_MEMORY; };

template <typename Config>
concept has_pdl_config = requires { { Config::USE_PDL } -> std::convertible_to<bool>; };
template <typename Config>
inline constexpr bool use_pdl = false;
template <typename Config> requires has_pdl_config<Config>
inline constexpr bool use_pdl<Config> = Config::USE_PDL;

template <typename Config, typename Globals, auto Kernel>
__host__ static inline void launch_kernel(const Globals &G, cudaStream_t stream) {
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

#if defined(KITTENS_SM90)
    static_assert(Config::CLUSTER_SIZE <= 8, "Cluster size must be less than or equal to 8 for Hopper");
#elif defined(KITTENS_SM10X) || defined(KITTENS_SM120)
    static_assert(Config::CLUSTER_SIZE <= 16, "Cluster size must be less than or equal to 16 for Blackwell");
    if constexpr (Config::CLUSTER_SIZE > 8)
        throw_if_cuda_error(cudaFuncSetAttribute(global_kernel<Config, Globals, Kernel>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1), "cudaFuncSetAttribute");
#endif
    if (dynamic_shared_memory > 0)
        throw_if_cuda_error(cudaFuncSetAttribute(global_kernel<Config, Globals, Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory), "cudaFuncSetAttribute");

    if constexpr (Config::CLUSTER_SIZE <= 1) {
        LaunchConfig<false, use_pdl<Config>> launch_config(grid, block, dynamic_shared_memory, stream);
        throw_if_cuda_error(cudaLaunchKernelEx(launch_config, global_kernel<Config, Globals, Kernel>, G), "cudaLaunchKernelEx");
    } else {
        LaunchConfig<true, use_pdl<Config>> launch_config(grid, block, dynamic_shared_memory, stream, Config::CLUSTER_SIZE);
        throw_if_cuda_error(cudaLaunchKernelEx(launch_config, global_kernel<Config, Globals, Kernel>, G), "cudaLaunchKernelEx");
    }
}

} // namespace detail
} // namespace kittens
