#pragma once

#include "../include/kittens.cuh"
#include "../common/common.cuh"

namespace kittens {
namespace prototype {
namespace lcsc {

template <typename config>
concept has_min_blocks_per_sm = requires { std::integral_constant<int, int(config::MIN_BLOCKS_PER_SM)>{}; };

template <typename config>
consteval int min_blocks_per_sm() {
    if constexpr(has_min_blocks_per_sm<config>)
        return config::MIN_BLOCKS_PER_SM;
    else
        return 1;
}

template <typename config, typename globals, typename lcsct>
__launch_bounds__(config::NUM_THREADS, min_blocks_per_sm<config>())
__global__ void main_kernel(const __grid_constant__ globals G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    const int num_comm_sms = G.num_comm_sms;
    const int num_comp_sms = gridDim.x - num_comm_sms;

    if (blockIdx.x >= num_comp_sms) {
        // Comm worker
        static_assert(sizeof(typename lcsct::comm_smem) <= config::DYNAMIC_SHARED_MEMORY, "Comm shared memory size exceeds dynamic shared memory limit");
        typename lcsct::comm_regs regs = {
            .worker_id = static_cast<int>(blockIdx.x) - num_comp_sms,
            .comm_workers = num_comm_sms,
            .comp_workers = num_comp_sms,
        };
        typename lcsct::comm_smem (&smem) = allocator.allocate<typename lcsct::comm_smem>();
        __shared__ typename lcsct::comm_sem sem;

        lcsct::comm_setup(G, sem, smem, regs);

        for (regs.task_id = regs.task_offset; regs.task_id < regs.num_tasks; regs.task_id += regs.task_stride)
            lcsct::communicator(G, sem, smem, regs);
    } else {
        // Comp worker
        static_assert(sizeof(typename lcsct::comp_smem) <= config::DYNAMIC_SHARED_MEMORY, "Comp shared memory size exceeds dynamic shared memory limit");
        typename lcsct::comp_regs regs = {
            .worker_id = static_cast<int>(blockIdx.x),
            .comm_workers = num_comm_sms,
            .comp_workers = num_comp_sms,
        };
        typename lcsct::comp_smem (&smem) = allocator.allocate<typename lcsct::comp_smem>();
        __shared__ typename lcsct::comp_sem sem;

        int warpgroup_id = warpgroup::groupid();
        int warp_id = warpgroup::warpid();
        int lane_id = warp::laneid();
        lcsct::comp_setup(G, sem, smem, regs);

        if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
            warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

            if (warp_id == 0 && lane_id == 0) {
                // Loader
                for (regs.task_id = regs.task_offset; regs.task_id < regs.num_tasks; regs.task_id += regs.task_stride)
                    lcsct::loader(G, sem, smem, regs);
            } else if (warp_id == 1 && lane_id == 0) {
                // Storer
                for (regs.task_id = regs.task_offset; regs.task_id < regs.num_tasks; regs.task_id += regs.task_stride)
                    lcsct::storer(G, sem, smem, regs);
            }
        } else {
            // Consumer
            warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

            for (regs.task_id = regs.task_offset; regs.task_id < regs.num_tasks; regs.task_id += regs.task_stride)
                lcsct::consumer(G, sem, smem, regs);
        }
    }
}

template <typename config>
concept static_grid = requires { config::NUM_BLOCKS; };

template <typename config>
concept static_block = requires { config::NUM_THREADS; };

template <typename config>
concept static_dynamic_shared_memory = requires { config::DYNAMIC_SHARED_MEMORY; };

template <typename config, typename globals, typename lcsct>
__host__ static inline void launch_kernel(const globals &G, cudaStream_t stream = 0) {
    dim3 grid;
    if constexpr (static_grid<config>)
        grid = dim3{config::NUM_BLOCKS, 1, 1};
    else
        grid = G.grid();

    dim3 block;
    if constexpr (static_block<config>)
        block = dim3{config::NUM_THREADS, 1, 1};
    else
        block = G.block();

    int dynamic_shared_memory;
    if constexpr (static_dynamic_shared_memory<config>)
        dynamic_shared_memory = static_cast<int>(config::DYNAMIC_SHARED_MEMORY);
    else
        dynamic_shared_memory = G.dynamic_shared_memory();

    CUDACHECK(cudaFuncSetAttribute(main_kernel<config, globals, lcsct>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
    main_kernel<config, globals, lcsct><<<grid, block, dynamic_shared_memory, stream>>>(G);
}

} // namespace lcsc
} // namespace prototype
} // namespace kittens