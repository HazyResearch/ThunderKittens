#pragma once

#include "../../common/common.cuh"
#include "../../types/shared/shared.cuh"

namespace kittens {
namespace dsmem {

/**
 * @brief Distributes a tile of data across shared memory in different thread blocks.
 *
 * @param dst_ The destination shared memory tile.
 * @param src_ The source shared memory tile.
 * @param cluster_size The size of the cluster of thread blocks.
 * @param dst_idx The index of the destination thread block.
 * @param size_bytes The size of the data in bytes.
 * @param barrier The barrier used for synchronization.
 */
template<int height, int width, st_layout layout>
__device__ static inline void tile_distribute_smem(st<bf16, height, width, layout> &dst_, st<bf16, height, width, layout> &src_, int cluster_size, int dst_idx, uint32_t size_bytes, uint64_t& barrier) 
{
    if (threadIdx.x == 0) {
        void const* const ptr = &barrier;
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        // load from src to dst in different threadblocks
        auto src = &src_;
        auto dst = &dst_;
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src)); 
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));  

        uint32_t neighbor_rank = dst_idx;

        // find dst addr in neighbor's cta
        uint32_t neighbor_addr_dst = dst_ptr;
        asm volatile (
            "mapa.shared::cluster.u32  %0, %1, %2;\n"
            : "=r"(neighbor_addr_dst)
            : "r"(dst_ptr), "r"(neighbor_rank)
        );

        uint32_t neighbor_addr_mbar = mbar_ptr;
        asm volatile (
            "mapa.shared::cluster.u32  %0, %1, %2;\n"
            : "=r"(neighbor_addr_mbar)
            : "r"(mbar_ptr), "r"(neighbor_rank)
        );

        // copy src into dst in neighbor's cta
        asm volatile (
            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :
            : "r"(neighbor_addr_dst), "r"(src_ptr), "r"(size_bytes), "r"(neighbor_addr_mbar)
            : "memory"
        );
    }
}

/**
 * @brief Waits for the distribution of shared memory tiles to complete.
 *
 * @param barrier The barrier used for synchronization.
 * @param kPhaseBit The phase bit used for the mbarrier.
 */
__device__ static inline void distribution_wait(uint64_t& barrier, int kPhaseBit) {
    void const* const ptr = &barrier;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

/**
 * @brief Initializes a barrier for shared memory tile distribution.
 *
 * @param barrier The barrier to initialize.
 * @param tc The thread count for the barrier.
 */
__device__ static inline void init_barrier(uint64_t& barrier, int tc) {
    if (threadIdx.x == 0) {
        void const* const ptr = &barrier;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(bar_ptr), "r"(tc)
        );
    }
}

/**
 * @brief Sets the expected transaction bytes for a barrier.
 *
 * @param barrier The barrier to set.
 * @param bytes The expected transaction bytes.
 */
__device__ static inline void set_barrier_bytes(uint64_t& barrier, uint32_t bytes) {
    if (threadIdx.x == 0) {
        void const* const ptr = &barrier;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile (
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(bar_ptr), "r"(bytes)
        );

    }
}

}
}
