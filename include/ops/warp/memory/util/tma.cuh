#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

#include <cuda.h>
#include <iostream>

namespace kittens {
/**
 * @brief A namespace for all of ThunderKittens' TMA functionality.
*/
namespace tma {

/* ----------   Barrier functions for async load  ---------- */

/**
* @brief Sets the number of bytes expected at the barrier.
*
* This function sets the number of bytes expected at the barrier for the first thread in the warp.
* It converts the barrier pointer to a generic shared memory pointer and uses an inline assembly
* instruction to set the expected number of bytes.
*
* @param barrier Reference to the barrier variable.
* @param bytes The number of bytes expected at the barrier.
*/
__device__ static inline void expect_bytes(barrier& bar, uint32_t bytes) {
    if (::kittens::laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(bar_ptr), "r"(bytes));
    }
}
/**
* @brief Sets the number of bytes expected at the barrier.
*
* This function sets the number of bytes expected at the barrier for the first thread in the warp.
* It converts the barrier pointer to a generic shared memory pointer and uses an inline assembly
* instruction to set the expected number of bytes.
*
* @tparam T The type of the data to be stored at the barrier.
* @param barrier Reference to the barrier variable.
*/
template<typename T>
__device__ static inline void expect(barrier& bar) {
    expect_bytes(bar, sizeof(T));
}

/* ----------   Synchronization functions for async store  ---------- */

/**
 * @brief Commits previous asynchronous TMA stores to a group and performs them.
*/
__device__ static inline void store_commit_group() {
    if (::kittens::laneid() == 0) {
        asm volatile("cp.async.bulk.commit_group;");
    } 
}
/**
 * @brief Waits for previous committed TMA store groups to complete.
 *
 * @tparam N The maximum number of remaining TMA store groups. Defaults to 0.
*/
template <int N=0>
__device__ static inline void store_async_wait() {
    asm volatile (
        "cp.async.bulk.wait_group %0;"
        :
        : "n"(N)
        : "memory"
    );
    __syncwarp();
}

/* ----------   Cluster-scope operations  ---------- */

namespace cluster {

/**
* @brief Waits for the requested barrier phase, at cluster scope
*
* @param barrier Reference to the barrier variable.
* @param kPhaseBit The phase bit used for the barrier.
*/
__device__ static inline void wait(barrier& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

/**
* @brief Sets the number of bytes expected at the barrier, assuming a multicast instruction.
*
* This function sets the number of bytes expected at the barrier for the first thread in the warp.
* It converts the barrier pointer to a generic shared memory pointer and uses an inline assembly
* instruction to set the expected number of bytes.
*
* @param barrier Reference to the barrier variable.
* @param bytes The number of bytes expected at the barrier.
*/
__device__ static inline void expect_bytes(barrier& bar, uint32_t bytes, int dst_cta) {
    if (::kittens::laneid() == 0) {
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar)); 
        uint32_t neighbor_mbar_addr;
        asm volatile (
            "mapa.shared::cluster.u32  %0, %1, %2;\n"
            : "=r"(neighbor_mbar_addr)
            : "r"(mbar_addr), "r"(dst_cta)
        );

        asm volatile ("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
            :: "r"(neighbor_mbar_addr), "r"(bytes));
    }
}
/**
* @brief Sets the number of bytes expected at the barrier.
*
* This function sets the number of bytes expected at the barrier for the first thread in the warp.
* It converts the barrier pointer to a generic shared memory pointer and uses an inline assembly
* instruction to set the expected number of bytes.
*
* @tparam T The type of the data to be stored at the barrier.
* @param barrier Reference to the barrier variable.
*/
template<typename T>
__device__ static inline void expect(barrier& bar, int dst_cta) {
    expect_bytes(bar, sizeof(T), dst_cta);
}

/**
* @brief Arrives at a barrier in cluster scope.
*
* Marks a thread arrival at an mbarrier
*
* @param barrier Reference to the barrier variable.
* @param kPhaseBit The phase bit used for the barrier.
*/
__device__ static inline void arrive(barrier& bar, int dst_cta, uint32_t count=1) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar)); 
    uint32_t neighbor_mbar_addr;
    asm volatile (
        "mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(neighbor_mbar_addr)
        : "r"(mbar_addr), "r"(dst_cta)
    );
    asm volatile (
        "mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n"
        :
        : "r"(neighbor_mbar_addr), "r" (count)
        : "memory"
    );
}

// Generic transfer
__device__ static inline void store_async(void *dst, void *src, int cluster_size, int dst_cta, uint32_t size_bytes, barrier& bar) {
    if (laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t mbarrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        // **************************************************
        // load from src to dst in different threadblocks
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

        // mapa instr = https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa 
        // find dst addr in neighbor's cta
        uint32_t neighbor_addr_dst;
        asm volatile (
            "mapa.shared::cluster.u32  %0, %1, %2;\n"
            : "=r"(neighbor_addr_dst)
            : "r"(dst_ptr), "r"(dst_cta)
        );
        
        uint32_t neighbor_addr_mbarrier = mbarrier_ptr;
        asm volatile (
            "mapa.shared::cluster.u32  %0, %1, %2;\n"
            : "=r"(neighbor_addr_mbarrier)
            : "r"(mbarrier_ptr), "r"(dst_cta)
        );
        
        // cp.async instr = https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk 
        // copy src into dst in neighbor's cta
        asm volatile (
            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :
            : "r"(neighbor_addr_dst), "r"(src_ptr), "r"(size_bytes), "r"(neighbor_addr_mbarrier)
            : "memory"
        );
    }
}

// Templated transfer for convenience
template<typename T>
__device__ static inline void store_async(T &dst_, T &src_, int cluster_size, int dst_cta, barrier& bar) {
    constexpr uint32_t size_bytes = sizeof(T);
    store_async((void*)&dst_, (void*)&src_, cluster_size, dst_cta, size_bytes, bar);
}

} // namespace cluster
} // namespace tma
} // namespace kittens