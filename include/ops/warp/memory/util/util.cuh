/**
 * @file
 * @brief General memory utilities not specialized for either tiles or vectors.
 */

#pragma once

namespace kittens {

/* ----------   Generic (non-Hopper specific) barrier functions  ---------- */

using barrier = uint64_t;

/**
 * @brief Initializes a synchronization barrier with a transaction count and sets the expected number of bytes.
 *
 * This function sets up a barrier that is used to synchronize threads within a block during asynchronous operations.
 * It initializes the barrier with a thread count barrier.
 *
 * Additionally, if it is given a shared tile type, it will also call `set_bytes` to prepare for the memory transaction.
 *
 * @param[out] barrier The barrier variable to initialize.
 * @param[in] tc The thread counter for the barrier.
 */
__device__ static inline void init_barrier(barrier& bar, int thread_count, int transaction_count) {
    if (::kittens::laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(bar_ptr), "r"(thread_count+transaction_count)
        );
    }
}

/**
* @brief Arrives at a barrier.
*
* Marks a thread arrival at an mbarrier
*
* @param barrier Reference to the barrier variable.
* @param kPhaseBit The phase bit used for the barrier.
*/
__device__ static inline void arrive(barrier& bar) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(1)
        : "memory"
    );
}

/**
* @brief Waits for the requested barrier phase.
*
* @param barrier Reference to the barrier variable.
* @param kPhaseBit The phase bit used for the barrier.
*/
__device__ static inline void wait(barrier& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

#ifdef KITTENS_HOPPER
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
#else
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "nanosleep.u32 5;\n" // wait a few nanoseconds on pre-Hopper architectures to save instruction issue slots
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
#endif
}

__device__ static inline void arrive_and_wait(barrier& bar, int kPhaseBit) {
    arrive(bar);
    wait(bar, kPhaseBit);
}

}

#ifdef KITTENS_HOPPER
#include "tma.cuh"
#endif