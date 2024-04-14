#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/shared/shared.cuh"

namespace kittens {
namespace dsmem {

using barrier = uint64_t;

/**
 * @brief Waits at a dsmem barrier until the memory and sufficient threads have arrived.
 *
 * This function is used to synchronize threads at a barrier. Each thread waits at the barrier
 * until the local memory has arrived.
 *
 * @param bar Reference to the barrier variable.
 * @param kPhaseBit The phase bit used for the barrier.
 */
__device__ static inline void wait(barrier& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbar.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        ::
        "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

/**
 * @brief Sets the number of bytes expected at the barrier.
 *
 * This function is called by the first thread in the warp (laneid() == 0) to set the number of bytes
 * expected at the barrier. It converts the barrier pointer to a generic shared memory pointer and
 * uses inline assembly to set the expected number of bytes.
 *
 * @param bar Reference to the barrier variable.
 * @param bytes The number of bytes expected at the barrier.
 */
__device__ static inline void set_bytes(barrier& bar, uint32_t bytes) {
    if (laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile (
            "mbar.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(bar_ptr), "r"(bytes)
        );

    }
}


/**
 * @brief Initialize a distribute shared memory barrier
 *
 * If the template arguments are left blank, the user is expected to call set_bytes manually.
 * Alternatively, if a shared tile or shared vector type is passed, along with optional array
 * dimensions, the barrier will be automatically initialized with the correct transaction size, too.
 *
 * @tparam T the type of the shared memory object being passed. Defaults to kittens::ducks::default_type.
 * @tparam dims... Dimensions of the multidimensional array, if an array is being transferred. If blank, a single object is transferred.
 * @param[out] bar Reference to the barrier variable.
 * @param[in] tc The number of arriving threads the barrier should also wait for.
 */
template<typename T=ducks::default_type, int... dims>
__device__ static inline void init_bar(barrier& bar, int tc=1) {
    if (laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile (
            "mbar.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(bar_ptr), "r"(tc)
        );
    }
    // Now initialize the bar bytes
    if constexpr (ducks::st::all<T> || ducks::sv::all<T>) {
        set_bytes(bar, kittens::detail::transfer_bytes<T, dims...>::bytes);
    }
}

// Generic transfer
template<typename T>
__device__ static inline void distribute(T &dst_, T &src_, int cluster_size, int dst_idx, uint32_t size_bytes, barrier& bar) {
    if (laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        // **************************************************
        // load from src to dst in different threadblocks
        auto src = &src_;
        auto dst = &dst_;
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src)); 
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));  

        uint32_t neighbor_rank = dst_idx;

        // mapa instr = https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa 
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
        
        // cp.async instr = https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk 
        // copy src into dst in neighbor's cta
        asm volatile (
            "cp.async.bulk.shared::cluster.shared::cta.mbar::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :
            : "r"(neighbor_addr_dst), "r"(src_ptr), "r"(size_bytes), "r"(neighbor_addr_mbar)
            : "memory"
        );
    }
}

}
}