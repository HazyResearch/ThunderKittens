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

namespace detail {

// Concepts for tiles

template<typename T> concept st_type_2d_tma_layout = (
    ducks::st::all<T> && 
    (
        std::is_same_v<typename T::layout, ducks::st_layout::naive> || 
        std::is_same_v<typename T::layout, ducks::st_layout::tma_swizzle>
    )
);
template<typename T> concept st_type_wgmma_row_layout = (
    ducks::st::all<T> && std::is_same_v<typename T::layout, ducks::st_layout::wgmma_row_0b>
);
template<typename T> concept st_type_wgmma_col_t_layout = (
    ducks::st::all<T> && std::is_same_v<typename T::layout, ducks::st_layout::wgmma_col_t_0b>
);
template<typename T> concept st_type_tma_layout = (
    st_type_2d_tma_layout<T> || st_type_wgmma_row_layout<T> || st_type_wgmma_col_t_layout<T>
);

}; 

using barrier = uint64_t;

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
__device__ static inline void set_bytes(barrier& bar, uint32_t bytes) {
    if (::kittens::laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(bar_ptr), "r"(bytes));
    }
}
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
template<typename T=ducks::default_type, int... dims>
__device__ static inline void init_barrier(barrier& bar, int tc=1) {
    static_assert(detail::st_type_tma_layout<T> || ducks::sv::all<T> || std::is_same_v<T, ducks::default_type>);
    if (::kittens::laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile ("mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(bar_ptr), "r"(tc));

        if constexpr (detail::st_type_tma_layout<T> || ducks::sv::all<T>) {
            set_bytes(bar, kittens::detail::transfer_bytes<T, dims...>::bytes); // set barrier bytes automatically
        }
    }
}

/**
* @brief Arrives at the barrier and waits for all threads to arrive.
*
* This function is used to synchronize threads at a barrier. Each thread arrives at the barrier
* and waits until all threads have arrived. The function uses inline assembly to perform the
* barrier wait operation.
*
* @param barrier Reference to the barrier variable.
* @param kPhaseBit The phase bit used for the barrier.
*/
__device__ static inline void arrive_and_wait(barrier& bar, int kPhaseBit) {
    void const* const ptr = &bar;
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

} // namespace tma
} // namespace kittens