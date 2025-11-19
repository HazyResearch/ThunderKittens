/**
 * @file
 * @brief Various utilities for group TMA memory operations.
 */

namespace tma {

/* ----------   Barrier functions for async load  ---------- */

/**
* @brief Sets the number of bytes expected at the semaphore.
*
* This function sets the number of bytes expected at the semaphore for the first thread in the warp.
* It converts the semaphore pointer to a generic shared memory pointer and uses an inline assembly
* instruction to set the expected number of bytes.
*
* @param semaphore Reference to the semaphore variable.
* @param bytes The number of bytes expected at the semaphore.
*/
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes) {
    if(laneid() == 0) {
        ::kittens::tma::expect_bytes(bar, bytes);
    }
}
/**
* @brief Sets the number of bytes expected at the semaphore.
*
* This function sets the number of bytes expected at the mbarrier before the transaction arrives.
*/
template<typename T, typename... args>
__device__ static inline void expect(semaphore& bar, const T& _1, const args&... _2) {
    expect_bytes(bar, size_bytes<T, args...>);
}

/* ----------   Synchronization functions for async store  ---------- */

/**
 * @brief Commits previous asynchronous TMA stores to a group and performs them.
*/
__device__ static inline void store_commit_group() {
    asm volatile("cp.async.bulk.commit_group;");
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
}
/**
 * @brief Waits for previous committed TMA store groups to finish reading from shared memory.
 *
 * @tparam N The maximum number of remaining TMA store groups. Defaults to 0.
*/
template <int N=0>
__device__ static inline void store_async_read_wait() {
    asm volatile (
        "cp.async.bulk.wait_group.read %0;"
        :
        : "n"(N)
        : "memory"
    );
}

namespace cluster {

/**
* @brief Waits for the requested semaphore phase, at cluster scope
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void wait(semaphore& bar, int kPhaseBit) {
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
* @brief Sets the number of bytes expected at the semaphore, assuming a multicast instruction.
*
* This function sets the number of bytes expected at the semaphore for the first thread in the warp.
* It converts the semaphore pointer to a generic shared memory pointer and uses an inline assembly
* instruction to set the expected number of bytes.
* 
* It's worth being aware that this function is particularly necessary for multicast loads, and
* distributed shared memory can actually be done with a normal tma::expect followed by wait. See
* the unit tests of dsmem for an example.
*
* @param semaphore Reference to the semaphore variable.
* @param bytes The number of bytes expected at the semaphore.
*/
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes, int dst_cta) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::expect_bytes(bar, bytes, dst_cta);
    }
}
/**
* @brief Sets the number of bytes expected at the semaphore.
*
* This function sets the number of bytes expected at the semaphore for the first thread in the warp.
* It converts the semaphore pointer to a generic shared memory pointer and uses an inline assembly
* instruction to set the expected number of bytes.
*
* @tparam T The type of the data to be stored at the semaphore.
* @param semaphore Reference to the semaphore variable.
*/
/**
* @brief Sets the number of bytes expected at the semaphore.
*
* This function sets the number of bytes expected at the mbarrier before the transaction arrives.
*/
template<typename T, typename... args>
__device__ static inline void expect(semaphore& bar, int dst_cta, const T& _1, const args&... _2) {
    expect_bytes(bar, size_bytes<T, args...>, dst_cta);
}

/**
* @brief Arrives at a semaphore in cluster scope.
*
* Marks a thread arrival at an mbarrier
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void arrive(semaphore& bar, int dst_cta, uint32_t count=1) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::arrive(bar, dst_cta, count);
    }
}

// Generic transfer
__device__ static inline void store_async(void *dst, void *src, int dst_cta, uint32_t size_bytes, semaphore& bar) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::store_async(dst, src, dst_cta, size_bytes, bar);
    }
}

// Templated transfer for convenience
template<typename T>
__device__ static inline void store_async(T &dst_, T &src_, int dst_cta, semaphore& bar) {
    store_async((void*)&dst_, (void*)&src_, dst_cta, size_bytes<T>, bar);
}

} // namespace cluster
} // namespace tma
