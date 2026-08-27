/**
 * @file
 * @brief Various utilities for group cluster-wide TMA memory operations.
 */

/**
* @brief Waits for the requested semaphore phase, at cluster scope
*
* @param bar Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
template <memory_model M = memory_model::ACQUIRE>
__device__ static inline void wait(semaphore& bar, int kPhaseBit) {
    ::kittens::tma::cluster::wait<M>(bar, kPhaseBit);
}

template <memory_model M = memory_model::ACQUIRE>
__device__ static inline bool try_wait(semaphore &bar, int kPhaseBit) {
    return ::kittens::tma::cluster::try_wait<M>(bar, kPhaseBit);
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
* @param bar Reference to the semaphore variable.
* @param bytes The number of bytes expected at the semaphore.
*/
template <memory_model M = memory_model::RELEASE>
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::expect_bytes<M>(bar, bytes);
    }
}
template <memory_model M = memory_model::RELEASE>
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes, int dst_cta) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::expect_bytes<M>(bar, bytes, dst_cta);
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
template<memory_model M = memory_model::RELEASE, typename T, typename... args>
__device__ static inline void expect(semaphore& bar, const T& _1, const args&... _2) {
    expect_bytes<M>(bar, size_bytes<T, args...>);
}

/**
* @brief Arrives at a semaphore in cluster scope.
*
* Marks a thread arrival at an mbarrier
*
* @param bar Reference to the semaphore variable.
* @param dst_cta The destination CTA index.
* @param count The count value for the arrival.
*/
template <memory_model M = memory_model::RELEASE>
__device__ static inline void arrive(semaphore& bar, int dst_cta, uint32_t count=1) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::arrive<M>(bar, dst_cta, count);
    }
}

/* ------- Non-tensor TMA transfers ------- */

__device__ static inline void load_async(void *dst, void *src, uint32_t size_bytes, semaphore& bar, uint16_t cta_mask) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::load_async(dst, src, size_bytes, bar, cta_mask);
    }
}
template<typename T>
__device__ static inline void load_async(T &dst, T &src, uint32_t size_bytes, semaphore& bar, uint16_t cta_mask) {
    load_async(reinterpret_cast<void*>(&dst), reinterpret_cast<void*>(&src), size_bytes, bar, cta_mask);
}
