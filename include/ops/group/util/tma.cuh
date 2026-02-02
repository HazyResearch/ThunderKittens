/**
 * @file
 * @brief Various utilities for group TMA memory operations.
 */

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

/* ------- Non-tensor TMA transfers ------- */

__device__ static inline void load_async(void *dst, void *src, uint32_t size_bytes, semaphore& bar) {
    if(laneid() == 0) {
        ::kittens::tma::load_async(dst, src, size_bytes, bar);
    }
}
template<typename T>
__device__ static inline void load_async(T &dst, T &src, uint32_t size_bytes, semaphore& bar) {
    load_async(reinterpret_cast<void*>(&dst), reinterpret_cast<void*>(&src), size_bytes, bar);
}

__device__ static inline void store_async(void *dst, void *src, uint32_t size_bytes) {
    if(laneid() == 0) {
        ::kittens::tma::store_async(dst, src, size_bytes);
    }
}
template<typename T>
__device__ static inline void store_async(T &dst, T &src, uint32_t size_bytes) {
    store_async(reinterpret_cast<void*>(&dst), reinterpret_cast<void*>(&src), size_bytes);
}
