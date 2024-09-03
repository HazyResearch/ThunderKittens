/**
 * @file
 * @brief Various utilities for group memory operations.
 */


template<int N=0> __device__ static inline void load_async_wait() { // for completing (non-TMA) async loads
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
    sync();
}