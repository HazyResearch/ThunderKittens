/**
 * @file
 * @brief Various utilities for group memory operations.
 */


template<int N=0> __device__ static inline void load_async_wait(int bar_id) { // for completing (non-TMA) async loads
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
    sync(bar_id);
}

__device__ static inline void arrive(barrier<N_WARPS> bar) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(bar.barrier_id), "n"(N_WARPS*WARP_THREADS) : "memory");
}
__device__ static inline void arrive_and_wait(barrier<N_WARPS> bar) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(bar.barrier_id), "n"(N_WARPS*WARP_THREADS) : "memory");
}