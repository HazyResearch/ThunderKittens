

template <int NUM_DEVICES>
__device__ static inline void signal(
    const barrier_t<NUM_DEVICES> &barrier, const coord<ducks::default_type> &idx, const int dst_dev_idx, const int val
) {
    asm volatile("{red.release.sys.global.add.s32 [%0], %1;}" :: "l"(&barrier[dst_dev_idx][idx]), "r"(val) : "memory");
}

template <int NUM_DEVICES>
__device__ static inline void signal_all(
    const barrier_t<NUM_DEVICES> &barrier, const coord<ducks::default_type> &idx, const int val
) {
    asm volatile("{multimem.red.release.sys.global.add.s32 [%0], %1;}" :: "l"(barrier.mc_ptr_at(idx)), "r"(val) : "memory");
}

template <int NUM_DEVICES>
__device__ static inline void wait(
    const barrier_t<NUM_DEVICES> &barrier, const coord<ducks::default_type> &idx, const int dev_idx, const int expected
) {
    int val;
    do {
        asm volatile("{ld.relaxed.sys.global.s32 %0, [%1];}" : "=r"(val) : "l"(&barrier[dev_idx][idx]) : "memory");
    } while (val != expected);
}

template <int NUM_DEVICES>
__device__ static inline void barrier(
    const barrier_t<NUM_DEVICES> &barrier, const coord<ducks::default_type> &idx, const int dev_idx
) {
    signal_all(barrier, idx, 1);
    wait(barrier, idx, dev_idx, NUM_DEVICES);
    asm volatile("{red.release.sys.global.add.s32 [%0], %1;}" :: "l"(&barrier[dev_idx][idx]), "r"(-NUM_DEVICES) : "memory");
}
