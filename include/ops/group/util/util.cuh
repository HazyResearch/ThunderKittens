/**
 * @file
 * @brief Utilities run by groups.
 */

#include "sync.cuh"
// tma.cuh and tma_cluster.cuh are included in group.cuh

/* CLC scheduler operations */

#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)

struct clc {

/**
 * @brief Schedules a new threadblock. Must be called by a single thread in the entire CTA cluster.
 *        The caller must wait on the semaphore with tma::cluster::expect_bytes followed by tma::cluster::wait.
 *        The handle is multicasted to all CTAs in the cluster and signals the semaphore of all CTAs in the cluster.
 * @param h The CLC handle.
 * @param sem The semaphore that the caller will wait on.
 */
__device__ static inline void schedule(kittens::clc::handle &h, kittens::semaphore &sem) {
    if (laneid() == 0)
        kittens::clc::schedule(h, sem);
}

/**
 * @brief Queries the result of a schedule operation. Calling this again after failure is undefined behavior.
 * @param h The CLC handle.
 */
__device__ static inline kittens::clc::result query(kittens::clc::handle &h) {
    return kittens::clc::query(h);
}

};

#endif

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)

/**
 * @brief Programmatic Dependent Kernel Launch (PDL) utilities. Available on Hopper and later.
 *
 * PDL allows partial overlap between two consecutive kernels in the same stream.
 *
 * @note The secondary kernel must be launched with `cudaLaunchAttributeProgrammaticStreamSerialization`
 *       attribute and `programmaticStreamSerializationAllowed` set to 1.
 */
struct pdl {

/**
 * @brief Signals that a primary kernel has completed its dependent work, enabling a secondary kernel to launch.
 *
 * @note The secondary kernel will only launch when all threadblocks in the primary kernel have called this function.
 *       If a threadblock does not call this, the arrival is implicitly triggered at threadblock exit.
 * @note This does not guarantee memory visibility. For memory visibility, the secondary kernel must call wait().
 */
__device__ static inline void arrive() {
    if (laneid() == 0) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

/**
 * @brief Blocks until the primary kernel fully completes and flushes memory.
 */
__device__ static inline void wait() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

};

#endif
