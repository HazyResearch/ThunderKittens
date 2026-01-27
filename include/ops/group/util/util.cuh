/**
 * @file
 * @brief Utilities run by groups.
 */

#include "sync.cuh"
// tma.cuh and tma_cluster.cuh are included in group.cuh

/* CLC scheduler operations */

#ifdef KITTENS_BLACKWELL

struct clc {

/**
 * @brief Schedules a new threadblock. Must be called by a single thread in the entire CTA cluster.
 *        The caller must wait on the semaphore with tma::cluster::expect_bytes followed by tma::cluster::wait.
 *        The handle is multicasted to all CTAs in the cluster and signals the semaphore of all CTAs in the cluster.
 * @param h The CLC handle.
 * @param sem The semaphore that the caller will wait on.
 */
__device__ static inline void schedule(kittens::clc::handle &h, kittens::semaphore &sem) {
    if (laneid() == 0) {
        asm volatile("{clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&h.internal_value))), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&sem)))
            : "memory"
        );
    }
}

/**
 * @brief Queries the result of a schedule operation. Calling this again after failure is undefined behavior.
 * @param h The CLC handle.
 */
__device__ static inline kittens::clc::result query(kittens::clc::handle &h) {
    kittens::clc::result r;
    asm volatile(
        "{\n"
        ".reg .pred SUCCESS;\n"
        ".reg .b128 CLC_HANDLE;\n"
        "ld.shared.b128 CLC_HANDLE, [%4];\n"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 SUCCESS, CLC_HANDLE;\n"
        "selp.u32 %0, 1, 0, SUCCESS;\n"
        "@!SUCCESS bra.uni DONE;\n"
        "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%1, %2, %3, _}, CLC_HANDLE;\n"
        "fence.proxy.async.shared::cta;\n"
        "DONE:\n"
        "}"
        : "=r"(r.success), "=r"(r.x), "=r"(r.y), "=r"(r.z)
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&h.internal_value)))
        : "memory"
    );
    return r;
}

};

#endif

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)

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
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

/**
 * @brief Blocks until the primary kernel fully completes and flushes memory.
 */
__device__ static inline void wait() {
    cudaGridDependencySynchronize();
}

};

#endif
