#pragma once

#include "kittens.cuh"
#include "../../common/common.cuh"
#include "../util.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {

template<typename config, int PAGE_COUNT> __device__ static inline void try_assign(::kittens::prototype::vm::state<config> &kvms, semaphore *arrived, semaphore *finished, int (&assignment)[config::PAGE_RING_SIZE], int assignment_ring, uint32_t assignment_counter, int &local_assignment_counter, uint32_t shift_mask) {
    constexpr int membermask = 0xFFFFFFFF >> (32-PAGE_COUNT);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&finished[laneid()]));
    uint32_t page_ballot, my_page_available;
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%2], %3;\n"
        "selp.u32 %1, 1, 0, P1;\n"
        "vote.sync.ballot.b32 %0, P1, %4;\n"
        "}\n"
        : "=r"(page_ballot), "=r"(my_page_available)
        : "r"(mbar_ptr), "n"(0), "r"(membermask)
    );
// #ifdef KVM_DEBUG
    // if(laneid() == 0) printf("(page allocator): page_ballot = %d\n", page_ballot);
    // asm volatile("bar.warp.sync %0;\n" :: "n"(membermask));
// #endif
    if(page_ballot != 0) {
        int assignment_index = __popc(page_ballot & shift_mask); // How many ones are below us?
        if (my_page_available) {
            int local_assignment_ring = (assignment_ring + assignment_index) % config::PAGE_RING_SIZE;
            assignment[local_assignment_ring] = laneid();
            kittens::arrive(finished[laneid()], config::NUM_CONSUMER_WARPS); // Flip the phase to 1 so we can't use it again until another thread has marked it.
            kittens::arrive(arrived[laneid()], config::NUM_CONSUMER_WARPS); // Flip the phase to 1 so we can't use it again until another thread has marked it.
        }
        asm volatile("bar.warp.sync %0;\n" :: "n"(membermask)); // need to sync so that the arrival on that semaphore is visible to all threads.
        int num_pages_assigned = __popc(page_ballot);
        local_assignment_counter += num_pages_assigned;
        if(laneid() == 0) {
            asm volatile("st.shared.u32 [%0], %1;\n" : : "r"(assignment_counter), "r"(local_assignment_counter) : "memory");
        }
    }
    else {
        // No mini pages currently available to assign :(
        return;
    }
}

template<typename config, typename globals, int end_thread, typename... ops> __device__ void inline page_allocator_loop(const globals &g, ::kittens::prototype::vm::state<config> &kvms) {
    constexpr uint32_t membermask = 0xFFFFFFFF >> (32-end_thread);
    int num_iters = g.instructions.rows();
    constexpr int MAX_PAGES = config::NUM_PAGES > config::NUM_MINI_PAGES ? config::NUM_PAGES : config::NUM_MINI_PAGES;
    uint32_t shift_mask = 0xFFFFFFFFu >> (32-laneid());
    while(test_wait(kvms.cleanup, 1)) {
        constexpr int CLEANUP_POLL_INTERVAL = 20;
        for(int i = 0; i < CLEANUP_POLL_INTERVAL; i++) {
            if(MAX_PAGES == config::NUM_MINI_PAGES || laneid() < config::NUM_MINI_PAGES) {
                try_assign<config, config::NUM_MINI_PAGES>(
                    kvms,
                    &kvms.mini_page_arrived[0],
                    &kvms.mini_page_finished[0],
                    kvms.mini_page_assignment,
                    kvms.mini_page_ring(),
                    kvms.mini_page_assignment_counter(),
                    kvms.mini_page_iter,
                    shift_mask
                );
            }
            if(MAX_PAGES == config::NUM_PAGES || laneid() < config::NUM_PAGES) {
                try_assign<config, config::NUM_PAGES>(
                    kvms,
                    &kvms.page_arrived[0],
                    &kvms.page_finished[0],
                    kvms.page_assignment,
                    kvms.page_ring(),
                    kvms.page_assignment_counter(),
                    kvms.page_iter,
                    shift_mask
                );
            }
            __nanosleep(20);
        }
    }
    asm volatile("bar.warp.sync %0;\n" :: "n"(membermask));
}

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens