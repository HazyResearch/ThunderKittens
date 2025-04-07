#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"
#include "templates.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {


template<typename config, typename globals, typename op>
struct page_allocator_op_dispatcher {
    template<int PAGE_COUNT> __device__ static inline int try_assign(state<config> &cs, semaphore *arrived, semaphore *ready, int &managed_phase, uint32_t (&assignment)[config::PAGE_RING_SIZE], int &assignment_ring, uint32_t assignment_counter) {
        constexpr int membermask = 0xFFFFFFFF >> (32-PAGE_COUNT);
        int available = test_wait(ready[cs.managed_index], 1); // managed_phase);
        uint32_t ballot;
        asm volatile("vote.sync.ballot.b32 %0, %1, %2;\n" : "=r"(ballot) : "r"(available), "r"(membermask));
        if(ballot != 0) {
            int next_assignment = __ffs(ballot);
            if(next_assignment == cs.managed_index) {
                assignment[assignment_ring] = next_assignment;
                kittens::arrive(ready[cs.managed_index], config::NUM_CONSUMER_WARPS); // Flip the phase so we can't use it again until another thread has marked it.
                kittens::arrive(arrived[cs.managed_index], config::NUM_CONSUMER_WARPS); // Flip the phase so we can't use it again until another thread has marked it.
                asm volatile("atom.release.cta.shared::cta.inc.u32 _, [%0];\n" :: "r"(assignment_counter) : "memory");
                // managed_phase ^= 1; // Flip the desired phase so we can't use it again until another thread has marked it.
            }
            assignment_ring = ring_advance<config::PAGE_RING_SIZE>(assignment_ring); // Advance the ring of where we are in assignments.
            return 1;
        }
        else {
            // No mini pages currently available to assign :(
            return 0;
        }
    }
    __device__ static inline void run(const globals &g, state<config> &cs) {
        int num_pages_to_allocate = op::num_pages(cs.instruction[cs.ring], g);
        int num_mini_pages_to_allocate = op::num_mini_pages(cs.instruction[cs.ring], g);
        int pages_allocated = 0, mini_pages_allocated = 0;
        constexpr int MAX_PAGES = config::NUM_PAGES > config::NUM_MINI_PAGES ? config::NUM_PAGES : config::NUM_MINI_PAGES;
        while(pages_allocated < num_pages_to_allocate || mini_pages_allocated < num_mini_pages_to_allocate) {
            if(mini_pages_allocated < num_mini_pages_to_allocate && (MAX_PAGES == config::NUM_MINI_PAGES || cs.managed_index < config::NUM_MINI_PAGES)) {
                mini_pages_allocated += try_assign<config::NUM_MINI_PAGES>(
                    cs,
                    cs.mini_page_arrived,
                    cs.mini_page_ready,
                    cs.managed_mini_page_phase,
                    cs.mini_page_assignment,
                    cs.mini_page_ring,
                    cs.mini_page_assignment_counter
                );
            }
            if (pages_allocated < num_pages_to_allocate && (MAX_PAGES == config::NUM_PAGES || cs.managed_index < config::NUM_PAGES)) {
                pages_allocated += try_assign<config::NUM_PAGES>(
                    cs,
                    cs.page_arrived,
                    cs.page_ready,
                    cs.managed_page_phase,
                    cs.page_assignment,
                    cs.page_ring,
                    cs.page_assignment_counter
                );
            }
            __nanosleep(20);
        }
    }
};

template<typename config, typename globals, int end_thread, typename... ops> __device__ void inline page_allocator_loop(const globals &g, state<config> &cs) {
    constexpr uint32_t membermask = 0xFFFFFFFF >> (32-end_thread);
    int num_iters = g.instructions.rows();
    cs.managed_index = laneid();
    for(cs.task_iter = 0, cs.ring = 0; cs.task_iter < num_iters; cs.task_iter++, cs.ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(cs.ring)) {
        // Wait for the instruction to be ready.
        wait(cs.instruction_arrived[cs.ring], (cs.task_iter/cs.ring)&1);
        // Dispatch the instruction.
        kittens::prototype::vm::dispatch_op<config, globals, state<config>, page_allocator_op_dispatcher<config>, ops...>::run(cs.instruction[cs.ring][0], g, cs);
        // We need to sync before we can mark this instruction as finished.
        asm volatile("bar.warp.sync %0;\n" :: "n"(membermask));
        // Wait for the instruction to be finished.
        if(cs.managed_index == 0) kittens::arrive(cs.instruction_finished[cs.ring]); // Single thread needs to arrive.
    }

}

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens