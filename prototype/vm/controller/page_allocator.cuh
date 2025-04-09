#pragma once

#include "kittens.cuh"
#include "../../common/common.cuh"
#include "../util.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {

template<typename config, typename globals>
struct page_allocator_op_dispatcher {
    template<typename op>
    struct dispatcher {
        template<int PAGE_COUNT> __device__ static inline int try_assign(::kittens::prototype::vm::state<config> &kvms, semaphore *arrived, semaphore *finished, int (&assignment)[config::PAGE_RING_SIZE], int assignment_ring, uint32_t assignment_counter, int &local_assignment_counter) {
            constexpr int membermask = 0xFFFFFFFF >> (32-PAGE_COUNT);
            int available = test_wait(finished[laneid()], 1); // managed_phase);
            uint32_t ballot;
            asm volatile("vote.sync.ballot.b32 %0, %1, %2;\n" : "=r"(ballot) : "r"(available), "r"(membermask));
            if(ballot != 0) {
                int next_assignment = __ffs(ballot);
                if(next_assignment == laneid()) {
                    assignment[assignment_ring] = next_assignment;
                    kittens::arrive(finished[laneid()], config::NUM_CONSUMER_WARPS); // Flip the phase so we can't use it again until another thread has marked it.
                    kittens::arrive(arrived[laneid()], config::NUM_CONSUMER_WARPS); // Flip the phase so we can't use it again until another thread has marked it.
                    asm volatile("atom.release.cta.shared::cta.inc.u32 _, [%0];\n" :: "r"(assignment_counter) : "memory");
                }
                local_assignment_counter++;
                return 1;
            }
            else {
                // No mini pages currently available to assign :(
                return 0;
            }
        }
        __device__ static inline void run(const globals &g, ::kittens::prototype::vm::state<config> &kvms) {
            int num_pages_to_allocate = op::num_pages(g, kvms);
            int num_mini_pages_to_allocate = op::num_mini_pages(g, kvms);
            int pages_allocated = 0, mini_pages_allocated = 0;
            constexpr int MAX_PAGES = config::NUM_PAGES > config::NUM_MINI_PAGES ? config::NUM_PAGES : config::NUM_MINI_PAGES;
            while(pages_allocated < num_pages_to_allocate || mini_pages_allocated < num_mini_pages_to_allocate) {
                if(mini_pages_allocated < num_mini_pages_to_allocate && (MAX_PAGES == config::NUM_MINI_PAGES || laneid() < config::NUM_MINI_PAGES)) {
                    mini_pages_allocated += try_assign<config::NUM_MINI_PAGES>(
                        kvms,
                        &kvms.mini_page_arrived[0],
                        &kvms.mini_page_finished[0],
                        kvms.mini_page_assignment,
                        kvms.mini_page_ring(),
                        kvms.mini_page_assignment_counter(),
                        kvms.mini_page_iter
                    );
                }
                if (pages_allocated < num_pages_to_allocate && (MAX_PAGES == config::NUM_PAGES || laneid() < config::NUM_PAGES)) {
                    pages_allocated += try_assign<config::NUM_PAGES>(
                        kvms,
                        &kvms.page_arrived[0],
                        &kvms.page_finished[0],
                        kvms.page_assignment,
                        kvms.page_ring(),
                        kvms.page_assignment_counter(),
                        kvms.page_iter
                    );
                }
                __nanosleep(20);
            }
        }
    };
};

template<typename config, typename globals, int end_thread, typename... ops> __device__ void inline page_allocator_loop(const globals &g, ::kittens::prototype::vm::state<config> &kvms) {
    constexpr uint32_t membermask = 0xFFFFFFFF >> (32-end_thread);
    int num_iters = g.instructions.rows();
    for(kvms.instruction_index = 0, kvms.instruction_ring = 0; kvms.instruction_index < num_iters; kvms.instruction_index++, kvms.instruction_ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(kvms.instruction_ring)) {
        // Wait for the instruction to be ready.
        wait(kvms.instruction_arrived[kvms.instruction_ring], (kvms.instruction_index/config::INSTRUCTION_PIPELINE_STAGES)&1);
        // Dispatch the instruction.
        kittens::prototype::vm::dispatch_op<
            config,
            globals,
            ::kittens::prototype::vm::state<config>,
            page_allocator_op_dispatcher<config, globals>::dispatcher,
            ops...
        >::run(kvms.instruction()[0], g, kvms);
        // We need to sync before we can mark this instruction as finished.
        asm volatile("bar.warp.sync %0;\n" :: "n"(membermask));
        // Wait for the instruction to be finished.
        if(laneid() == 0) {
#ifdef KVM_DEBUG
            printf("Thread %d (page allocator): arriving at instruction finished %d\n", threadIdx.x, kvms.instruction_ring);
#endif
            kittens::arrive(kvms.instruction_finished[kvms.instruction_ring]); // Single thread needs to arrive.
        }
    }

}

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens