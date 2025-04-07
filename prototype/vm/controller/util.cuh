#pragma once

#include "kittens.cuh"
#include "../common/common.cuh"
#include "templates.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {

template<typename config> struct state {
    using instruction_array_t = int[config::INSTRUCTION_PIPELINE_STAGES][config::INSTRUCTION_WIDTH];
    using timing_array_t = int[config::INSTRUCTION_PIPELINE_STAGES][config::TIMING_EVENTS];
    using instruction_semaphore_array_t = kittens::semaphore[config::INSTRUCTION_PIPELINE_STAGES];
    instruction_array_t &instructions;
    timing_array_t &timings;
    instruction_semaphore_array_t &instruction_arrived, &instruction_finished;
    int task_iter, ring;

    // Used only by page allocators.
    using page_semaphore_array_t = kittens::semaphore[config::NUM_PAGES];
    using mini_page_semaphore_array_t = kittens::semaphore[config::NUM_MINI_PAGES];

    page_semaphore_array_t &page_arrived, &page_finished;
    mini_page_semaphore_array_t &mini_page_arrived, &mini_page_finished;

    using page_assignment_array_t = int[config::PAGE_RING_SIZE];
    page_assignment_array_t &page_assignment, &mini_page_assignment;
    int page_ring, mini_page_ring; // Where in the controller assignments we are.
    uint32_t page_assignment_counter, mini_page_assignment_counter; // this is a shared memory address incremented as pages are assigned.

    int managed_index; // , manged_page_phase, managed_mini_page_phase; // Which page is this thread responsible for?
};

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens