#pragma once

#include "kittens.cuh"
#include "../../common/common.cuh"
#include "../util.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {

template <typename config, typename globals>
__device__ void inline load_instructions(int *instruction, int instruction_index, const globals &g) {
    auto laneid = ::kittens::laneid();

    auto src_ptr = &g.instructions[kittens::coord<>{(int)(get_worker_id()), instruction_index, 0}];
    // static assert it's an int*
    static_assert(std::is_same<decltype(src_ptr), int *>::value, "src_ptr is not an int*");

    static_assert(config::INSTRUCTION_WIDTH <= 32);

    if (laneid < config::INSTRUCTION_WIDTH) {
        instruction[laneid] = src_ptr[laneid];
    }
}

template <typename config, typename globals>
__device__ void inline instruction_fetch_loop(const globals &g, ::kittens::prototype::vm::state<config> &kvms) {
    static_assert(config::INSTRUCTION_PIPELINE_STAGES <= 16, "This would be an absurd thing to do.");
    int num_iters = g.instructions.rows();
    for (kvms.instruction_index = 0, kvms.instruction_ring = 0; kvms.instruction_index < num_iters; kvms.instruction_index++, kvms.instruction_ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(kvms.instruction_ring)) {
        int phasebit = (kvms.instruction_index / config::INSTRUCTION_PIPELINE_STAGES - 1) & 1;
        if (kvms.instruction_index >= config::INSTRUCTION_PIPELINE_STAGES)
            wait(kvms.instruction_finished[kvms.instruction_ring], phasebit);
        load_instructions<config, globals>(&kvms.instruction()[0], kvms.instruction_index, g, kvms.instruction_arrived[kvms.instruction_ring]);
    }
}

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens
