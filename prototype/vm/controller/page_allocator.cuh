#pragma once

#include "kittens.cuh"
#include "../../common/common.cuh"
#include "../util.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {

template<typename config, typename globals> struct page_allocator_op_dispatcher {
    template<typename op>
    struct dispatcher {
        __device__ static inline int run(const globals &g, typename config::instruction_t &instruction, int &query) {
            return op::controller::release_lid(g, instruction, query);
        }
    };
};

template<typename config, typename globals, typename... ops> __device__ void inline page_allocator_loop(const globals &g, ::kittens::prototype::vm::state<config> &kvms) {
    static_assert(config::INSTRUCTION_PIPELINE_STAGES <= 16, "This would be an absurd thing to do.");
    constexpr uint32_t membermask = 0xFFFFFFFF >> (32-config::NUM_PAGES);
    int num_iters = g.instructions.rows();
    for (kvms.instruction_index = 0, kvms.instruction_ring = 0;
         kvms.instruction_index < num_iters;
         kvms.instruction_index++, kvms.instruction_ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(kvms.instruction_ring)) {
        
        int phasebit = (kvms.instruction_index / config::INSTRUCTION_PIPELINE_STAGES - 1) & 1;
        if (kvms.instruction_index >= config::INSTRUCTION_PIPELINE_STAGES) wait(kvms.instruction_finished[kvms.instruction_ring], phasebit);

        int next_pid;
        if(kvms.instruction_index == 0) next_pid = laneid();
        else {
            int last_instruction_ring = (kvms.instruction_ring+config::INSTRUCTION_PIPELINE_STAGES-1)%config::INSTRUCTION_PIPELINE_STAGES;
            wait(kvms.instruction_arrived[last_instruction_ring], ((kvms.instruction_index - 1) / config::INSTRUCTION_PIPELINE_STAGES) & 1);
            int lane = laneid();
            int opcode = kvms.all_instructions[last_instruction_ring].instructions[0];
            int lid = dispatch_op<page_allocator_op_dispatcher<config, globals>::dispatcher, ops...>::template run<int, config, globals, config::instruction_t, int>(
                opcode, g, kvms.all_instructions[last_instruction_ring].instructions, lane
            );
            next_pid = kvms.all_instructions[last_instruction_ring].pid_order[lid];
        }
        kvms.pid_order()[laneid()] = next_pid;
        asm volatile("bar.warp.sync %0;\n" :: "n"(membermask));
        if(laneid() == 0) arrive(kvms.instruction_arrived[kvms.instruction_ring], 1);
    }
}

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens