#pragma once

#include "kittens.cuh"
#include "../../common/common.cuh"
#include "../util.cuh"

namespace kittens {
namespace prototype {
namespace vm {
namespace controller {

template<typename config, typename globals> struct semaphore_initializer_op_dispatcher {
    template<typename op>
    struct dispatcher {
        __device__ static inline int run(const globals &g, ::kittens::prototype::vm::state<config> &kvms) {
            return op::controller::init_semaphores(g, kvms);
        }
    };
};

template<typename config, typename globals, typename... ops> __device__ void inline semaphore_initializer_loop(const globals &g, ::kittens::prototype::vm::state<config> &kvms) {
    static_assert(config::INSTRUCTION_PIPELINE_STAGES == 2, "Need to be changed.");
    int num_iters = g.instructions.rows();
    int tic = 0;
    uint32_t semaphore_bitfield = 0x00000000;
    for(kvms.instruction_index = 0, kvms.instruction_ring = 0;
         kvms.instruction_index < num_iters;
         kvms.instruction_index++, kvms.instruction_ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(kvms.instruction_ring), tic=1-tic) {
        wait(kvms.instruction_arrived[kvms.instruction_ring], get_phasebit<0>(semaphore_bitfield, kvms.instruction_ring));
        int opcode = kvms.instruction()[0];
        int num_semaphores = dispatch_op<semaphore_initializer_op_dispatcher<config, globals>::dispatcher, ops...>::template run<int, config, globals, ::kittens::prototype::vm::state<config>>(
            opcode, g, kvms
        );
        arrive(kvms.semaphores_ready);
        arrive(kvms.instruction_finished[kvms.instruction_ring]); // We can also signal now that we, too, have done our part.
        wait(kvms.instruction_finished[kvms.instruction_ring], get_phasebit<0>(semaphore_bitfield, kvms.instruction_ring));
        for(int i = 0; i < num_semaphores; i++) {
            invalidate_semaphore(kvms.semaphores()[i]);
        }
        update_phasebit<0>(semaphore_bitfield, kvms.instruction_ring);
    }
}

} // namespace controller
} // namespace vm
} // namespace prototype
} // namespace kittens
