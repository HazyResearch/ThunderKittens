#pragma once

#include "kittens.cuh"
#include "../../common/common.cuh"
#include "../util.cuh"
#include "instruction_fetch.cuh"
#include "timings_store.cuh"
#include "semaphore_constructor.cuh"
#include "page_allocator.cuh"

namespace kittens
{
    namespace prototype
    {
        namespace vm
        {
            namespace controller
            {

                template <typename config, typename globals, typename... ops>
                __device__ void main_loop(const globals &g, ::kittens::prototype::vm::state<config> &kvms)
                {
                    if (kittens::laneid() > 0)
                    {
                        return;
                    }

                    int num_iters = g.instructions.rows();
                    int num_semaphores[config::INSTRUCTION_PIPELINE_STAGES];

                    for (kvms.instruction_index = 0, kvms.instruction_ring = 0;
                         kvms.instruction_index < num_iters;
                         kvms.instruction_index++, kvms.instruction_ring = ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(kvms.instruction_ring))
                    {

                        // Step 0. if the slot was used in the previous iteration, wait for the previous instruction to complete & invalidate its semaphores
                        if (kvms.instruction_index >= config::INSTRUCTION_PIPELINE_STAGES)
                        {
                            auto last_slot_instruction_index = kvms.instruction_index - config::INSTRUCTION_PIPELINE_STAGES;

                            int phasebit = (last_slot_instruction_index / config::INSTRUCTION_PIPELINE_STAGES) & 1;
                            wait(kvms.instruction_finished[kvms.instruction_ring], phasebit);
                            for (int i = 0; i < num_semaphores[kvms.instruction_ring]; i++)
                            {
                                invalidate_semaphore(kvms.all_instructions[kvms.instruction_ring].semaphores[i]);
                            }

                            store_timings_and_reset<config, globals>(&kvms.all_instructions[kvms.instruction_ring].timings[0], last_slot_instruction_index, g);
                        }

                        // Step 1. Load instructions (no semaphores used)
                        load_instructions<config, globals>(&kvms.instruction()[0], kvms.instruction_index, g);

                        // Step 2. Establish physical page order
                        int last_instruction_ring = (kvms.instruction_ring + config::INSTRUCTION_PIPELINE_STAGES - 1) % config::INSTRUCTION_PIPELINE_STAGES;

                        if (kvms.instruction_index == 0)
                        {
                            for (int i = 0; i < config::NUM_PAGES; i++)
                            {
                                kvms.pid_order()[i] = i;
                            }
                        }
                        else
                        {
                            auto last_opcode = kvms.all_instructions[last_instruction_ring].instructions[0];

                            for (int i = 0; i < config::NUM_PAGES; i++)
                            {

                                int lid = dispatch_op<page_allocator_op_dispatcher<config, globals>::dispatcher, ops...>::template run<int, config, globals, config::instruction_t, int>(
                                    last_opcode, g, kvms.all_instructions[last_instruction_ring].instructions, i);

                                kvms.pid_order()[i] = kvms.all_instructions[last_instruction_ring].pid_order[lid];
                            }
                        }

                        // Step 3. Construct semaphores
                        int opcode = kvms.instruction()[0];
                        if (opcode == 0)
                        {
                            num_semaphores[kvms.instruction_ring] = 0;
                        }
                        else
                        {
                            num_semaphores[kvms.instruction_ring] = dispatch_op<semaphore_constructor_op_dispatcher<config, globals>::dispatcher, ops...>::template run<int, config, globals, ::kittens::prototype::vm::state<config>>(
                                opcode, g, kvms);
                        }

                        // Step 4. Let the rest of the world know that next instruction is ready to roll!
                        arrive(kvms.instruction_arrived[kvms.instruction_ring], 1);
                    }

                    // invalidate remaining semaphores and write out remaining timings
                    for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES; i++)
                    {
                        auto instruction_index = num_iters - config::INSTRUCTION_PIPELINE_STAGES + i;
                        if (instruction_index < 0)
                        {
                            continue;
                        }

                        auto instruction_ring = instruction_index % config::INSTRUCTION_PIPELINE_STAGES;

                        auto phasebit = (instruction_index / config::INSTRUCTION_PIPELINE_STAGES) & 1;
                        wait(kvms.instruction_finished[instruction_ring], phasebit);

                        for (int j = 0; j < num_semaphores[instruction_ring]; j++)
                        {
                            invalidate_semaphore(kvms.all_instructions[instruction_ring].semaphores[j]);
                        }

                        // technically don't need to reset, whatevs?
                        store_timings_and_reset<config, globals>(&kvms.all_instructions[instruction_ring].timings[0], instruction_index, g);
                    }
                }

            } // namespace controller
        } // namespace vm
    } // namespace prototype
} // namespace kittens
