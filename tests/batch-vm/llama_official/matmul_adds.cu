#pragma once

#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{
    using globals = llama_70b_globals;
    template <
        auto WeightsPtr,
        auto InputActivationsPtr,
        auto OutputActivationsPtr,
        int _opcode,
        int _prev_opcode = 0,
        typename Config = kittens::prototype::vm::default_config>
    struct MatMulAddOp
    {
        static constexpr int opcode = _opcode;
        static constexpr int prev_opcode = _prev_opcode;

        static constexpr int PIPELINE_STAGES = 3;

        using a_tile = st_bf<64, 128>;    // 16KB
        using b_tile = st_bf<128, 128>;   // 32KB
        using c_tile = st_bf<64, 128>;    // 16KB

        struct parsed_instruction
        {
            int layer, row, col, iters;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                layer = instruction[1];               // in units of 1
                row = instruction[2];
                col = instruction[3];
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };
        __device__ static inline semaphore &inputs_arrived(state<config> &s, int id) {
            return s.semaphores()[id];
        }
        __device__ static inline semaphore &inputs_finished(state<config> &s, int id) {
            return s.semaphores()[id+PIPELINE_STAGES];
        }
        __device__ static inline semaphore &outputs_arrived(state<config> &s, int id) {
            return s.semaphores()[id+PIPELINE_STAGES*2];
        }
        __device__ static inline semaphore &outputs_shared(state<config> &s, int id) {
            return s.semaphores()[id+PIPELINE_STAGES*2+2];
        }
    
        __device__ static inline int get_a_page(state<config> &s, int stage, int offset) {
            return stage*4 + offset;
        }
        __device__ static inline int get_b_page(state<config> &s, int stage) {
            return stage*4 + 2;
        }
        __device__ static inline int get_store_page(state<config> &s, parsed_instruction &inst, int offset) {
            return ((inst.iters+2)%PIPELINE_STAGES)*4 + offset*2;
        }

        struct controller
        {
            static __device__ int release_lid(const globals &g, typename Config::instruction_t &instruction, int &query)
            {
                // TODO: get right order
                return query;
                // int ret_order[] = {5, 6, 7, 8, 9, 10, 11, 12, PAGE_ACTIVATION, PAGE_WEIGHT_START, PAGE_WEIGHT_START + 1, PAGE_WEIGHT_START + 2, PAGE_WEIGHT_START + 3};
                // return ret_order[query];
            }
            static __device__ int init_semaphores(const globals &g, state<config> &s) {
                for(int i = 0; i < PIPELINE_STAGES; i++) {
                    init_semaphore(inputs_arrived(s, i), 1);
                    init_semaphore(inputs_finished(s, i), 2);
                }
                for(int i = 0; i < 2; i++) {
                    init_semaphore(outputs_arrived(s, i), 1);
                    init_semaphore(outputs_shared(s, i), 1);
                }
                return 2*PIPELINE_STAGES + 4; // Total semaphores initialized
            }
        };
        struct loader {
            static __device__ void run(const globals &g, state<config> &s) {
                parsed_instruction inst{s};
                uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
                int pipeline_stage = 0;
                for(int i = 0; i < inst.iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                    wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                    // if (laneid() == 0) printf(BLUE_TEXT "Loader Passed stage %d\n" RESET_TEXT, pipeline_stage);
                    warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), sizeof(a_tile)*2 + sizeof(b_tile));
                    if(laneid() < 2) {
                        int a_page = get_a_page(s, pipeline_stage, laneid());
                        if(i < PIPELINE_STAGES) {
                            s.wait_page_ready(a_page);
                        }
                        auto &a_global = g.*InputActivationsPtr;
                        a_tile &a = *reinterpret_cast<a_tile *>(s.pages[a_page].data);
                        tma::load_async(a, a_global, {inst.row + laneid(), i}, inputs_arrived(s, pipeline_stage));
                    } else if (laneid() == 2) {
                        int b_page = get_b_page(s, pipeline_stage);
                        if(i < PIPELINE_STAGES) {
                            s.wait_page_ready(b_page);
                            s.wait_page_ready(b_page+1);
                        }
                        auto &b_global = g.*WeightsPtr;
                        b_tile &b = *reinterpret_cast<b_tile *>(s.pages[b_page].data);
                        tma::load_async(b, b_global, {inst.col, i}, inputs_arrived(s, pipeline_stage));
                    }
                    update_phasebit<1>(semaphore_bitfield, pipeline_stage);
                }
                warp::sync(); // Ensure all loads are issued
                // if (laneid() == 0) printf(BLUE_TEXT "Loader finished issuing loads\n" RESET_TEXT);
    
                if(laneid() >= 28) {
                    for(int i = 0; i < PIPELINE_STAGES-1; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                        wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                        int release_pid = pipeline_stage*4 + laneid() - 28;
                        s.finish_page(release_pid, config::NUM_CONSUMER_WARPS);
                    }
                }
            }
        };
        struct launcher { // launches mma's
            static __device__ void run(const globals &g, state<config> &s) {
    
                parsed_instruction inst{s};
                uint32_t semaphore_bitfield = 0xFFFF0000;
                int pipeline_stage = 0;
    
                wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                // if (laneid() == 0) printf(GREEN_TEXT "Launcher Passed stage %d\n" RESET_TEXT, pipeline_stage);
                s.wait_tensor_ready();
                if(laneid() < 2) {
                    // auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(laneid(), 0);
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(0, laneid() * 128);
                    a_tile &a = *reinterpret_cast<a_tile *>(s.pages[get_a_page(s, pipeline_stage, laneid())].data);
                    b_tile &b = *reinterpret_cast<b_tile *>(s.pages[get_b_page(s, pipeline_stage)].data);
                    mm<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
                }
                // if (laneid() == 0) printf(GREEN_TEXT "Finished first mma\n" RESET_TEXT);
                update_phasebit<0>(semaphore_bitfield, pipeline_stage);
                pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage);
                
                for(int i = 1; i < inst.iters-1; i++, update_phasebit<0>(semaphore_bitfield, pipeline_stage), pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                    wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                    // if (laneid() == 0) printf(GREEN_TEXT "Launcher Passed stage %d\n" RESET_TEXT, pipeline_stage);
                    if(laneid() < 2) {
                        // auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(laneid(), 0);
                        auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(0, laneid() * 128);
                        a_tile &a = *reinterpret_cast<a_tile *>(s.pages[get_a_page(s, pipeline_stage, laneid())].data);
                        b_tile &b = *reinterpret_cast<b_tile *>(s.pages[get_b_page(s, pipeline_stage)].data);
                        mma<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
                    }
                }
                
                wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                // if (laneid() == 0) printf(GREEN_TEXT "Launcher Passed stage %d\n" RESET_TEXT, pipeline_stage);
    
                if(laneid() < 2) {
                    // auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(laneid(), 0);
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(0, laneid() * 128);
                    a_tile &a = *reinterpret_cast<a_tile *>(s.pages[get_a_page(s, pipeline_stage, laneid())].data);
                    b_tile &b = *reinterpret_cast<b_tile *>(s.pages[get_b_page(s, pipeline_stage)].data);
                    mma<transpose::N, transpose::T>(accumulator, a, b, outputs_arrived(s, laneid()));
                }
                warp::sync();
                // if (laneid() == 0) printf(RED_TEXT "Finished launcher\n" RESET_TEXT);
    
            }
        };
        struct consumer {
            static __device__ void run(const globals &g, state<config> &s) {
                parsed_instruction inst{s};
                int groupid = warpgroup::groupid();
                if (groupid < 2)
                {
                    wait(outputs_arrived(s, groupid), 0);
        
                    // auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(groupid, 0);
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(0, groupid * 128);
                    
                    rt_fl<16, 128> acc_rt;
                    rt_bf<16, 128> acc_bf16;
                    
                    warpgroup::load_async(acc_rt, accumulator);
                    tensor_load_wait();
                    warp::copy(acc_bf16, acc_rt);
                    warp::arrive(s.tensor_finished);
                    
                    int store_page_id = get_store_page(s, inst, groupid);
                    c_tile &store_buffer = *reinterpret_cast<c_tile *>(s.pages[store_page_id].data);
                    warpgroup::store(store_buffer, acc_bf16);
                    warpgroup::sync(groupid);
                    warpgroup::arrive(outputs_shared(s, groupid));
                }
            }
        };
        struct storer
        {
            // Uses 4 full pages for outputs.
            static __device__ void run(const globals &g, state<Config> &s)
            {
                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_STORE_START);
                }
            
                parsed_instruction inst{s};
                
                if (laneid() < 2) {
                    wait(outputs_shared(s, laneid()), 0);
                    auto &OutputActivations = g.*OutputActivationsPtr;
                    int store_page = get_store_page(s, inst, laneid());
                    c_tile &output = *reinterpret_cast<c_tile *>(s.pages[store_page].data);
                    tma::store_add_async(OutputActivations, output, {inst.row+laneid(), inst.col});
                    tma::store_async_wait();
                    s.finish_page(store_page, config::NUM_CONSUMER_WARPS);
                }
                warp::sync();

                asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.
                if (laneid() == 0)
                {
                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, 0}], 1);
                }

                warp::sync();
                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_STORE_END);
                }
            }
        };
    };

    template <typename Config, typename Globals>
    struct downproj : MatMulAddOp<
                          &Globals::down_weights,
                          &Globals::silu_out,
                          &Globals::hidden_states,
                          OPCODE_DownProjResidual,
                          OPCODE_DownProjResidual - 1,
                          Config>
    {
    };

    template <typename Config, typename Globals>
    struct o_proj : MatMulAddOp<
                        &Globals::o_weights,
                        &Globals::attn_out,
                        &Globals::hidden_states,
                        OPCODE_O_ProjResidual,
                        OPCODE_O_ProjResidual - 1,
                        Config>
    {
    };
}
