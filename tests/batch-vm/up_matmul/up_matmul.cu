#include "llama.cuh"
using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_70b_globals;
    using config = default_config;

    template <typename Config, typename Globals>
    struct up_matmul
    {
        static constexpr int opcode = OPCODE_UpMatmul;
        static constexpr int prev_opcode = opcode - 1;
        static constexpr int PIPELINE_STAGES = 3;

        using a_tile = st_bf<64, 128>;    // 16KB
        using b_tile = st_bf<128, 128>;   // 32KB
        using c_tile = st_bf<64, 128>;    // 16KB
        using c_subtile = st_subtile<c_tile, 16, 128>;

        struct parsed_instruction
        {
            int layer;
            int row, col, iters;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                layer = instruction[1];
                row   = instruction[2];
                col   = instruction[3];
                iters = instruction[4];
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        //  semaphores
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
        __device__ static inline semaphore &silu_arrived(state<config> &s, int id) {
            return s.semaphores()[id+PIPELINE_STAGES*2+4];
        }

        // getters
        __device__ static inline int get_a_page(state<config> &s, int stage, int offset) {
            return stage*4 + offset;
        }
        __device__ static inline int get_b_page(state<config> &s, int stage) {
            return stage*4 + 2;
        }
        __device__ static inline int get_store_page(state<config> &s, parsed_instruction &inst, int offset) {
            return ((inst.iters+2)%PIPELINE_STAGES)*4 + offset;
        }
        __device__ static inline int get_gate_silu_page(state<config> &s, parsed_instruction &inst, int offset) {
            return ((inst.iters+2)%PIPELINE_STAGES)*4 + 2 + offset;
        }
        
        // 
        __device__ static inline c_tile &get_gate_silu_buffer(state<config> &s, int gate_silu_page) {
            char *page_base_ptr = reinterpret_cast<char *>(s.pages[gate_silu_page].data);
            return *reinterpret_cast<c_tile *>(page_base_ptr);
        }

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
            {
                // first the pages we don't use (we use 10 pages)
                // then input, then rms scale, then up, then gate
                // TODO: Edit page release order
                
                return query; 
                // int ret_order[] = {9, 10, 11, 12, PAGE_RMS_SCALE_ACTIVATION, PAGE_UP_START, PAGE_UP_START + 1, PAGE_UP_START + 2, PAGE_UP_START + 3, PAGE_GATE_START, PAGE_GATE_START + 1, PAGE_GATE_START + 2, PAGE_GATE_START + 3};

                // return ret_order[query];
            }
            static __device__ int init_semaphores(const Globals &g, state<Config> &s)
            {
                for(int i = 0; i < PIPELINE_STAGES; i++) {
                    init_semaphore(inputs_arrived(s, i), 1);
                    init_semaphore(inputs_finished(s, i), 2);
                }
                for(int i = 0; i < 2; i++) {
                    init_semaphore(outputs_arrived(s, i), 1);
                    init_semaphore(outputs_shared(s, i), 1);
                    init_semaphore(silu_arrived(s, i), 1);
                }
                return (PIPELINE_STAGES * 2) + (2 * 3);
                // return (PIPELINE_STAGES * 2) + (2 * 2) + 1;
            }
        };

        struct loader {
            static __device__ void run(const globals &g, state<config> &s) {
                parsed_instruction inst{s};
                uint32_t semaphore_bitfield = 0xFFFF0000;
                int pipeline_stage = 0;

                for(int i = 0; i < inst.iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                    wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                    warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), sizeof(a_tile)*2 + sizeof(b_tile));
                    if(laneid() < 2) {
                        int a_page = get_a_page(s, pipeline_stage, laneid());
                        if(i < PIPELINE_STAGES) {
                            s.wait_page_ready(a_page);
                        }
                        a_tile &a = *reinterpret_cast<a_tile *>(s.pages[a_page].data);
                        tma::load_async(a, g.rms_gate_intermediates, {inst.row + laneid(), i}, inputs_arrived(s, pipeline_stage));
                    } else if (laneid() == 2) {
                        int b_page = get_b_page(s, pipeline_stage);
                        if(i < PIPELINE_STAGES) {
                            s.wait_page_ready(b_page);
                            s.wait_page_ready(b_page+1);
                        }
                        b_tile &b = *reinterpret_cast<b_tile *>(s.pages[b_page].data);
                        tma::load_async(b, g.up_weights, {inst.col, i}, inputs_arrived(s, pipeline_stage));
                    } 
                    update_phasebit<1>(semaphore_bitfield, pipeline_stage);
                }
                warp::sync(); // Ensure all loads are issued

                if (laneid() == 28) {
                    wait(outputs_arrived(s, 0), 0);
                    wait(outputs_arrived(s, 1), 0);
                    // wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                    
                    int gate_silu_page = get_gate_silu_page(s, inst, 1);
                    c_tile &silu_out = get_gate_silu_buffer(s, gate_silu_page);
                    tma::expect(silu_arrived(s, 0), silu_out);
                    tma::load_async(silu_out, g.gate_silu_intermediates, {inst.row, inst.col}, silu_arrived(s, 0));

                    gate_silu_page = get_gate_silu_page(s, inst, 0);
                    silu_out = get_gate_silu_buffer(s, gate_silu_page);
                    tma::expect(silu_arrived(s, 1), silu_out);
                    tma::load_async(silu_out, g.gate_silu_intermediates, {inst.row, inst.col}, silu_arrived(s, 1));
                } 
                // if(laneid() >= 28) {
                //     for(int i = 0; i < PIPELINE_STAGES-1; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                //         wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                //         int release_pid = pipeline_stage*4 + laneid() - 28;
                //         s.finish_page(release_pid, config::NUM_CONSUMER_WARPS);
                //     }
                // }
                warp::sync();
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
                    
                    rt_fl<16, 128> acc_rt, silu_rt;
                    rt_bf<16, 128> acc_bf16;
                    
                    warpgroup::load_async(acc_rt, accumulator);
                    tensor_load_wait();
                    
                    // Multiply by the silu gate
                    wait(silu_arrived(s, groupid), 0);
                    int gate_silu_page = get_gate_silu_page(s, inst, groupid);
                    c_tile &silu_out = get_gate_silu_buffer(s, gate_silu_page);
                    c_subtile silu_subtile(silu_out, {warpgroup::warpid(), 0});
                    
                    warp::load(acc_rt, silu_subtile);
                    // warp::load(silu_rt, silu_subtile);
                    // warp::sync();
                    // warp::mul(acc_rt, acc_rt, silu_rt);
                    // warpgroup::sync(groupid);

                    // Store in bf16
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
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_STORE_START);
                }

                parsed_instruction inst{s};

                if (laneid() < 2) {
                    wait(outputs_shared(s, laneid()), 0);
                    int store_page = get_store_page(s, inst, laneid());
                    c_tile &output = *reinterpret_cast<c_tile *>(s.pages[get_store_page(s, inst, laneid())].data);
                    tma::store_async(g.silu_out, output, {inst.row+laneid(), inst.col});
                    tma::store_async_read_wait();
                    s.finish_page(store_page, config::NUM_CONSUMER_WARPS);
                }
                warp::sync();

                asm volatile("fence.acq_rel.gpu;");
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
}
