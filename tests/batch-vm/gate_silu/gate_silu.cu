#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{
    using globals = llama_70b_globals;
    using config = default_config;

    template <typename Config, typename Globals>
    struct up_matmul {
        static constexpr int opcode = OPCODE_UpMatmul;
        static constexpr int prev_opcode = opcode - 1;
        static constexpr int PIPELINE_STAGES = 3;
        static constexpr int NUM_ITERS = Globals::hidden_dim / Globals::matmul_out_block_size;

        using weight_tile = st_bf<128, 128>;
        using weight_subtile = st_subtile<weight_tile, 128 / config::NUM_CONSUMER_WARPS, 128>;
        using activation_tile = st_bf<128, 128>;
        using output_tile = st_bf<128, 128>;

        struct parsed_instruction {
            int layer;
            int row;
            int col;
            __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
                layer = instruction[1];
                row = instruction[2];
                col = instruction[3];
            }
            __device__ inline parsed_instruction(state<config> &s) : parsed_instruction(s.instruction()) {}
        };

        __device__ static inline int get_weight_page(state<config> &s, int stage)     { return 0 + stage*2; } // 32 KB pages
        __device__ static inline int get_activation_page(state<config> &s, int stage) { return 6 + stage*2; } // 32 KB pages

        __device__ static inline semaphore &inputs_arrived(state<config> &s, int stage)  { return s.semaphores()[PIPELINE_STAGES*0 + stage]; }
        __device__ static inline semaphore &inputs_finished(state<config> &s, int stage) { return s.semaphores()[PIPELINE_STAGES*1 + stage]; }
        __device__ static inline semaphore &outputs_arrived(state<config> &s, int stage) { return s.semaphores()[PIPELINE_STAGES*2 + stage]; }
        __device__ static inline semaphore &outputs_shared(state<config> &s)             { return s.semaphores()[PIPELINE_STAGES*3 + 0]; }
        __device__ static inline semaphore &silu_arrived(state<config> &s)                { return s.semaphores()[PIPELINE_STAGES*3 + 1]; }

        struct controller {
            static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
                return query;
            }
            static __device__ int init_semaphores(const globals &g, state<config> &s) {
                for(int i = 0; i < PIPELINE_STAGES; i++) {
                    init_semaphore(inputs_arrived(s, i), 0, 2);
                    init_semaphore(inputs_finished(s, i), 0, 1);
                    init_semaphore(outputs_arrived(s, i), 0, 1);
                }
                init_semaphore(outputs_shared(s), 0, config::NUM_CONSUMER_WARPS);
                init_semaphore(silu_arrived(s), 0, 1);
                return 3*PIPELINE_STAGES + 2;
            }
        };

        struct loader {
            static __device__ void run(const globals &g, state<config> &s) {
                s.wait_page_ready(12);
                s.warp_finish_page(12, config::NUM_CONSUMER_WARPS); // release the unused page immediately
                parsed_instruction inst{s};
                int laneid = warp::laneid();

                if (laneid == 0) { // load B
                    uint32_t phasebits = 0xFFFF0000;
                    for (int i = 0; i < NUM_ITERS; i++) {
                        int stage = i % PIPELINE_STAGES;
                        int weight_page = get_weight_page(s, stage);
                        weight_tile &weight = *reinterpret_cast<weight_tile *>(s.pages[weight_page].data);

                        wait(inputs_finished(s, stage), get_phasebit<1>(phasebits, stage));
                        update_phasebit<1>(phasebits, stage);
                        tma::expect(inputs_arrived(s, stage), weight);

                        if(i < PIPELINE_STAGES) {
                            s.wait_page_ready(weight_page);
                            s.wait_page_ready(weight_page + 1);
                        }
                        tma::load_async(weight, g.up_weights, {inst.col, i}, inputs_arrived(s, stage));
                    }
                } else if (laneid == 1) { // load A
                    uint32_t phasebits = 0xFFFF0000;
                    for (int i = 0; i < NUM_ITERS; i++) {
                        int stage = i % PIPELINE_STAGES;
                        int activation_page = get_activation_page(s, stage);
                        activation_tile &activation = *reinterpret_cast<activation_tile *>(s.pages[activation_page].data);

                        wait(inputs_finished(s, stage), get_phasebit<1>(phasebits, stage));
                        update_phasebit<1>(phasebits, stage);
                        tma::expect(inputs_arrived(s, stage), activation);

                        if(i < PIPELINE_STAGES) {
                            s.wait_page_ready(activation_page);
                            s.wait_page_ready(activation_page + 1);
                        }
                        tma::load_async(activation, g.rms_gate_intermediates, {inst.row, i}, inputs_arrived(s, stage));
                    }
                }

                warp::sync();
                if (laneid == 0) {
                    int last_stage = (NUM_ITERS - 1) % PIPELINE_STAGES;
                    wait(outputs_arrived(s, last_stage), 0);
                    
                    int gate_silu_page = get_weight_page(s, last_stage);
                    weight_tile &silu_out = *reinterpret_cast<weight_tile *>(s.pages[gate_silu_page].data);
                    tma::expect(silu_arrived(s), silu_out);
                    tma::load_async(silu_out, g.silu_out, {inst.row, inst.col}, silu_arrived(s));
                }
                
                if (laneid == 1) {
                    wait(outputs_shared(s), 0);
                    for (int i = 0; i < PIPELINE_STAGES - 1; i++) { // first stage used for gate silu 
                        int stage = (NUM_ITERS + i) % PIPELINE_STAGES;
                        int weight_page = get_weight_page(s, stage);
                        s.finish_page(weight_page, config::NUM_CONSUMER_WARPS);
                        s.finish_page(weight_page + 1, config::NUM_CONSUMER_WARPS);
                    }
                    for (int i = 0; i < PIPELINE_STAGES - 1; i++) { // last stage is used as output page
                        int stage = (NUM_ITERS + i) % PIPELINE_STAGES;
                        int activation_page = get_activation_page(s, stage);
                        s.finish_page(activation_page, config::NUM_CONSUMER_WARPS);
                        s.finish_page(activation_page + 1, config::NUM_CONSUMER_WARPS);
                    }
                }
            }
        };

        struct launcher {
            static __device__ void run(const globals &g, state<config> &s) {
                parsed_instruction inst{s};
                int laneid = warp::laneid();
                uint32_t phasebits = 0xFFFF0000;

                s.wait_tensor_ready();

                if (laneid == 0) {
                    for (int i = 0; i < NUM_ITERS; i++) {
                        int stage = i % PIPELINE_STAGES;
                        wait(inputs_arrived(s, stage), get_phasebit<0>(phasebits, stage));
                        update_phasebit<0>(phasebits, stage);
                        auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(stage*128);
                        weight_tile &weight = *reinterpret_cast<weight_tile *>(s.pages[get_weight_page(s, stage)].data);
                        activation_tile &activation = *reinterpret_cast<activation_tile *>(s.pages[get_activation_page(s, stage)].data);
                        if (i < PIPELINE_STAGES)
                            mm<transpose::N, transpose::T>(accumulator, activation, weight, inputs_finished(s, stage));
                        else if (i >= NUM_ITERS - PIPELINE_STAGES)
                            mma<transpose::N, transpose::T>(accumulator, activation, weight, outputs_arrived(s, stage));
                        else
                            mma<transpose::N, transpose::T>(accumulator, activation, weight, inputs_finished(s, stage));
                    }
                }
            }
        };

        struct consumer {
            static __device__ void run(const globals &g, state<config> &s) {
                static_assert(config::NUM_CONSUMER_WARPS == 8, "NUM_CONSUMER_WARPS must be 8");
                using consumer = group<config::NUM_CONSUMER_WARPS>;

                parsed_instruction inst{s};
                rt_fl<128 / config::NUM_CONSUMER_WARPS, 128> output_fl, silu_fl;
                consumer::zero(output_fl);


                for (int i = 0; i < PIPELINE_STAGES; i++) {
                    int stage = (NUM_ITERS + i) % PIPELINE_STAGES;
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(stage*128);
                    wait(outputs_arrived(s, stage), 0);
                    rt_fl<128 / config::NUM_CONSUMER_WARPS, 128> acc_fl;
                    consumer::load_async(acc_fl, accumulator);
                    tensor_load_wait(); 
                    __syncwarp();
                    consumer::add(output_fl, output_fl, acc_fl);
                }
                warp::sync();
                warp::arrive(s.tensor_finished);

                // warp::load(silu_out, g.silu_out, {inst.row, inst.col});
                // warp::store(silu_subtile, silu_fl);

                wait(silu_arrived(s), 0);
                int last_stage = (NUM_ITERS - 1) % PIPELINE_STAGES;
                int gate_silu_page = get_weight_page(s, last_stage);
                weight_tile &silu_out = *reinterpret_cast<weight_tile *>(s.pages[gate_silu_page].data);
                weight_subtile silu_subtile(silu_out, {warpgroup::warpid(), 0});
                warp::load(silu_fl, silu_subtile);

                consumer::mul(output_fl, output_fl, silu_fl);

                rt_bf<128 / config::NUM_CONSUMER_WARPS, 128> output_bf;
                consumer::copy(output_bf, output_fl);

                int output_page = get_activation_page(s, last_stage);
                output_tile &output = *reinterpret_cast<output_tile *>(s.pages[output_page].data);
                consumer::store(output, output_bf);
                __syncwarp();
                warp::arrive(outputs_shared(s));
            }
        };

        struct storer {
            static __device__ void run(const globals &g, state<config> &s) {
                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_STORE_START);
                }

                parsed_instruction inst{s};
                int laneid = warp::laneid();
                int last_stage = (NUM_ITERS - 1) % PIPELINE_STAGES;
                wait(outputs_shared(s), 0);

                if (laneid == 0) {
                    int output_page = get_activation_page(s, last_stage);
                    output_tile &output = *reinterpret_cast<output_tile *>(s.pages[output_page].data);
                    tma::store_async(g.silu_out, output, {inst.row, inst.col});
                    tma::store_async_wait();
                    s.finish_page(output_page, config::NUM_CONSUMER_WARPS);
                    s.finish_page(output_page + 1, config::NUM_CONSUMER_WARPS);
                } else if (laneid == 1) {
                    int gate_silu_page = get_weight_page(s, last_stage);
                    s.finish_page(gate_silu_page, config::NUM_CONSUMER_WARPS);
                    s.finish_page(gate_silu_page + 1, config::NUM_CONSUMER_WARPS);
                }

                warp::sync();

                asm volatile("fence.acq_rel.gpu;");
                if (kittens::laneid() == 0)
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

