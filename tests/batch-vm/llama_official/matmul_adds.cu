#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{
    using globals = llama_8b_globals;
    template <
        auto InputActivationsPtr,
        auto WeightsPtr,
        auto OutputActivationsPtr,
        int iters,
        int _opcode,
        typename gmem_waiter,
        typename config>
    struct MatMulAddOp
    {
        static constexpr int opcode = _opcode;
        static constexpr int PIPELINE_STAGES = 3;

        using weight_tile = st_bf<128, 128>;
        using activation_tile = st_bf<128, 128>;
        using output_tile = st_bf<128, 128>;

        struct parsed_instruction
        {
            int layer;
            int batch_idx;
            int output_idx;
            __device__ inline parsed_instruction(typename config::instruction_t &instruction)
            {
                layer = instruction[1];
                batch_idx = instruction[2];
                output_idx = instruction[3];
            }
            __device__ inline parsed_instruction(state<config> &s) : parsed_instruction(s.instruction()) {}
        };

        __device__ static inline int get_weight_page(state<config> &s, int stage) { return 0 + stage * 2; }     // 32 KB pages
        __device__ static inline int get_activation_page(state<config> &s, int stage) { return 6 + stage * 2; } // 32 KB pages

        __device__ static inline semaphore &inputs_arrived(state<config> &s, int stage) { return s.semaphores()[PIPELINE_STAGES * 0 + stage]; }
        __device__ static inline semaphore &inputs_finished(state<config> &s, int stage) { return s.semaphores()[PIPELINE_STAGES * 1 + stage]; }
        __device__ static inline semaphore &outputs_arrived(state<config> &s, int stage) { return s.semaphores()[PIPELINE_STAGES * 2 + stage]; }
        __device__ static inline semaphore &outputs_shared(state<config> &s) { return s.semaphores()[PIPELINE_STAGES * 3 + 0]; }

        struct controller
        {
            static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query)
            {
                return query;
            }
            static __device__ int init_semaphores(const globals &g, state<config> &s)
            {
                for (int i = 0; i < PIPELINE_STAGES; i++)
                {
                    init_semaphore(inputs_arrived(s, i), 0, 2);
                    init_semaphore(inputs_finished(s, i), 0, 1);
                    init_semaphore(outputs_arrived(s, i), 0, 1);
                }
                init_semaphore(outputs_shared(s), 0, config::NUM_CONSUMER_WARPS);
                return 3 * PIPELINE_STAGES + 1;
            }
        };

        struct loader
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                s.wait_page_ready(12);
                s.warp_finish_page(12, config::NUM_CONSUMER_WARPS); // release the unused page immediately
                parsed_instruction inst{s};
                int laneid = warp::laneid();

                if (laneid == 0)
                { // load B
                    uint32_t phasebits = 0xFFFF0000;
                    for (int i = 0; i < iters; i++)
                    {
                        int stage = i % PIPELINE_STAGES;
                        int weight_page = get_weight_page(s, stage);
                        weight_tile &weight = *reinterpret_cast<weight_tile *>(s.pages[weight_page].data);

                        wait(inputs_finished(s, stage), get_phasebit<1>(phasebits, stage));
                        update_phasebit<1>(phasebits, stage);
                        tma::expect(inputs_arrived(s, stage), weight);

                        if (i < PIPELINE_STAGES)
                        {
                            s.wait_page_ready(weight_page);
                            s.wait_page_ready(weight_page + 1);
                        }
                        auto &Weights = g.*WeightsPtr;
                        tma::load_async(weight, Weights, {inst.layer, inst.output_idx, i}, inputs_arrived(s, stage));
                    }
                }
                else if (laneid == 1)
                { // load A
                    uint32_t phasebits = 0xFFFF0000;
                    for (int i = 0; i < iters; i++)
                    {
                        int stage = i % PIPELINE_STAGES;
                        int activation_page = get_activation_page(s, stage);
                        activation_tile &activation = *reinterpret_cast<activation_tile *>(s.pages[activation_page].data);

                        wait(inputs_finished(s, stage), get_phasebit<1>(phasebits, stage));
                        update_phasebit<1>(phasebits, stage);
                        tma::expect(inputs_arrived(s, stage), activation);

                        if (i < PIPELINE_STAGES)
                        {
                            s.wait_page_ready(activation_page);
                            s.wait_page_ready(activation_page + 1);
                        }
                        auto &Activations = g.*InputActivationsPtr;

                        gmem_waiter::gmem_wait(g, s, inst);

                        tma::load_async(activation, Activations, {inst.batch_idx, i}, inputs_arrived(s, stage));
                    }
                }

                warp::sync();
                if (laneid == 0)
                {
                    wait(outputs_shared(s), 0);
                    for (int i = 0; i < PIPELINE_STAGES; i++)
                    {
                        int stage = (iters + i) % PIPELINE_STAGES;
                        int weight_page = get_weight_page(s, stage);
                        s.warp_finish_page(weight_page, config::NUM_CONSUMER_WARPS);
                        s.warp_finish_page(weight_page + 1, config::NUM_CONSUMER_WARPS);
                    }
                    for (int i = 0; i < PIPELINE_STAGES - 1; i++)
                    { // last stage is used as output page
                        int stage = (iters + i) % PIPELINE_STAGES;
                        int activation_page = get_activation_page(s, stage);
                        s.warp_finish_page(activation_page, config::NUM_CONSUMER_WARPS);
                        s.warp_finish_page(activation_page + 1, config::NUM_CONSUMER_WARPS);
                    }
                }
            }
        };

        struct launcher
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                parsed_instruction inst{s};
                int laneid = warp::laneid();
                uint32_t phasebits = 0xFFFF0000;

                s.wait_tensor_ready();

                if (laneid == 0)
                {
                    for (int i = 0; i < iters; i++)
                    {
                        int stage = i % PIPELINE_STAGES;
                        wait(inputs_arrived(s, stage), get_phasebit<0>(phasebits, stage));
                        update_phasebit<0>(phasebits, stage);
                        auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(stage * 128);
                        weight_tile &weight = *reinterpret_cast<weight_tile *>(s.pages[get_weight_page(s, stage)].data);
                        activation_tile &activation = *reinterpret_cast<activation_tile *>(s.pages[get_activation_page(s, stage)].data);
                        if (i < PIPELINE_STAGES)
                            mm<transpose::N, transpose::T>(accumulator, activation, weight, inputs_finished(s, stage));
                        else if (i >= iters - PIPELINE_STAGES)
                            mma<transpose::N, transpose::T>(accumulator, activation, weight, outputs_arrived(s, stage));
                        else
                            mma<transpose::N, transpose::T>(accumulator, activation, weight, inputs_finished(s, stage));
                    }
                }
            }
        };

        struct consumer
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                static_assert(config::NUM_CONSUMER_WARPS == 8, "NUM_CONSUMER_WARPS must be 8");
                using consumer = group<config::NUM_CONSUMER_WARPS>;

                parsed_instruction inst{s};
                rt_fl<128 / config::NUM_CONSUMER_WARPS, 128> output_fl;
                consumer::zero(output_fl);

                for (int i = 0; i < PIPELINE_STAGES; i++)
                {
                    int stage = (iters + i) % PIPELINE_STAGES;
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(stage * 128);
                    wait(outputs_arrived(s, stage), 0);
                    rt_fl<128 / config::NUM_CONSUMER_WARPS, 128> acc_fl;
                    consumer::load_async(acc_fl, accumulator);
                    tensor_load_wait();
                    __syncwarp();
                    consumer::add(output_fl, output_fl, acc_fl);
                }
                warp::arrive(s.tensor_finished);

                rt_bf<128 / config::NUM_CONSUMER_WARPS, 128> output_bf;
                consumer::copy(output_bf, output_fl);

                int last_stage = (iters - 1) % PIPELINE_STAGES;
                int output_page = get_activation_page(s, last_stage);
                output_tile &output = *reinterpret_cast<output_tile *>(s.pages[output_page].data);
                consumer::store(output, output_bf);
                __syncwarp();
                warp::arrive(outputs_shared(s));
            }
        };

        struct storer
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                parsed_instruction inst{s};
                int laneid = warp::laneid();

                wait(outputs_shared(s), 0);
                int output_page = get_activation_page(s, (iters - 1) % PIPELINE_STAGES);
                output_tile &output = *reinterpret_cast<output_tile *>(s.pages[output_page].data);

                if (laneid == 0)
                {
                    auto &OutputActivations = g.*OutputActivationsPtr;
                    tma::store_add_async(OutputActivations, output, {inst.batch_idx, inst.output_idx});
                    tma::store_async_wait();
                    s.finish_page(output_page, config::NUM_CONSUMER_WARPS);
                    s.finish_page(output_page + 1, config::NUM_CONSUMER_WARPS);
                }
                warp::sync();

                asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.
                if (laneid == 0)
                {
                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.batch_idx, 0}], 1);
                }

                warp::sync();
                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_STORE_END);
                }
            }
        };
    };

    struct o_proj_gmem_waiter
    {
        template <typename config, typename Globals, typename instruction_t>
        static __device__ inline void gmem_wait(const Globals &g, state<config> &s, instruction_t &inst)
        {
            while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_GQA_AttentionDecode - 1, inst.batch_idx, 0}] < Globals::matmul_batch_block_size * Globals::num_kv_heads)
            {
                __nanosleep(20);
            }
        }
    };

    template <typename config, typename Globals>
    struct o_proj : MatMulAddOp<
                        &Globals::attn_out,
                        &Globals::o_weights,
                        &Globals::hidden_states,
                        Globals::hidden_dim / Globals::matmul_out_block_size,
                        OPCODE_O_ProjResidual,
                        o_proj_gmem_waiter,
                        config>
    {
    };

    struct downproj_gmem_waiter
    {
        template <typename config, typename Globals, typename instruction_t>
        static __device__ inline void gmem_wait(const Globals &g, state<config> &s, instruction_t &inst)
        {
            while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_UpMatmul - 1, inst.batch_idx, 0}] < Globals::intermediate_dim / Globals::matmul_out_block_size)
            {
                __nanosleep(20);
            }
        }
    };

    template <typename config, typename Globals>
    struct downproj : MatMulAddOp<
                          &Globals::silu_out,
                          &Globals::down_weights,
                          &Globals::hidden_states,
                          Globals::intermediate_dim / Globals::matmul_out_block_size,
                          OPCODE_DownProjResidual,
                          downproj_gmem_waiter,
                          config>
    {
    };

}
