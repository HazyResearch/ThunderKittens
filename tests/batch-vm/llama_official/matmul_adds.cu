#include "llama.cuh"
#include "matmul_pipeline.cuh"

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
            int row;
            int col;
            __device__ inline parsed_instruction(typename config::instruction_t &instruction)
            {
                layer = instruction[1];
                row = instruction[2];
                col = instruction[3];
            }
            __device__ inline parsed_instruction(state<config> &s) : parsed_instruction(s.instruction()) {}
        };

        using matmul_pipeline = matmul_pipeline<config, globals, parsed_instruction, InputActivationsPtr, WeightsPtr, iters>;

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
                return matmul_pipeline::release_lid(g, instruction, query);
            }
            static __device__ int init_semaphores(const globals &g, state<config> &s)
            {
                return matmul_pipeline::init_semaphores(s);
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
                        tma::load_async(weight, Weights, {inst.layer, inst.col, i}, inputs_arrived(s, stage));
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

                        tma::load_async(activation, Activations, {inst.row, i}, inputs_arrived(s, stage));
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
                matmul_pipeline::launcher_loop(s, g);
            }
        };
        using half_consumer = group<config::NUM_CONSUMER_WARPS/2>;
        using constorer = group<config::NUM_CONSUMER_WARPS + 1>;
        struct consumer
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                parsed_instruction inst{s};
                wait(matmul_pipeline::outputs_arrived(s), 0);
                rt_bf<16, 256> out;
                auto dt = s.tensor_alloc.template allocate<tt<float, 128, 256>>(half_consumer::groupid() * 256);
                half_consumer::load_async(out, dt);
                tensor_load_wait();
                __syncwarp();
                warp::arrive(s.tensor_finished);
                int store_bar = 10 + s.instruction_index%2;
                auto &smem = *reinterpret_cast<st_bf<16, 256>*>(s.scratch());
                for(int i = 0; i < config::NUM_CONSUMER_WARPS; i++) {
                    if(warpid() == i) warp::store(smem, out);
                    constorer::sync(store_bar); // arrive for storer
                    constorer::sync(store_bar); // await release from storer
                }
            }
        };

        struct storer
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                parsed_instruction inst{s};
                int store_bar = 10 + s.instruction_index%2;
                auto &smem = *reinterpret_cast<st_bf<16, 256>*>(s.scratch());
                for(int i = 0; i < config::NUM_CONSUMER_WARPS; i++) {
                    constorer::sync(store_bar); // await arrive from consumer
                    auto &OutputActivations = g.*OutputActivationsPtr;
                    warp::tma::store_add_async(OutputActivations, smem, {16*inst.row + 8*(i>=8) + 2*(i%4) + ((i%8)/4), inst.col});
                    tma::store_async_read_wait();
                    constorer::sync(store_bar); // release back to consumer
                }

                warp::sync();
                asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.
                if (laneid == 0)
                {
                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.row, 0}], 1);
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
            while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_GQA_AttentionDecode - 1, inst.row, 0}] < Globals::matmul_batch_block_size * Globals::num_kv_heads)
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
            while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_UpMatmul - 1, inst.row, 0}] < Globals::intermediate_dim / Globals::matmul_out_block_size)
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