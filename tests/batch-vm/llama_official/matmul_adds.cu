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
                matmul_pipeline::loader_loop(s, g);
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
