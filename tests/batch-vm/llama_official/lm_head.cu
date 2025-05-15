#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{
    using globals = llama_8b_globals;

    struct lm_head_gmem_waiter;

    template <typename Config, typename Globals>
    struct lm_head
    {
        static constexpr int opcode = OPCODE_LM_Head;
        static constexpr int NUM_ITERS = Globals::hidden_dim / PIPELINE_K_DIM;

        struct parsed_instruction
        {
            int row;
            int col;
            __device__ inline parsed_instruction(typename config::instruction_t &instruction)
            {
                row = instruction[1];
                col = instruction[2];
            }
            __device__ inline parsed_instruction(state<config> &s) : parsed_instruction(s.instruction()) {}
        };

        using matmul_pipeline = matmul_pipeline<config, globals, parsed_instruction, lm_head_gmem_waiter, &Globals::rms_lm_head_intermediates, &Globals::lm_head_weights, NUM_ITERS>;

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
                    warp::tma::store_async(g.logits, smem, {16*inst.row + 8*(i>=8) + 2*(i%4) + ((i%8)/4), inst.col});
                    tma::store_async_read_wait();
                    constorer::sync(store_bar); // release back to consumer
                }

                warp::sync();
                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_STORE_END);
                }
            }
        };
    };

    struct lm_head_gmem_waiter {
        template <typename config, typename Globals, typename instruction_t>
        static __device__ inline void gmem_wait(const Globals &g, state<config> &s, instruction_t &inst) {
            while (*(volatile int *)&g.Bar[{0, OPCODE_LM_HeadNorm - 1, inst.row, 0}] < Globals::matmul_batch_block_size)
            {
                __nanosleep(20);
            }
        }
    };
}
