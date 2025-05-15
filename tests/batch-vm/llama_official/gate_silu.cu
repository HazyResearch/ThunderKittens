#include "llama.cuh"
#include "matmul_pipeline.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{
    using globals = llama_8b_globals;
    using config = llama_config;

    template <typename Config, typename Globals>
    struct gate_silu {
        static constexpr int opcode = OPCODE_GateSiLU;
        static constexpr int prev_opcode = opcode - 1;
        static constexpr int NUM_ITERS = Globals::hidden_dim / PIPELINE_K_DIM;

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

        using matmul_pipeline = matmul_pipeline<config, globals, parsed_instruction, &Globals::rms_gate_intermediates, &Globals::gate_weights, NUM_ITERS>;

        struct controller {
            static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
                return matmul_pipeline::release_lid(g, instruction, query);
            }
            static __device__ int init_semaphores(const globals &g, state<config> &s) {
                return matmul_pipeline::init_semaphores(s);
            }
        };

        struct loader {
            static __device__ void run(const globals &g, state<config> &s) {
                parsed_instruction inst{s};
                matmul_pipeline::loader_loop(s, g, inst.layer);
            }
        };

        struct launcher {
            static __device__ void run(const globals &g, state<config> &s) {
                matmul_pipeline::launcher_loop(s, g);
            }
        };
        using half_consumer = group<config::NUM_CONSUMER_WARPS/2>;
        using constorer = group<config::NUM_CONSUMER_WARPS + 1>;
        struct consumer {
            static __device__ void run(const globals &g, state<config> &s) {
                parsed_instruction inst{s};
                wait(matmul_pipeline::outputs_arrived(s), 0);
                rt_fl<16, 256> out_fl, gate_buf;
                rt_bf<16, 256> out_bf;
                auto dt = s.tensor_alloc.template allocate<tt<float, 128, 256>>(half_consumer::groupid() * 256);
                half_consumer::load_async(out_fl, dt);
                tensor_load_wait();
                
                half_consumer::copy(gate_buf, out_fl);
                half_consumer::mul(gate_buf, gate_buf, -1);
                half_consumer::exp(gate_buf, gate_buf);
                half_consumer::add(gate_buf, gate_buf, 1);
                half_consumer::div(out_fl, out_fl, gate_buf);
                half_consumer::copy(out_bf, out_fl);

                __syncwarp();
                warp::arrive(s.tensor_finished);
                int store_bar = 10 + s.instruction_index%2;
                auto &smem = *reinterpret_cast<st_bf<16, 256>*>(s.scratch());
                for(int i = 0; i < config::NUM_CONSUMER_WARPS; i++) {
                    if(warpid() == i) warp::store(smem, out_bf);
                    constorer::sync(store_bar); // arrive for storer
                    constorer::sync(store_bar); // await release from storer
                }
            }
        };

        struct storer {
            static __device__ void run(const globals &g, state<config> &s) {
                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_STORE_START);
                }
                
                parsed_instruction inst{s};
                int store_bar = 10 + s.instruction_index%2;
                auto &smem = *reinterpret_cast<st_bf<16, 256>*>(s.scratch());
                for(int i = 0; i < config::NUM_CONSUMER_WARPS; i++) {
                    constorer::sync(store_bar); // await arrive from consumer
                    warp::tma::store_async(g.silu_out, smem, {16*inst.row + 8*(i>=8) + 2*(i%4) + ((i%8)/4), inst.col});
                    tma::store_async_read_wait();
                    constorer::sync(store_bar); // release back to consumer
                }
                
                warp::sync();
                asm volatile("fence.acq_rel.gpu;");
                if (kittens::laneid() == 0)
                {
                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.row, inst.col}], 1);
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


