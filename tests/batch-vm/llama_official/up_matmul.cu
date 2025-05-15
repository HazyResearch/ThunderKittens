#include "llama.cuh"
#include "matmul_pipeline.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{
    using globals = llama_8b_globals;
    using config = llama_config;

    struct up_matmul_gmem_waiter;

    template <typename Config, typename Globals>
    struct up_matmul {
        static constexpr int opcode = OPCODE_UpMatmul;
        static constexpr int prev_opcode = opcode - 1;
        static constexpr int NUM_ITERS = Globals::hidden_dim / PIPELINE_K_DIM;

        using silu_tile = st_bf<128, 256>;

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

        using matmul_pipeline = matmul_pipeline<config, globals, parsed_instruction, up_matmul_gmem_waiter, &Globals::rms_gate_intermediates, &Globals::up_weights, NUM_ITERS>;

        __device__ static inline semaphore &silu_arrived(state<config> &s, int laneid)      { return s.semaphores()[matmul_pipeline::SEM_COUNT + laneid]; }

        struct controller {
            static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
                return matmul_pipeline::release_lid(g, instruction, query);
            }
            static __device__ int init_semaphores(const globals &g, state<config> &s) {
                for (int i = 0; i < 2; i++) {
                    init_semaphore(silu_arrived(s, i), 1);
                }
                return matmul_pipeline::init_semaphores(s) + 2; // +2 for silu_arrived
            }
        };

        struct loader {
            static __device__ void run(const globals &g, state<config> &s) {
                parsed_instruction inst{s};

                // // Once this loop is done, all pages used are released and ready for reuse.
                matmul_pipeline::template loader_loop<2>(s, g, inst.layer);

                // /*
                // If specify two in loader_loop, now have FOUR free pages here:
                // */
                warp::sync(); // need to sync here 
                if (kittens::laneid() < 2) {
                    int unfreed_page_base = matmul_pipeline::get_used_page_at(2 + (2 * laneid()));
                    silu_tile &silu_out = *reinterpret_cast<silu_tile *>(s.pages[unfreed_page_base].data);

                    while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_GateSiLU - 1, inst.row, inst.col}] < 1)
                    {
                        __nanosleep(20);
                    }

                    tma::expect(silu_arrived(s, laneid()), silu_out);
                    tma::load_async(silu_out, g.silu_out, {(inst.row * 2) + laneid(), inst.col}, silu_arrived(s, laneid()));
                } 
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
                rt_fl<16, 256> out_fl, silu_fl;
                rt_bf<16, 256> out_bf;
                auto dt = s.tensor_alloc.template allocate<tt<float, 128, 256>>(half_consumer::groupid() * 256);
                half_consumer::load_async(out_fl, dt);
                tensor_load_wait();

                // Load gate_silu intermediates here 
                wait(silu_arrived(s, half_consumer::groupid()), 0);
                
                int unfreed_page_base = matmul_pipeline::get_used_page_at(2 + (2 * half_consumer::groupid()));
                silu_tile &silu_out = *reinterpret_cast<silu_tile *>(s.pages[unfreed_page_base].data);
                half_consumer::load(silu_fl, silu_out);

                half_consumer::mul(out_fl, out_fl, silu_fl);
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
                
                // Need to free gate_silu pages
                if (kittens::laneid() < 2) {
                    int unfreed_page_base = matmul_pipeline::get_used_page_at(2 + (2 * laneid()));
                    s.finish_page(unfreed_page_base, config::NUM_CONSUMER_WARPS);
                    s.finish_page(unfreed_page_base + 1, config::NUM_CONSUMER_WARPS);
                }

                warp::sync();

                asm volatile("fence.acq_rel.gpu;");
                if (kittens::laneid() == 0)
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

    struct up_matmul_gmem_waiter {
        template <typename config, typename Globals, typename instruction_t>
        static __device__ inline void gmem_wait(const Globals &g, state<config> &s, instruction_t &inst) {
            while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_MlpNorm - 1, inst.row, 0}] < Globals::matmul_batch_block_size)
            {
                __nanosleep(20);
            }
        }
    };
}

