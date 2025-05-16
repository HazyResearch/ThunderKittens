#include "llama.cuh"
#include "matmul_pipeline.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_8b_globals;
using config = llama_config;

struct qkv_gmem_waiter;

template <typename Config, typename Globals>
struct qkv_rope_append {
    static constexpr int opcode = OPCODE_QKV_RopeAppend;
    static constexpr int K_BLOCK_START = Globals::num_attention_heads;
    static constexpr int V_BLOCK_START = Globals::num_attention_heads + Globals::num_kv_heads;
    static constexpr int NUM_ITERS = Globals::hidden_dim / PIPELINE_K_DIM;

    using rope_vec = sv_fl<128>;

    struct parsed_instruction {
        int layer;
        int row;
        int col;
        __device__ inline parsed_instruction(typename Config::instruction_t &instruction) {
            layer = instruction[1];
            row = instruction[2];
            col = instruction[3];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    using matmul_pipeline = matmul_pipeline<Config, Globals, parsed_instruction, qkv_gmem_waiter, &Globals::rms_rope_intermediates, &Globals::qkv_weights, NUM_ITERS>;

    __device__ static inline semaphore &rope_arrived(state<Config> &s) { return s.semaphores()[matmul_pipeline::SEM_COUNT]; }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
            return matmul_pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            init_semaphore(rope_arrived(s), 1);
            return matmul_pipeline::init_semaphores(s) + 1;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            
            matmul_pipeline::template loader_loop(s, g, inst.layer);
            warp::sync();
            
            // TODO: Should probably update to use scratch
            if ((inst.col * 2) < V_BLOCK_START && laneid() == 0) {
                rope_vec &rope_cos = *reinterpret_cast<rope_vec*>((uint8_t*)(s.scratch()) + 8192);
                rope_vec &rope_sin = *reinterpret_cast<rope_vec*>((uint8_t*)(s.scratch()) + 8192 + sizeof(rope_vec));
                tma::expect_bytes(rope_arrived(s), 2 * sizeof(rope_vec));
                tma::load_async(rope_cos, g.rope_cos, {(int)g.pos_id, 0}, rope_arrived(s));
                tma::load_async(rope_sin, g.rope_sin, {(int)g.pos_id, 0}, rope_arrived(s));
            }
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {
            matmul_pipeline::launcher_loop(s, g);
        }
    };
    using half_consumer = group<config::NUM_CONSUMER_WARPS/2>;
    using constorer = group<config::NUM_CONSUMER_WARPS + 1>;
    using consumer_group = group<config::NUM_CONSUMER_WARPS>;
    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            if ((inst.col * 2) < V_BLOCK_START) {

                rt_fl<16, 16> intermediate_out, intermediate_rotated_out;

                rt_bf<16, 16> output[16]; // 64 registers for the final output.

                wait(rope_arrived(s), 0);
                
                wait(matmul_pipeline::outputs_arrived(s), 0);
                
                #pragma unroll
                for(int i = 0; i < 16; i++) {
                    auto dt = s.tensor_alloc.template allocate<tt<float, 128, 16>>(half_consumer::groupid() * 256 + 16 * i);
                    half_consumer::load_async(intermediate_out, dt);

                    sv_fl<16> &rope_cos = *reinterpret_cast<sv_fl<16> *>((uint8_t*)(s.scratch()) + 8192 + (i%8)*64);
                    sv_fl<16> &rope_sin = *reinterpret_cast<sv_fl<16> *>((uint8_t*)(s.scratch()) + 8192 + sizeof(sv_fl<128>) + (i%8)*64);
                    rv_fl<16, ducks::rv_layout::align> rope_cos_vec, rope_sin_vec;

                    warp::load(rope_cos_vec, rope_cos);
                    warp::load(rope_sin_vec, rope_sin);

                    // Rotate
                    #pragma unroll
                    for (int k = 0; k < intermediate_out.packed_per_tile; k++) {
                        intermediate_rotated_out.tiles[0][0].data[k].x = -intermediate_out.tiles[0][0].data[k].y;
                        intermediate_rotated_out.tiles[0][0].data[k].y =  intermediate_out.tiles[0][0].data[k].x;
                    }

                    warp::mul_col(intermediate_out, intermediate_out, rope_cos_vec);
                    warp::mul_col(intermediate_rotated_out, intermediate_rotated_out, rope_sin_vec);
                    warp::add(intermediate_out, intermediate_out, intermediate_rotated_out);

                    warp::copy(output[i], intermediate_out);
                }   

                tensor_load_wait();
                __syncwarp();
                warp::arrive(s.tensor_finished);
                int store_bar = 10 + s.instruction_index%2;
                auto &smem = *reinterpret_cast<st_bf<16, 256>*>(s.scratch());
                for(int i = 0; i < config::NUM_CONSUMER_WARPS; i++) {
                    if(warpid() == i) {
                        warp::store(smem, reinterpret_cast<rt_bf<16, 256>&>(output));
                    }
                    constorer::sync(store_bar); // arrive for storer
                    constorer::sync(store_bar); // await release from storer
                }
            }
            else {
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
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            int store_bar = 10 + s.instruction_index%2;
            auto &smem = *reinterpret_cast<st_bf<16, 256>*>(s.scratch());
            for(int i = 0; i < config::NUM_CONSUMER_WARPS; i++) {
                constorer::sync(store_bar); // await arrive from consumer

                if (inst.col < K_BLOCK_START)
                {
                    warp::tma::store_async(g.q_post_rope, smem, {16*inst.row + 8*(i>=8) + 2*(i%4) + ((i%8)/4), inst.col});
                }
                else if (inst.col < V_BLOCK_START)
                {   
                    warp::tma::store_async(g.k_cache, smem, 
                        {
                            inst.layer*(int)g.batch_size/(int)256 + (16*inst.row + 8*(i>=8) + 2*(i%4) + ((i%8)/4)), 
                            (int)g.pos_id,
                            inst.col - K_BLOCK_START, 
                            0
                        });
                }
                else
                {
                    warp::tma::store_async(g.v_cache, smem, 
                        {
                            inst.layer*(int)g.batch_size/(int)256 + (16*inst.row + 8*(i>=8) + 2*(i%4) + ((i%8)/4)), 
                            (int)g.pos_id,
                            inst.col - V_BLOCK_START, 
                            0
                        });
                }

                tma::store_async_read_wait();
                constorer::sync(store_bar); // release back to consumer
            }
            warp::sync();

            if (kittens::laneid() == 0)
            {
                int start_bar = (inst.col * Globals::matmul_out_block_size) / Globals::head_dim;
                int num_generated_heads = Globals::matmul_out_block_size / Globals::head_dim;
                for (int i = 0; i < num_generated_heads; i++) {
                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.row, start_bar + i}], 1);
                }
            }
        }
    };
};

struct qkv_gmem_waiter {
    template <typename config, typename Globals, typename instruction_t>
    static __device__ inline void gmem_wait(const Globals &g, state<config> &s, instruction_t &inst)
    {
        while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_AttnNorm - 1, inst.row, 0}] < Globals::matmul_batch_block_size)
        {
            __nanosleep(20);
        }
    }
};

}