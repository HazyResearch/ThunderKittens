#include "llama.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_1b_globals;

    template <typename Config, typename Globals>
    struct rms_qkv_rope_append
    {
        static constexpr int opcode = OPCODE_RMS_QKV_MatVecRopeAppend; // Op index within the layer -- controls which barrier to listen to.

        static constexpr int K_BLK_START = 2048 / Globals::matvec_block_size;
        static constexpr int V_BLK_START = 2560 / Globals::matvec_block_size;
        static constexpr int EXPECTED_ARRIVAL_COUNT = 512;

        struct parsed_instruction
        {
            int layer_idx, start_block_idx, end_block_idx, iters;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                layer_idx = instruction[1];       // in units of 1
                start_block_idx = instruction[2]; // in units of 16 elements
                end_block_idx = instruction[3];   // in units of 16 elements
                iters = end_block_idx - start_block_idx;
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        struct pipeline_specifics
        {

            static __device__ inline void gmem_wait(const Globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};
                if (inst.layer_idx > 0)
                {
                    while (*(volatile int *)&g.Bar[{inst.layer_idx - 1, OPCODE_DownProjResidual - 1, 0}] < EXPECTED_ARRIVAL_COUNT)
                    {
                        __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                    }
                }
            }

            static __device__ inline void store(state<Config> &s, const Globals &g, parsed_instruction &inst, int output_idx, int output_stage, semaphore &sem, int bit)
            {

                int block_idx = inst.start_block_idx + output_idx;

                // apply rope

                // even for V, we need to cast from float to bf16
                sv_fl<16> &qkv_proj_smem = *reinterpret_cast<sv_fl<16> *>((float *)s.scratch() + (32 * output_stage));
                sv_bf<16> &qkv_proj_smem_bf = *reinterpret_cast<sv_bf<16> *>((float *)s.scratch() + (32 * output_stage));

                rv_fl<16> qkv_proj, rope_cos, rope_sin;

                warp::load(rope_cos, g.rope_cos, {0, 0, static_cast<int>(g.pos_id), block_idx % 4});
                warp::load(rope_sin, g.rope_sin, {0, 0, static_cast<int>(g.pos_id), block_idx % 4});

                wait(sem, bit);
                warp::load(qkv_proj, qkv_proj_smem);

                if (block_idx < V_BLK_START)
                { // only Q & K need RoPE

                    // Fetch the neighbor values
                    int mod = (laneid() & 0b1) ? -1 : 1; // 1 for even, -1 for odd
                    warp::sync();
                    float pair_val = __shfl_sync(MASK_ALL, qkv_proj[0][0], laneid() + mod);

                    // Compute RoPE in-place
                    if (laneid() < 16)
                    {
                        // will clean this up later
                        qkv_proj[0][0] = float(qkv_proj[0][0]) * rope_cos[0][0] + float(-1 * mod) * float(pair_val) * rope_sin[0][0];
                    }
                }

                warp::store(qkv_proj_smem_bf, qkv_proj);
                warp::sync();

                if (laneid() == 0)
                {

                    if (block_idx < K_BLK_START)
                    { // Q
                        tma::store_async<cache_policy::NORMAL>(g.q_post_rope, qkv_proj_smem_bf, {0, 0, 0, block_idx});
                    }
                    else if (block_idx < V_BLK_START)
                    { // K
                        int base_index = (block_idx - K_BLK_START) * Globals::matvec_block_size;
                        int head_idx = base_index / Globals::head_dim;
                        int dim_idx = (base_index % Globals::head_dim) / Globals::matvec_block_size;
                        tma::store_async<cache_policy::NORMAL>(g.k_cache, qkv_proj_smem_bf, {inst.layer_idx, static_cast<int>(g.pos_id), head_idx, dim_idx});
                    }
                    else
                    { // V
                        int base_index = (block_idx - V_BLK_START) * Globals::matvec_block_size;
                        int head_idx = base_index / Globals::head_dim;
                        int dim_idx = (base_index % Globals::head_dim) / Globals::matvec_block_size;
                        tma::store_async<cache_policy::NORMAL>(g.v_cache, qkv_proj_smem_bf, {inst.layer_idx, static_cast<int>(g.pos_id), head_idx, dim_idx});
                    }

                    tma::store_async_wait();              // not just read wait! full wait! must be visible in global!
                    asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.
                    atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, block_idx / 4}], 1);
                }

                warp::sync();
                warp::zero(qkv_proj_smem);
                warp::sync();
            }
        };

        using pipeline = rms_matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics>;

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
            {
                return pipeline::release_lid(g, instruction, query);
            }
            static __device__ int init_semaphores(const Globals &g, state<Config> &s)
            {
                return pipeline::init_semaphores(s);
            }
        };
        struct loader
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
                s.template zero_scratch<1024>();

                parsed_instruction inst{s};
                pipeline::loader_loop<&Globals::qkv_weights>(s, g, inst.layer_idx);
            }
        };
        struct launcher
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {

                parsed_instruction inst{s};
                pipeline::launcher_load_rms_and_activations<&Globals::hidden_states, &Globals::attn_norm_weights>(s, g, inst.layer_idx);
            }
        };
        struct consumer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                pipeline::consumer_loop(s, g);
            }
        };
        struct storer
        {
            // Uses 4 full pages for outputs.
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                pipeline::storer_loop(s, g);
            }
        };
    };
}
