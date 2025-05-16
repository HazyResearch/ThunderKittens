#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_8b_globals;

    template <
        auto weights_ptr,
        auto outputs_ptr,
        int _opcode,
        typename gmem_waiter,
        typename Config = llama_config>
    struct rms_op
    {
        static constexpr int opcode = _opcode;
        static constexpr int REDUCTION_DIM_PER_WARP = globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        using activations_vec = sv_bf<globals::hidden_dim>; 
        using weights_vec = sv_bf<globals::hidden_dim>;

        struct parsed_instruction
        {
            int layer_idx;
            int batch_idx;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                layer_idx = instruction[1];
                batch_idx = instruction[2];
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        // Semaphores
        __device__ static inline semaphore &activations_arrived(state<Config> &s) { return s.semaphores()[0]; }
        __device__ static inline semaphore &weights_arrived(state<Config> &s) { return s.semaphores()[1]; }
        __device__ static inline semaphore &outputs_arrived(state<Config> &s) { return s.semaphores()[2]; }

        static constexpr int SHARED_PAGE = 0; // 32kb shared page

        __device__ static inline activations_vec &get_activations_vec(state<Config> &s)
        {
            char *page_base_ptr = reinterpret_cast<char *>(s.pages[s.pid(SHARED_PAGE)].data);
            return *reinterpret_cast<activations_vec *>(page_base_ptr);
        }

        __device__ static inline weights_vec &get_weights_vec(state<Config> &s)
        {
            char *page_base_ptr = reinterpret_cast<char *>(s.pages[s.pid(SHARED_PAGE)].data);
            size_t offset = sizeof(activations_vec);
            return *reinterpret_cast<weights_vec *>(page_base_ptr + offset);
        }

        struct controller
        {
            static __device__ int release_lid(const globals &g, typename Config::instruction_t &instruction, int &query)
            {

                return query;
            }
            static __device__ int init_semaphores(const globals &g, state<Config> &s)
            {
                init_semaphore(activations_arrived(s), 1);
                init_semaphore(weights_arrived(s), 1);
                init_semaphore(outputs_arrived(s), Config::NUM_CONSUMER_WARPS);
                return 3;
            }
        };
        struct loader
        {
            static __device__ inline void gmem_wait(const globals &g, state<Config> &s) {}

            static __device__ void run(const globals &g, state<Config> &s)
            {
                // if (warp::laneid() == 0)
                // {
                //     s.record(TEVENT_LOADER_START);
                // }
                parsed_instruction inst{s};
                // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
                ((uint64_t *)s.scratch())[laneid()] = 0;
                warp::sync(); // done, now we can proceed to other things.

                if (laneid() == 0)
                {
                    s.wait_page_ready(s.pid(SHARED_PAGE));
                    
                    // RMS scale
                    weights_vec &rms_scale = get_weights_vec(s);
                    tma::expect(weights_arrived(s), rms_scale);
                    auto &weights_global = g.*weights_ptr;
                    tma::load_async(rms_scale, weights_global, {inst.layer_idx, 0}, weights_arrived(s));

                    s.record(TEVENT_AT_GMEM_WAIT);
                    gmem_waiter::gmem_wait(g, s, inst);
                    s.record(TEVENT_DONE_GMEM_WAIT);

                    // Activation
                    activations_vec &act_vec = get_activations_vec(s);
                    tma::expect(activations_arrived(s), act_vec);
                    tma::load_async(act_vec, g.hidden_states, {inst.batch_idx, 0}, activations_arrived(s));
                }
                else if (laneid() >= 1 && laneid() < Config::NUM_PAGES)
                {
                    // Unused pages
                    s.wait_page_ready(s.pid(laneid()));
                    s.finish_page(s.pid(laneid()), Config::NUM_CONSUMER_WARPS);
                }

                // warp::sync();
                // if (warp::laneid() == 0)
                // {
                //     s.record(TEVENT_LOADER_END);
                // }
            }
        };
        struct launcher
        {
            static __device__ void run(const globals &g, state<Config> &s)
            {
                if (warp::laneid() == 0)
                {
                    s.wait_tensor_ready();
                    arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
                }
            }
        };
        struct consumer
        {
            static __device__ void run(const globals &g, state<Config> &s)
            {
                // if (warp::laneid() == 0)
                // {
                //     s.record(TEVENT_CONSUMER_START + warpid());
                // }

                // Setup
                parsed_instruction inst{s};
                rv_fl<REDUCTION_DIM_PER_WARP> act_vec, copy_activations_vec, rms_scale_vec; // 4096 / 16 = 256

                sv_bf<REDUCTION_DIM_PER_WARP> *activations_smem = 
                    reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP> *>(s.pages[s.pid(SHARED_PAGE)].ptr());
                sv_bf<REDUCTION_DIM_PER_WARP> *rms_scale_smem = 
                    reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP> *>(s.pages[s.pid(SHARED_PAGE)].ptr(sizeof(activations_vec)));

                // Setup
                wait(activations_arrived(s), 0);

                warp::load(act_vec, activations_smem[warpid()]);
                warp::sync();

                // Step 2: Apply RMS normalization
                warp::copy(copy_activations_vec, act_vec);                           // cast to float
                warp::mul(copy_activations_vec, copy_activations_vec, copy_activations_vec); // square
                float partial_sum = warp::sum(copy_activations_vec);

                auto smem_rms_partial_sums = ((float *)s.scratch());
                // aggregate sums across the consumer warps
                if (laneid() == 0)
                {
                    smem_rms_partial_sums[warpid()] = partial_sum;
                }

                group<Config::NUM_CONSUMER_WARPS>::sync(0);

                float full_sum = 0;
                for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++)
                {
                    full_sum += smem_rms_partial_sums[i];
                }

                float variance = full_sum / (float)globals::hidden_dim;
                float rms_scale = rsqrtf(variance + g.rms_norm_eps);

                warp::copy(copy_activations_vec, act_vec); // unsquare
                warp::mul(copy_activations_vec, copy_activations_vec, rms_scale);
                warp::copy(act_vec, copy_activations_vec);

                // multiply by rms scale
                wait(weights_arrived(s), 0);

                warp::load(rms_scale_vec, rms_scale_smem[warpid()]);
                warp::sync();

                warp::mul(act_vec, act_vec, rms_scale_vec);

                // Need to ensure storing here is correct!!!
                warp::store(activations_smem[warpid()], act_vec);
                warp::sync();
                warp::arrive(outputs_arrived(s));

                // if (warp::laneid() == 0)
                // {
                //     s.record(TEVENT_CONSUMER_END - (Config::NUM_CONSUMER_WARPS) +
                //             (kittens::group<Config::NUM_CONSUMER_WARPS>::warpid()));
                // }
            }
        };
        struct storer
        {
            // Uses 4 full pages for outputs.
            static __device__ void run(const globals &g, state<Config> &s)
            {
                // if (warp::laneid() == 0)
                // {
                //     s.record(TEVENT_STORE_START);
                // }

                parsed_instruction inst{s};

                if (warp::laneid() == 0)
                {
                    wait(outputs_arrived(s), 0);
                    activations_vec &act_vec = get_activations_vec(s);
                    auto &outputs_global = g.*outputs_ptr;
                    tma::store_async<cache_policy::NORMAL>(outputs_global, act_vec, {inst.batch_idx, 0});
                    tma::store_async_wait();

                    s.finish_page(s.pid(SHARED_PAGE), Config::NUM_CONSUMER_WARPS);
                }

                warp::sync();
                asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.

                if (warp::laneid() == 0)
                {
                    int batch_block_idx = inst.batch_idx / globals::matmul_batch_block_size;
                    atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, batch_block_idx, 0}], 1);
                }

                // warp::sync();
                // if (warp::laneid() == 0)
                // {
                //     s.record(TEVENT_STORE_END);
                // }
            }
        };
    };

    struct attn_norm_gmem_waiter
    {
        template <typename Config, typename Globals, typename instruction_t>
        static __device__ inline void gmem_wait(const Globals &g, state<Config> &s, instruction_t &inst)
        {
            int batch_block_idx = inst.batch_idx / globals::matmul_batch_block_size;
            if (inst.layer_idx > 0)
            {
                while (*(volatile int *)&g.Bar[{inst.layer_idx - 1, OPCODE_DownProjResidual - 1, batch_block_idx, 0}] < globals::num_output_blocks)
                {
                    __nanosleep(20);
                }
            }
        }
    };

    template <typename Config, typename Globals>
    struct attn_norm : rms_op<&globals::attn_norm_weights, &globals::rms_rope_intermediates, OPCODE_AttnNorm, attn_norm_gmem_waiter, Config>
    {
    };

    struct mlp_norm_gmem_waiter
    {
        template <typename Config, typename Globals, typename instruction_t>
        static __device__ inline void gmem_wait(const Globals &g, state<Config> &s, instruction_t &inst)
        {
            int batch_block_idx = inst.batch_idx / globals::matmul_batch_block_size;
            while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_O_ProjResidual - 1, batch_block_idx, 0}] < globals::num_output_blocks)
            {
                __nanosleep(20);
            }
        }
    };

    template <typename Config, typename Globals>
    struct mlp_norm : rms_op<&globals::mlp_norm_weights, &globals::rms_gate_intermediates, OPCODE_MlpNorm, mlp_norm_gmem_waiter, Config>
    {
    };

    struct lm_head_norm_gmem_waiter
    {
        template <typename Config, typename Globals, typename instruction_t>
        static __device__ inline void gmem_wait(const Globals &g, state<Config> &s, instruction_t &inst)
        {
            int batch_block_idx = inst.batch_idx / globals::matmul_batch_block_size;
            while (*(volatile int *)&g.Bar[{globals::num_hidden_layers - 1, OPCODE_DownProjResidual - 1, batch_block_idx, 0}] < globals::num_output_blocks)
            {
                __nanosleep(20);
            }
        }
    };

    template <typename Config, typename globals>
    struct lm_head_norm : rms_op<&globals::lm_head_norm_weights, &globals::rms_lm_head_intermediates, OPCODE_LM_HeadNorm, lm_head_norm_gmem_waiter, Config>
    {
    };

}

