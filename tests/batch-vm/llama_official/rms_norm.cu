#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;

/*
What do we need to do here... normalize entire hidden state vector 
Need to prevent activation values from getting too large or small as go through layers 
Calculate rms_norm for one entire hidden state 1 x LLAMA_8B_HIDDEN_DIM 
*/

namespace kittens::prototype::vm
{

    using globals = llama_8b_globals;

    template <
        auto A_Ptr,
        auto B_Ptr,
        auto C_Ptr,
        int _opcode,
        int _prev_opcode = 0,
        typename Config = kittens::prototype::vm::default_config>
    struct rms_op
    {
        static constexpr int opcode = _opcode;
        static constexpr int REDUCTION_DIM_PER_WARP = globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

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

        // Pages (very naive for now, no fine-grained usage)
        static constexpr int PAGE_WEIGHT = 0;
        static constexpr int PAGE_ACTIVATION = 1;
        __device__ static inline int get_weight_page(state<Config> &s) { return s.pid(PAGE_WEIGHT); }
        __device__ static inline int get_activation_page(state<Config> &s) { return s.pid(PAGE_ACTIVATION); }

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
            static __device__ void run(const globals &g, state<Config> &s)
            {
                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_LOADER_START);
                }
                parsed_instruction inst{s};
                // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
                ((uint64_t *)s.scratch())[laneid()] = 0;
                warp::sync(); // done, now we can proceed to other things.

                if (laneid() == 0)
                {
                    // RMS scale
                    int weight_page = get_weight_page(s);
                    s.wait_page_ready(weight_page);
                    auto &rms_scale = *reinterpret_cast<sv_bf<LLAMA_8B_HIDDEN_DIM>*>(s.pages[weight_page].ptr());
                    s.record(TEVENT_TRIPLES_START);
                    tma::expect(weights_arrived(s), rms_scale);
                    auto& b_global = g.*B_Ptr;
                    tma::load_async(rms_scale, b_global, {inst.layer_idx, 0}, weights_arrived(s));

                    // Activation
                    int act_page = get_activation_page(s);
                    s.wait_page_ready(act_page);
                    s.record(TEVENT_AT_GMEM_WAIT);

                    // TODO: Add barrier back in 
                    // while (inst.layer_idx > 0 && *(volatile int *)&g.Bar[{inst.layer_idx - 1, OPCODE_DownProjResidual - 1, 0}] < 512)
                    //     __nanosleep(20);

                    s.record(TEVENT_DONE_GMEM_WAIT);
                    auto &activations = *reinterpret_cast<sv_bf<LLAMA_8B_HIDDEN_DIM>*>(s.pages[act_page].ptr());
                    s.record(TEVENT_TRIPLES_START + 7);
                    tma::expect(activations_arrived(s), activations);
                    auto& a_global = g.*A_Ptr;
                    tma::load_async(activations, a_global, {}, activations_arrived(s));
                }

                else if (laneid() >= 2 && laneid() <= 12)
                {
                    // Unused pages
                    s.wait_page_ready(s.pid(laneid()));
                    arrive(s.page_finished[s.pid(laneid())][0], Config::NUM_CONSUMER_WARPS);
                }

                warp::sync();
                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_LOADER_END);
                }
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
                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_CONSUMER_START + warpid());
                }

                // Setup
                parsed_instruction inst{s};
                rv_fl<REDUCTION_DIM_PER_WARP> activations_vec, copy_activations_vec, rms_scale_vec;
                sv_bf<REDUCTION_DIM_PER_WARP>* rms_scale_smem   = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP>*>(s.pages[get_weight_page(s)].ptr());
                sv_bf<REDUCTION_DIM_PER_WARP>* activations_smem = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP>*>(s.pages[get_activation_page(s)].ptr());

                // Setup
                wait(activations_arrived(s), 0);

                warp::load(activations_vec, activations_smem[warpid()]);
                warp::sync();

                // Step 2: Apply RMS normalization
                warp::copy(copy_activations_vec, activations_vec);                           // cast to float
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

                warp::copy(copy_activations_vec, activations_vec); // unsquare
                warp::mul(copy_activations_vec, copy_activations_vec, rms_scale);
                warp::copy(activations_vec, copy_activations_vec);

                // multiply by rms scale
                wait(weights_arrived(s), 0);

                warp::load(rms_scale_vec, rms_scale_smem[warpid()]);
                warp::sync();

                warp::mul(activations_vec, activations_vec, rms_scale_vec);

                // Need to ensure storing here is correct!!!
                warp::store(activations_smem[warpid()], activations_vec);
                warp::sync();
                warp::arrive(outputs_arrived(s));
            }
        };
        struct storer
        {
            // Uses 4 full pages for outputs.
            static __device__ void run(const globals &g, state<Config> &s)
            {
                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_STORE_START);
                }

                parsed_instruction inst{s};

                if (warp::laneid() == 0)
                {
                    wait(outputs_arrived(s), 0);
                    int activation_page = get_activation_page(s);
                    auto &rms_activations = *reinterpret_cast<sv_bf<LLAMA_8B_HIDDEN_DIM>*>(s.pages[activation_page].ptr());
                    auto &c_global = g.*C_Ptr;
                    tma::store_async<cache_policy::NORMAL>(c_global, rms_activations, {inst.batch_idx, 0});
                    tma::store_async_wait(); 
                    
                    s.finish_page(activation_page, Config::NUM_CONSUMER_WARPS);
                    s.finish_page(get_weight_page(s), Config::NUM_CONSUMER_WARPS);
                }

                warp::sync();
                asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.

                // if (warp::laneid() == 0)
                //     atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, inst.qkv_block_idx / 4}], 1);

                warp::sync();
                if (laneid() == 0)
                    s.record(TEVENT_STORE_END);
            }
        };
    };

    template <typename Config, typename globals>
    struct pre_rms_norm : rms_op<
                          &globals::attn_norm_weights,
                          &globals::hidden_states,
                          &globals::rms_rope_intermediates,
                          OPCODE_RMS_NORM,
                          OPCODE_RMS_NORM - 1,
                          Config>
    {};

    template <typename Config, typename globals>
    struct post_rms_norm : rms_op<
                        &globals::mlp_norm_weights,
                        &globals::hidden_states,
                        &globals::rms_gate_intermediates,
                        OPCODE_POST_RMS_NORM,
                        OPCODE_POST_RMS_NORM - 1,
                        Config>
    {};
}



