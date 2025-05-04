#include "llama.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_1b_globals;

    template <typename Config, typename Globals>
    struct rms_lm_head
    {
        static constexpr int opcode = OPCODE_RMS_LM_Head; // Op index within the layer -- controls which barrier to listen to.
        static constexpr int INPUT_PIPELINE_STAGES = 3;
        static constexpr int STAGE_PAGES = 4;
        static constexpr int OUTPUT_PIPELINE_STAGES = 2;
        static constexpr int ACTIVATION_RMS_SCALE_PAGE = 0;
        static constexpr int WEIGHTS_START_PAGE = 1;

        static constexpr int SEM_COUNT = (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2 + 1;

        static constexpr int EXPECTED_ARRIVAL_COUNT = 512;

        static constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        // Semaphores
        __device__ static inline semaphore &activations_rms_scale_arrived(state<Config> &s) { return s.semaphores()[2 * (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES)]; }

        // Pages (very naive for now, no fine-grained usage)
        __device__ static inline int get_rms_scale_activation_page(state<Config> &s) { return s.pid(0); }
        __device__ static inline int get_weight_page(state<Config> &s, int stage, int offset) { return s.pid(1 + stage * STAGE_PAGES + offset); }

        struct parsed_instruction
        {
            int start_block_idx, end_block_idx, iters;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                start_block_idx = instruction[1];
                end_block_idx = instruction[2];
                iters = end_block_idx - start_block_idx;
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        struct pipeline_specifics
        {
            static __device__ inline void store(state<Config> &s, const Globals &g, parsed_instruction &inst, int output_idx, int output_stage, semaphore &sem, int bit)
            {
                wait(sem, bit);

                int block_idx = inst.start_block_idx + output_idx;

                sv_fl<16> &logits_smem = *reinterpret_cast<sv_fl<16> *>((float *)s.scratch() + (32 * output_stage));
                sv_bf<16> &logits_smem_bf = *reinterpret_cast<sv_bf<16> *>((float *)s.scratch() + (32 * output_stage));

                rv_fl<16> logits_rv;
                warp::load(logits_rv, logits_smem);
                warp::store(logits_smem_bf, logits_rv);
                warp::sync();

                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_OUTPUT_READY);

                    tma::store_async<cache_policy::NORMAL>(g.logits, logits_smem_bf, {0, 0, 0, block_idx});

                    tma::store_async_read_wait(); // not just read wait! full wait! must be visible in global!
                }

                warp::zero(logits_smem);
                warp::sync();
            }
        };

        using pipeline = matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics>;

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

                pipeline::loader_loop<&Globals::lm_head_weights>(s, g, 0);
            }
        };
        struct launcher
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                if (laneid() == 0)
                {
                    s.wait_tensor_ready();
                    arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);

                    parsed_instruction inst{s};

                    auto activation_page = pipeline::get_activation_page(s);
                    auto &sem = pipeline::activations_arrived(s);

                    s.wait_page_ready(activation_page);
                    auto &activations = *reinterpret_cast<sv_bf<2048> *>(s.pages[activation_page].ptr(sizeof(sv_bf<2048>)));
                    auto &rms_scale = *reinterpret_cast<sv_bf<2048> *>(s.pages[activation_page].ptr());
                    tma::expect(sem, activations, rms_scale);
                    tma::load_async(rms_scale, g.lm_head_norm_weights, {}, sem);

                    // Activation
                    s.record(TEVENT_AT_GMEM_WAIT);
                    while (*(volatile int *)&g.Bar[{globals::num_layers - 1, OPCODE_DownProjResidual - 1, 0}] < EXPECTED_ARRIVAL_COUNT)
                    {
                        __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                    }
                    s.record(TEVENT_DONE_GMEM_WAIT);
                    tma::load_async(activations, g.hidden_states, {}, sem);
                }
            }
        };
        struct consumer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                auto activation_page = pipeline::get_activation_page(s);
                auto &sem = pipeline::activations_arrived(s);

                wait(sem, 0);

                auto rms_scale_smem = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP> *>(s.pages[activation_page].ptr());
                auto activations_smem = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP> *>(s.pages[activation_page].ptr(sizeof(sv_bf<2048>)));

                auto activations_vec = rms_norm<Config>(rms_scale_smem[warpid()], activations_smem[warpid()], g.rms_norm_eps, (void *)((uint8_t *)s.scratch() + (64 * 12)));

                warp::sync();
                s.warp_finish_page(activation_page, 1);

                pipeline::consumer_loop(s, g, activations_vec);
            }
        };
        struct storer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                pipeline::storer_loop(s, g);
            }
        };
    };
}
