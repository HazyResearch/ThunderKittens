#include "llama.cuh"
#include "utils.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_1b_globals;

    template <typename Config, typename Globals>
    struct rms_lm_head
    {
        static constexpr int opcode = OPCODE_RMS_LM_Head; // Op index within the layer -- controls which barrier to listen to.
        static constexpr int NUM_WEIGHT_PAGES = 4;

        static constexpr int PAGE_RMS_SCALE_ACTIVATION = 0;
        static constexpr int PAGE_WEIGHT_START = PAGE_RMS_SCALE_ACTIVATION + 1;
        static constexpr int PAGE_COUNT = PAGE_WEIGHT_START + NUM_WEIGHT_PAGES;

        static constexpr int SEM_COUNT = PAGE_COUNT + 1;

        static constexpr int EXPECTED_ARRIVAL_COUNT = 512;

        static constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        struct parsed_instruction
        {
            int out_block_idx;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                out_block_idx = instruction[1];
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        // Semaphores
        __device__ static inline semaphore &weights_arrived(state<Config> &s, int id) { return s.semaphores()[id]; }
        __device__ static inline semaphore &activations_arrived(state<Config> &s) { return s.semaphores()[NUM_WEIGHT_PAGES]; }
        __device__ static inline semaphore &rms_scale_arrived(state<Config> &s) { return s.semaphores()[NUM_WEIGHT_PAGES + 1]; }
        __device__ static inline semaphore &outputs_arrived(state<Config> &s) { return s.semaphores()[NUM_WEIGHT_PAGES + 2]; }

        // Pages (very naive for now, no fine-grained usage)
        __device__ static inline int get_rms_scale_activation_page(state<Config> &s) { return s.pid(PAGE_RMS_SCALE_ACTIVATION); }
        __device__ static inline int get_weight_page(state<Config> &s, int offset) { return s.pid(PAGE_WEIGHT_START + offset); }

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
            {

                // unused pages, then activation, then rms scale, then weights
                int ret_order[13] = {5, 6, 7, 8, 9, 10, 11, 12, PAGE_RMS_SCALE_ACTIVATION, PAGE_WEIGHT_START, PAGE_WEIGHT_START + 1, PAGE_WEIGHT_START + 2, PAGE_WEIGHT_START + 3};
                return ret_order[query];
            }
            static __device__ int init_semaphores(const Globals &g, state<Config> &s)
            {
                for (int i = 0; i < NUM_WEIGHT_PAGES; i++)
                {
                    init_semaphore(weights_arrived(s, i), 1);
                }
                init_semaphore(activations_arrived(s), 1);
                init_semaphore(rms_scale_arrived(s), 1);
                init_semaphore(outputs_arrived(s), Config::NUM_CONSUMER_WARPS);
                return SEM_COUNT;
            }
        };
        struct loader
        {
            static __device__ void run(const Globals &g, state<Config> &s)
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
                    int rms_scale_activation_page = get_rms_scale_activation_page(s);
                    s.wait_page_ready(rms_scale_activation_page);
                    auto &rms_scale = *reinterpret_cast<sv_bf<2048> *>(s.pages[rms_scale_activation_page].ptr());

                    s.record(TEVENT_TRIPLES_START);
                    tma::expect(rms_scale_arrived(s), rms_scale);
                    tma::load_async(rms_scale, g.lm_head_norm_weights, {}, rms_scale_arrived(s));

                    for (int i = 0; i < NUM_WEIGHT_PAGES; i++)
                    {
                        auto page_id = get_weight_page(s, i);

                        s.wait_page_ready(page_id);
                        auto &weight_chunk = reinterpret_cast<st_bf<16, 512> &>(s.pages[page_id]);
                        s.record(TEVENT_TRIPLES_START + 1 + i);
                        tma::expect(weights_arrived(s, i), weight_chunk);
                        tma::load_async(weight_chunk, g.lm_head_weights, {inst.out_block_idx, i}, weights_arrived(s, i));
                    }
                }

                else if (laneid() >= PAGE_COUNT && laneid() < Config::NUM_PAGES)
                {
                    // Unused pages
                    auto pid = s.pid(laneid());
                    s.wait_page_ready(pid);
                    s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
                }
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

                    int rms_scale_activation_page = get_rms_scale_activation_page(s);
                    s.wait_page_ready(rms_scale_activation_page);
                    auto &activations = *reinterpret_cast<sv_bf<2048> *>(s.pages[rms_scale_activation_page].ptr(sizeof(sv_bf<2048>)));

                    // Activation
                    s.record(TEVENT_AT_GMEM_WAIT);
                    while (*(volatile int *)&g.Bar[{globals::num_layers - 1, OPCODE_DownProjResidual - 1, 0}] < EXPECTED_ARRIVAL_COUNT) {
                        __nanosleep(20);
                    }
                    s.record(TEVENT_DONE_GMEM_WAIT);
                    tma::expect(activations_arrived(s), activations);
                    tma::load_async(activations, g.hidden_states, {}, activations_arrived(s));
                }
            }
        };
        struct consumer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {

                using float_rt_t = rt_fl<16, REDUCTION_DIM_PER_WARP>;
                using float_rv_t = rv_fl<16>;

                // Setup
                parsed_instruction inst{s};
                rv_fl<REDUCTION_DIM_PER_WARP> activations_vec_naive;
                typename float_rt_t::row_vec activations_vec;

                static_assert(Config::NUM_CONSUMER_WARPS % NUM_WEIGHT_PAGES == 0, "NUM_CONSUMER_WARPS must be divisible by NUM_WEIGHT_PAGES");
                constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / NUM_WEIGHT_PAGES;

                int page_index = warpid() / WARPS_PER_PAGE;

                rms_norm(g, s, activations_vec_naive, get_rms_scale_activation_page(s), activations_arrived(s), rms_scale_arrived(s), 16);

                // release the activation page
                warp::sync();
                s.warp_finish_page(get_rms_scale_activation_page(s), 1);
             
                warp::copy(activations_vec, activations_vec_naive);
                matvec<float_rt_t, WARPS_PER_PAGE>(g, s, activations_vec, weights_arrived(s, page_index), get_weight_page(s, page_index), 0);

                warp::sync();
                warp::arrive(outputs_arrived(s));

                // Release pages.
                for (int i = 0; i < NUM_WEIGHT_PAGES; i++)
                {
                    s.warp_finish_page(get_weight_page(s, i), 1);
                }

                if (laneid() == 0)
                {
                    s.record(TEVENT_CONSUMER_START + 32 + warpid());
                }
            }
        };
        struct storer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};

                sv_fl<16> &logits_smem = *reinterpret_cast<sv_fl<16> *>(s.scratch());
                sv_bf<16> &logits_smem_bf = *reinterpret_cast<sv_bf<16> *>(s.scratch());

                wait(outputs_arrived(s), 0);

                rv_fl<16> logits_rv;
                warp::load(logits_rv, logits_smem);
                warp::sync();
                warp::store(logits_smem_bf, logits_rv);

                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_OUTPUT_READY);

                    tma::store_async<cache_policy::NORMAL>(g.logits, logits_smem_bf, {0, 0, 0, inst.out_block_idx});

                    tma::store_async_wait(); // not just read wait! full wait! must be visible in global!
                }
            }
        };
    };
}
