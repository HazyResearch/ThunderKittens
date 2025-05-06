#include "llama.cuh"
#include "utils.cuh"

using namespace kittens;
using namespace kittens::prototype;

/*
What do we need to do here... normalize entire hidden state vector 
Need to prevent activation values from getting too large or small as go through layers 
Calculate rms_norm for one entire hidden state 1 x 8192 
*/

namespace kittens::prototype::vm
{

    using globals = llama_70b_globals;

    template <typename Config, typename Globals>
    struct rms_qkv_rope_append
    {
        static constexpr int opcode = OPCODE_RMS_QKV_MatVecRopeAppend; // Op index within the layer -- controls which barrier to listen to.
        static constexpr int NUM_WEIGHT_PAGES = 4;

        static constexpr int RMS_AND_ACTIVATION_PAGE = 0;
        static constexpr int OUTPUT_PAGE = 1;

        static constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        struct parsed_instruction
        {
            int layer_idx;
            int qkv_block_idx;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                layer_idx = instruction[1];     // in units of 1
                qkv_block_idx = instruction[2]; // in units of 16 elements
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        // Semaphores
        __device__ static inline semaphore &activations_arrived(state<Config> &s) { return s.semaphores()[NUM_WEIGHT_PAGES]; }
        __device__ static inline semaphore &rms_scale_arrived(state<Config> &s) { return s.semaphores()[NUM_WEIGHT_PAGES + 1]; }
        __device__ static inline semaphore &outputs_arrived(state<Config> &s) { return s.semaphores()[NUM_WEIGHT_PAGES + 4]; }

        // Pages (very naive for now, no fine-grained usage)
        __device__ static inline int get_rms_scale_page(state<Config> &s) { return s.pid(PAGE_RMS_SCALE); }
        __device__ static inline int get_activation_page(state<Config> &s) { return s.pid(PAGE_ACTIVATION); }

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
            {
                // TODO: How is proper page order decided??? 
                // unused pages, then activation, then rms scale, then weights, then rope cos, then rope sin
                int ret_order[13] = {8, 9, 10, 11, 12, PAGE_ACTIVATION, PAGE_RMS_SCALE, PAGE_WEIGHT_START, PAGE_WEIGHT_START + 1, PAGE_WEIGHT_START + 2, PAGE_WEIGHT_START + 3, PAGE_ROPE_COS, PAGE_ROPE_SIN};
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
                init_semaphore(outputs_arrived(s), 1);
                return 9;
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
                    s.wait_page_ready(get_rms_scale_page(s));
                    auto &rms_scale = reinterpret_cast<sv_bf<2048> &>(s.pages[get_rms_scale_page(s)]);
                    s.record(TEVENT_TRIPLES_START);
                    tma::expect(rms_scale_arrived(s), rms_scale);
                    tma::load_async(rms_scale, g.attn_norm_weights, {inst.layer_idx, 0}, rms_scale_arrived(s));

                    // Activation
                    auto act_page_id = get_activation_page(s);
                    s.wait_page_ready(act_page_id);
                    s.record(TEVENT_AT_GMEM_WAIT);
                    while (inst.layer_idx > 0 && *(volatile int *)&g.Bar[{inst.layer_idx - 1, OPCODE_DownProjResidual - 1, 0}] < 512)
                        __nanosleep(20);
                    s.record(TEVENT_DONE_GMEM_WAIT);
                    auto &activations = reinterpret_cast<sv_bf<2048> &>(s.pages[act_page_id]);
                    s.record(TEVENT_TRIPLES_START + 7);
                    tma::expect(activations_arrived(s), activations);
                    tma::load_async(activations, g.hidden_states, {}, activations_arrived(s));
                }

                else if (laneid() >= 8 && laneid() <= 12)
                {
                    // Unused pages
                    s.wait_page_ready(s.pid(laneid()));
                    arrive(s.page_finished[s.pid(laneid())], Config::NUM_CONSUMER_WARPS);
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
            static __device__ void run(const Globals &g, state<Config> &s)
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
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_CONSUMER_START + warpid());
                }

                // Setup
                using float_rt_t = rt_fl<16, REDUCTION_DIM_PER_WARP>;
                using float_rv_t = rv_fl<16>;

                parsed_instruction inst{s};
                typename float_rt_t::row_vec activations_vec;
                float_rv_t qkv_proj, rope_cos, rope_sin;

                static_assert(Config::NUM_CONSUMER_WARPS % NUM_WEIGHT_PAGES == 0, "NUM_CONSUMER_WARPS must be divisible by NUM_WEIGHT_PAGES");
                constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / NUM_WEIGHT_PAGES;

                int page_index = warpid() / WARPS_PER_PAGE;

                rms_norm(g, s, activations_vec, get_activation_page(s), get_rms_scale_page(s), activations_arrived(s), rms_scale_arrived(s), 16);
            }
        };
        struct storer
        {
            // Uses 4 full pages for outputs.
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_STORE_START);
                }

                parsed_instruction inst{s};

                if (warp::laneid() == 0)
                {
                    sv_bf<16> &qkv_proj_smem = *reinterpret_cast<sv_bf<16> *>(s.scratch());
                    wait(outputs_arrived(s), 0);
                    s.record(TEVENT_TRIPLES_OUTPUT_READY);

                    if (inst.qkv_block_idx < K_BLK_START)
                    { // Q
                        tma::store_async<cache_policy::NORMAL>(g.q_post_rope, qkv_proj_smem, {0, 0, 0, inst.qkv_block_idx});
                    }
                    else if (inst.qkv_block_idx < V_BLK_START)
                    { // K
                        int base_index = (inst.qkv_block_idx - K_BLK_START) * Globals::matvec_block_size;
                        int head_idx = base_index / Globals::head_dim;
                        int dim_idx = (base_index % Globals::head_dim) / Globals::matvec_block_size;
                        tma::store_async<cache_policy::NORMAL>(g.k_cache, qkv_proj_smem, {inst.layer_idx, static_cast<int>(g.pos_id), head_idx, dim_idx});
                    }
                    else
                    { // V
                        int base_index = (inst.qkv_block_idx - V_BLK_START) * Globals::matvec_block_size;
                        int head_idx = base_index / Globals::head_dim;
                        int dim_idx = (base_index % Globals::head_dim) / Globals::matvec_block_size;
                        tma::store_async<cache_policy::NORMAL>(g.v_cache, qkv_proj_smem, {inst.layer_idx, static_cast<int>(g.pos_id), head_idx, dim_idx});
                    }

                    tma::store_async_wait(); // not just read wait! full wait! must be visible in global!
                    s.record(126);
                }

                warp::sync();
                asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.

                if (warp::laneid() == 0)
                    atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, inst.qkv_block_idx / 4}], 1);

                warp::sync();
                if (laneid() == 0)
                    s.record(TEVENT_STORE_END);
            }
        };
    };
}



