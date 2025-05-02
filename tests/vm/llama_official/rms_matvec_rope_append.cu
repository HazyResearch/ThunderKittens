#include "llama.cuh"
#include "utils.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_1b_globals;

    template <typename Config, typename Globals>
    struct rms_qkv_rope_append
    {
        static constexpr int opcode = OPCODE_RMS_QKV_MatVecRopeAppend; // Op index within the layer -- controls which barrier to listen to.
        static constexpr int NUM_WEIGHT_PAGES = 4;

        static constexpr int PAGE_RMS_SCALE_ACTIVATION = 0;
        static constexpr int PAGE_ROPE_COS = PAGE_RMS_SCALE_ACTIVATION + 1;
        static constexpr int PAGE_ROPE_SIN = PAGE_ROPE_COS + 1;
        static constexpr int PAGE_WEIGHT_START = PAGE_ROPE_SIN + 1;
        static constexpr int PAGE_COUNT = PAGE_WEIGHT_START + NUM_WEIGHT_PAGES;

        static constexpr int K_BLK_START = 2048 / Globals::matvec_block_size;
        static constexpr int V_BLK_START = 2560 / Globals::matvec_block_size;

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
        __device__ static inline semaphore &weights_arrived(state<Config> &s, int id) { return s.semaphores()[id]; }
        __device__ static inline semaphore &activations_arrived(state<Config> &s) { return s.semaphores()[NUM_WEIGHT_PAGES]; }
        __device__ static inline semaphore &rms_scale_arrived(state<Config> &s) { return s.semaphores()[NUM_WEIGHT_PAGES + 1]; }
        __device__ static inline semaphore &rope_cos_arrived(state<Config> &s) { return s.semaphores()[NUM_WEIGHT_PAGES + 2]; }
        __device__ static inline semaphore &rope_sin_arrived(state<Config> &s) { return s.semaphores()[NUM_WEIGHT_PAGES + 3]; }
        __device__ static inline semaphore &outputs_arrived(state<Config> &s) { return s.semaphores()[NUM_WEIGHT_PAGES + 4]; }

        // Pages (very naive for now, no fine-grained usage)
        __device__ static inline int get_rms_scale_activation_page(state<Config> &s) { return s.pid(PAGE_RMS_SCALE_ACTIVATION); }
        __device__ static inline int get_weight_page(state<Config> &s, int offset) { return s.pid(PAGE_WEIGHT_START + offset); }
        __device__ static inline int get_rope_cos_page(state<Config> &s) { return s.pid(PAGE_ROPE_COS); }
        __device__ static inline int get_rope_sin_page(state<Config> &s) { return s.pid(PAGE_ROPE_SIN); }

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
            {

                // unused pages, then activation, then rms scale, then weights, then rope cos, then rope sin
                int ret_order[13] = {7, 8, 9, 10, 11, 12, PAGE_RMS_SCALE_ACTIVATION, PAGE_WEIGHT_START, PAGE_WEIGHT_START + 1, PAGE_WEIGHT_START + 2, PAGE_WEIGHT_START + 3, PAGE_ROPE_COS, PAGE_ROPE_SIN};
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
                init_semaphore(rope_cos_arrived(s), 1);
                init_semaphore(rope_sin_arrived(s), 1);
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
                    int rms_scale_activation_page = get_rms_scale_activation_page(s);
                    s.wait_page_ready(rms_scale_activation_page);
                    auto &rms_scale = *reinterpret_cast<sv_bf<2048>*>(s.pages[rms_scale_activation_page].ptr());
                    auto &activations = *reinterpret_cast<sv_bf<2048>*>(s.pages[rms_scale_activation_page].ptr(sizeof(sv_bf<2048>)));
                    s.record(TEVENT_TRIPLES_START);
                    tma::expect(rms_scale_arrived(s), rms_scale);
                    tma::load_async(rms_scale, g.attn_norm_weights, {inst.layer_idx, 0}, rms_scale_arrived(s));

                    for (int i = 0; i < NUM_WEIGHT_PAGES; i++)
                    {
                        // QKV projection weights
                        auto page_id = get_weight_page(s, i);

                        s.wait_page_ready(page_id);
                        auto &weight_chunk = reinterpret_cast<st_bf<16, 512> &>(s.pages[page_id]);
                        s.record(TEVENT_TRIPLES_START + 1 + i);
                        tma::expect(weights_arrived(s, i), weight_chunk);
                        tma::load_async(weight_chunk, g.qkv_weights, {inst.layer_idx, inst.qkv_block_idx, i}, weights_arrived(s, i));
                    }

                    // Rope cos
                    auto cos_page_id = get_rope_cos_page(s);
                    s.wait_page_ready(cos_page_id);
                    auto &rope_cos = reinterpret_cast<sv_fl<16> &>(s.pages[cos_page_id]);
                    s.record(TEVENT_TRIPLES_START + 5);
                    tma::expect(rope_cos_arrived(s), rope_cos);
                    tma::load_async(rope_cos, g.rope_cos, {0, 0, static_cast<int>(g.pos_id), inst.qkv_block_idx % 4}, rope_cos_arrived(s));

                    // Rope sin
                    auto sin_page_id = get_rope_sin_page(s);
                    s.wait_page_ready(sin_page_id);
                    auto &rope_sin = reinterpret_cast<sv_fl<16> &>(s.pages[sin_page_id]);
                    s.record(TEVENT_TRIPLES_START + 6);
                    tma::expect(rope_sin_arrived(s), rope_sin);
                    tma::load_async(rope_sin, g.rope_sin, {0, 0, static_cast<int>(g.pos_id), inst.qkv_block_idx % 4}, rope_sin_arrived(s));

                    // Activation
<<<<<<< HEAD
                    auto act_page_id = get_activation_page(s);
                    s.wait_page_ready(act_page_id);

=======
>>>>>>> 7f4b79c660f1113f06e5e9a90a03406cf2cacc55
                    s.record(TEVENT_AT_GMEM_WAIT);
                    while (inst.layer_idx > 0 && *(volatile int *)&g.Bar[{inst.layer_idx - 1, OPCODE_DownProjResidual - 1, 0}] < 512)
                        __nanosleep(20);
                    s.record(TEVENT_DONE_GMEM_WAIT);
                    s.record(TEVENT_TRIPLES_START + 7);
                    tma::expect(activations_arrived(s), activations);
                    tma::load_async(activations, g.hidden_states, {}, activations_arrived(s));
                }

                else if (laneid() >= PAGE_COUNT && laneid() < Config::NUM_PAGES)
                {
                    // Unused pages
                    auto pid = s.pid(laneid());
                    s.wait_page_ready(pid);
                    s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
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


                rms_norm(g, s, activations_vec, get_rms_scale_activation_page(s), activations_arrived(s), rms_scale_arrived(s), 16);

                matvec<float_rt_t, WARPS_PER_PAGE>(g, s, activations_vec, weights_arrived(s, page_index), get_weight_page(s, page_index), 0);

                group<Config::NUM_CONSUMER_WARPS>::sync(1); // must wait for all warps to finish atomic add

                // release pages
                for(int i = 0; i < NUM_WEIGHT_PAGES; i++) {
                    warp::arrive(s.page_finished[get_weight_page(s, i)]);
                }
                // release the activation page
                warp::arrive(s.page_finished[get_rms_scale_activation_page(s)]);

                // Step 5: Apply RoPE
                if (warpid() == 0)
                { // only a single warp needed from here!

                    // even for V, we need to cast from float to bf16
                    sv_fl<16> &qkv_proj_smem = *reinterpret_cast<sv_fl<16> *>(s.scratch());
                    sv_bf<16> &qkv_proj_smem_bf = *reinterpret_cast<sv_bf<16> *>(s.scratch());
                    warp::load(qkv_proj, qkv_proj_smem);

                    warp::sync();

                    int rope_cos_page = get_rope_cos_page(s);
                    int rope_sin_page = get_rope_sin_page(s);

                    if (inst.qkv_block_idx < V_BLK_START)
                    { // only Q & K need RoPE

                        sv_fl<16> &rope_cos_smem = reinterpret_cast<sv_fl<16> &>(s.pages[rope_cos_page]);
                        wait(rope_cos_arrived(s), 0);
                        if (laneid() == 0)
                        {
                            s.record(TEVENT_TRIPLES_END + 5);
                            s.record(TEVENT_CONSUMER_START + 48);
                        }
                        warp::load(rope_cos, rope_cos_smem);
                        // warp::arrive(s.page_finished[rope_cos_page], Config::NUM_CONSUMER_WARPS);
                        s.warp_finish_page(rope_cos_page, Config::NUM_CONSUMER_WARPS);

                        sv_fl<16> &rope_sin_smem = reinterpret_cast<sv_fl<16> &>(s.pages[rope_sin_page]);
                        wait(rope_sin_arrived(s), 0);
                        if (laneid() == 0)
                        {
                            s.record(TEVENT_TRIPLES_END + 6);
                            s.record(TEVENT_CONSUMER_START + 49);
                        }
                        warp::load(rope_sin, rope_sin_smem);
                        s.warp_finish_page(rope_sin_page, Config::NUM_CONSUMER_WARPS);

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
                    else
                    {
                        wait(rope_cos_arrived(s), 0);
                        s.warp_finish_page(rope_cos_page, Config::NUM_CONSUMER_WARPS);

                        wait(rope_sin_arrived(s), 0);
                        s.warp_finish_page(rope_sin_page, Config::NUM_CONSUMER_WARPS);
                    }

                    // Store back to the scratch
                    warp::store(qkv_proj_smem_bf, qkv_proj);
                    warp::sync();

                    warp::arrive(outputs_arrived(s));
                    if (kittens::group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
                    {
                        s.record(TEVENT_CONSUMER_END);
                    }
                }
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
