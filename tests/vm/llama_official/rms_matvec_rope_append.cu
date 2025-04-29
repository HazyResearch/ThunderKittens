#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;
// using namespace kittens::prototype::vm;

namespace kittens::prototype::vm
{

    using globals = llama_1b_globals;

    template <typename Config, typename Globals>
    struct rms_qkv_rope_append
    {
        static constexpr int opcode = OPCODE_RMS_QKV_MatVecRopeAppend; // Op index within the layer -- controls which barrier to listen to.
        static constexpr int NUM_WEIGHT_PAGES = 4;

        static constexpr int PAGE_RMS_SCALE = 0;
        static constexpr int PAGE_ROPE_COS = PAGE_RMS_SCALE + 1;
        static constexpr int PAGE_ROPE_SIN = PAGE_ROPE_COS + 1;
        static constexpr int PAGE_WEIGHT_START = PAGE_ROPE_SIN + 1;
        static constexpr int PAGE_ACTIVATION = PAGE_WEIGHT_START + NUM_WEIGHT_PAGES;
        static constexpr int PAGE_COUNT = PAGE_ACTIVATION + 1; // 8

        static constexpr int K_BLK_START = 2048 / Globals::matvec_block_size;
        static constexpr int V_BLK_START = 2560 / Globals::matvec_block_size;

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
        __device__ static inline int get_rms_scale_page(state<Config> &s) { return s.pid(PAGE_RMS_SCALE); }
        __device__ static inline int get_weight_page(state<Config> &s, int offset) { return s.pid(PAGE_WEIGHT_START + offset); }
        __device__ static inline int get_rope_cos_page(state<Config> &s) { return s.pid(PAGE_ROPE_COS); }
        __device__ static inline int get_rope_sin_page(state<Config> &s) { return s.pid(PAGE_ROPE_SIN); }
        __device__ static inline int get_activation_page(state<Config> &s) { return s.pid(PAGE_ACTIVATION); }

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
            {

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
                ((int *)s.scratch())[laneid()] = 0;
                warp::sync(); // done, now we can proceed to other things.

                if (laneid() == 0)
                {

                    // RMS scale
                    s.wait_page_ready(get_rms_scale_page(s));
                    s.record(26);
                    auto &rms_scale = reinterpret_cast<sv_bf<2048> &>(s.pages[get_rms_scale_page(s)]);
                    s.record(TEVENT_TRIPLES_START);
                    tma::expect(rms_scale_arrived(s), rms_scale);
                    tma::load_async(rms_scale, g.attn_norm_weights, {inst.layer_idx, 0}, rms_scale_arrived(s));

                    for (int i = 0; i < 4; i++)
                    {
                        // QKV projection weights
                        auto page_id = get_weight_page(s, i);

                        s.wait_page_ready(page_id);
                        // s.record(16 + i);
                        auto &weight_chunk = reinterpret_cast<st_bf<16, 512> &>(s.pages[page_id]);
                        s.record(TEVENT_TRIPLES_START + 1 + i);
                        tma::expect(weights_arrived(s, i), weight_chunk);
                        tma::load_async(weight_chunk, g.qkv_weights, {inst.layer_idx, inst.qkv_block_idx, i}, weights_arrived(s, i));
                    }

                    // Rope cos
                    auto cos_page_id = get_rope_cos_page(s);
                    s.wait_page_ready(cos_page_id);
                    // s.record(28);
                    auto &rope_cos = reinterpret_cast<sv_fl<16> &>(s.pages[cos_page_id]);
                    s.record(TEVENT_TRIPLES_START + 5);
                    tma::expect(rope_cos_arrived(s), rope_cos);
                    tma::load_async(rope_cos, g.rope_cos, {0, 0, static_cast<int>(g.pos_id), inst.qkv_block_idx % 4}, rope_cos_arrived(s));

                    // Rope sin
                    auto sin_page_id = get_rope_sin_page(s);
                    s.wait_page_ready(sin_page_id);
                    // s.record(30);
                    auto &rope_sin = reinterpret_cast<sv_fl<16> &>(s.pages[sin_page_id]);
                    s.record(TEVENT_TRIPLES_START + 6);
                    tma::expect(rope_sin_arrived(s), rope_sin);
                    tma::load_async(rope_sin, g.rope_sin, {0, 0, static_cast<int>(g.pos_id), inst.qkv_block_idx % 4}, rope_sin_arrived(s));

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
                parsed_instruction inst{s};
                rt_fl<16, 128> weights, broadcast_activations;
                typename rt_fl<16, 128>::row_vec activations_vec;
                typename rt_fl<16, 128>::row_vec fl_activations_vec;
                rv_fl<Config::NUM_CONSUMER_WARPS> rms_partial_sums;
                typename rt_fl<16, 128>::row_vec rms_scale_vec;
                typename rt_fl<16, 128>::col_vec qkv_proj_partial_col_format;
                rv_fl<16> qkv_proj_partial;
                rv_fl<16> qkv_proj;
                rv_fl<16> rope_cos;
                rv_fl<16> rope_sin;
                shared_allocator al((int *)s.scratch());
                sv_fl<Config::NUM_CONSUMER_WARPS>(&smem_rms_partial_sums) = al.template allocate<sv_fl<Config::NUM_CONSUMER_WARPS>>();
                int group_id = warpgroup::groupid();
                int warp_id = warpgroup::warpid();

                // Step 1: Load hidden states into register
                wait(activations_arrived(s), 0);
                if (warpid() == 0 && laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_START + 15);
                }

                // reinterpret the activations page as sv_bf<128>[16]
                int activation_page = get_activation_page(s);
                sv_bf<128>(&activations_smem)[16] = reinterpret_cast<sv_bf<128>(&)[16]>(s.pages[activation_page]);
                warp::load(activations_vec, activations_smem[warpid()]); // 128 elements per warp
                warp::sync();
                warp::arrive(s.page_finished[activation_page]);

                // Step 2: Apply RMS normalization
                warp::copy(fl_activations_vec, activations_vec);                       // cast to float
                warp::mul(fl_activations_vec, fl_activations_vec, fl_activations_vec); // square
                float partial_sum = warp::sum(fl_activations_vec);

                // aggregate sums across the 16 consumer warps
                if (laneid() == 0)
                {
                    smem_rms_partial_sums[warpid()] = partial_sum;
                }

                group<16>::sync(0);
                warp::load(rms_partial_sums, smem_rms_partial_sums);
                warp::sync();
                float full_sum = warp::sum(rms_partial_sums);
                float variance = full_sum / 2048.0f;
                float rms_scale = rsqrtf(variance + g.rms_norm_eps);

                warp::copy(fl_activations_vec, activations_vec); // reuse the reg vec
                warp::mul(fl_activations_vec, fl_activations_vec, rms_scale);
                warp::copy(activations_vec, fl_activations_vec); // back to bf16

                // multiply by rms scale
                wait(rms_scale_arrived(s), 0);
                if (warpid() == 0 && laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_START + 8);
                }
                int rms_scale_page = get_rms_scale_page(s);
                sv_bf<128>(&rms_scale_smem)[16] = reinterpret_cast<sv_bf<128>(&)[16]>(s.pages[rms_scale_page]);
                warp::load(rms_scale_vec, rms_scale_smem[warpid()]);
                warp::sync();
                warp::arrive(s.page_finished[rms_scale_page]);
                warp::mul(activations_vec, activations_vec, rms_scale_vec);

                if (laneid() == 0)
                {
                    s.record(TEVENT_CONSUMER_START + 16 + warpid());
                }

                // Step 3: Load QKV projection weights into register
                wait(weights_arrived(s, group_id), 0);
                if (warpgroup::warpid() == 0 && laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_START + 9 + group_id);
                }

                int weight_page = get_weight_page(s, group_id);
                st_bf<16, 128>(&weights_smem)[4] = reinterpret_cast<st_bf<16, 128>(&)[4]>(s.pages[weight_page]);
                warp::load(weights, weights_smem[warp_id]);
                warp::sync();

                warp::arrive(s.page_finished[weight_page], Config::NUM_CONSUMER_WARPS / 4); // called by each warp in the warpgroup

                // Steo 4: Apply QKV projection
                warp::broadcast_col(broadcast_activations, activations_vec);
                warp::mul(broadcast_activations, broadcast_activations, weights);
                warp::row_sum(qkv_proj_partial_col_format, broadcast_activations);
                warp::copy(qkv_proj_partial, qkv_proj_partial_col_format);
                // now the first 16 threads have the output.
                if (laneid() < 16)
                {
                    // this might be a bad idea but yolo, it's probably an okay start
                    // and fortunately this is code where ncu will tell us if it's bad..
                    atomicAdd(&((float *)s.scratch())[laneid()], qkv_proj_partial[0][0]);
                }
                group<16>::sync(1); // must wait for all warps to finish atomic add

                if (laneid() == 0)
                {
                    s.record(TEVENT_CONSUMER_START + 32 + warpid());
                }

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
                            s.record(TEVENT_TRIPLES_START + 13);
                            s.record(TEVENT_CONSUMER_START + 48);
                        }
                        warp::load(rope_cos, rope_cos_smem);
                        warp::arrive(s.page_finished[rope_cos_page], Config::NUM_CONSUMER_WARPS);

                        sv_fl<16> &rope_sin_smem = reinterpret_cast<sv_fl<16> &>(s.pages[rope_sin_page]);
                        wait(rope_sin_arrived(s), 0);
                        if (laneid() == 0)
                        {
                            s.record(TEVENT_TRIPLES_START + 14);
                            s.record(TEVENT_CONSUMER_START + 49);
                        }
                        warp::load(rope_sin, rope_sin_smem);
                        warp::arrive(s.page_finished[rope_sin_page], Config::NUM_CONSUMER_WARPS);

                        // Fetch the neighbor values
                        int mod = (laneid() & 0b1) ? -1 : 1; // 1 for even, -1 for odd
                        warp::sync();
                        float pair_val = __shfl_sync(MASK_ALL, qkv_proj[0][0], laneid() + mod);

                        // Compute RoPE in-place
                        if (laneid() < 16)
                            // will clean this up later
                            qkv_proj[0][0] = float(qkv_proj[0][0]) * rope_cos[0][0] + float(-1 * mod) * float(pair_val) * rope_sin[0][0];
                    }
                    else
                    {
                        wait(rope_cos_arrived(s), 0);
                        warp::arrive(s.page_finished[rope_cos_page], Config::NUM_CONSUMER_WARPS);

                        wait(rope_sin_arrived(s), 0);
                        warp::arrive(s.page_finished[rope_sin_page], Config::NUM_CONSUMER_WARPS);
                    }

                    // Store back to the scratch
                    warp::store(qkv_proj_smem_bf, qkv_proj);
                    warp::sync();

                    warp::arrive(outputs_arrived(s));
                    if (kittens::group<16>::laneid() == 0)
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
                    s.record(TEVENT_MATVEC_STORE_START);
                }

                parsed_instruction inst{s};

                if (warp::laneid() == 0)
                {
                    sv_bf<16> &qkv_proj_smem = *reinterpret_cast<sv_bf<16> *>(s.scratch());
                    wait(outputs_arrived(s), 0);
                    s.record(TEVENT_MATVEC_OUTPUT_READY);

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
