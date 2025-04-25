#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;
// using namespace kittens::prototype::vm;


namespace kittens::prototype::vm {

    using globals = llama_1b_globals;

    template <typename Config, typename Globals>
    struct rms_qkv_rope_append {
        static constexpr int opcode = OPCODE_RMS_QKV_MatVecRopeAppend; // Op index within the layer -- controls which barrier to listen to.
        static constexpr int NUM_WEIGHT_PAGES = 4;
        static constexpr int K_BLK_START = 2048 / Globals::matvec_block_size;
        static constexpr int V_BLK_START = 2560 / Globals::matvec_block_size;

        using activation_sv = sv_fl<2048>;
        using activation_sv_bf = sv_bf<2048>;
        using rope_sv = sv_fl<16>;
        using out_sv_fl = sv_fl<16>;
        using out_sv_bf = sv_bf<16>;
        using weight_st = st_bf<16, 512>;

        struct parsed_instruction {
            int layer_idx;
            int qkv_block_idx;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction) {
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
        __device__ static inline int get_weight_page(state<Config> &s, int offset) { return s.pid(offset); }
        __device__ static inline int get_activation_page(state<Config> &s) { return s.pid(NUM_WEIGHT_PAGES); }
        __device__ static inline int get_rms_scale_page(state<Config> &s) { return s.pid(NUM_WEIGHT_PAGES + 1); }
        __device__ static inline int get_rope_cos_page(state<Config> &s) { return s.pid(NUM_WEIGHT_PAGES + 2); }
        __device__ static inline int get_rope_sin_page(state<Config> &s) { return s.pid(NUM_WEIGHT_PAGES + 3); }

        struct controller {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
                int ret_order[13] = {8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7};
                return ret_order[query];
            }
            static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
                for (int i = 0; i < 4; i++) init_semaphore(weights_arrived(s, i), 1);
                init_semaphore(activations_arrived(s), 1);
                init_semaphore(rms_scale_arrived(s), 1);
                init_semaphore(rope_cos_arrived(s), 1);
                init_semaphore(rope_sin_arrived(s), 1);
                init_semaphore(outputs_arrived(s), 1);
                s.record(1);
                return 9;
            }
        };
        struct loader {
            static __device__ void run(const Globals &g, state<Config> &s) {
                parsed_instruction inst{s};
                // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
                ((int *)s.scratch())[laneid()] = 0;
                warp::sync(); // done, now we can proceed to other things.
            
                if (laneid() < 4) {
                    // QKV projection weights
                    s.wait_page_ready(get_weight_page(s, laneid()));
                    s.record(16 + laneid());
                    auto &weight_chunk = reinterpret_cast<weight_st &>(s.pages[get_weight_page(s, laneid())]);
                    tma::expect(weights_arrived(s, laneid()), weight_chunk);
                    tma::load_async(weight_chunk, g.qkv_weights, {inst.layer_idx, inst.qkv_block_idx, laneid()}, weights_arrived(s, laneid()));
                } else if (laneid() == 4) {
                    // Activation
                    s.wait_page_ready(get_activation_page(s));
                    s.record(23);
                    while (inst.layer_idx > 0 && *(volatile int *)&g.Bar[{inst.layer_idx - 1, OPCODE_DownProjResidual - 1, 0}] != 512) __nanosleep(20);
                    s.record(24);
                    auto &activations = reinterpret_cast<activation_sv &>(s.pages[get_activation_page(s)]);
                    tma::expect(activations_arrived(s), activations);
                    tma::load_async(activations, g.hidden_states, {}, activations_arrived(s));
                } else if (laneid() == 5) {
                    // RMS scale
                    s.wait_page_ready(get_rms_scale_page(s));
                    s.record(26);
                    auto &rms_scale = reinterpret_cast<activation_sv_bf &>(s.pages[get_rms_scale_page(s)]);
                    tma::expect(rms_scale_arrived(s), rms_scale);
                    tma::load_async(rms_scale, g.attn_norm_weights, {inst.layer_idx, 0}, rms_scale_arrived(s));
                } else if (laneid() == 6) {
                    // Rope cos
                    s.wait_page_ready(get_rope_cos_page(s));
                    s.record(28);
                    auto &rope_cos = reinterpret_cast<rope_sv &>(s.pages[get_rope_cos_page(s)]);
                    tma::expect(rope_cos_arrived(s), rope_cos);
                    tma::load_async(rope_cos, g.rope_cos, {0, 0, static_cast<int>(g.pos_id), inst.qkv_block_idx % 4}, rope_cos_arrived(s));
                } else if (laneid() == 7) {
                    // Rope sin
                    s.wait_page_ready(get_rope_sin_page(s));
                    s.record(30);
                    auto &rope_sin = reinterpret_cast<rope_sv &>(s.pages[get_rope_sin_page(s)]);
                    tma::expect(rope_sin_arrived(s), rope_sin);
                    tma::load_async(rope_sin, g.rope_sin, {0, 0, static_cast<int>(g.pos_id), inst.qkv_block_idx % 4}, rope_sin_arrived(s));
                } else if (laneid() >= 8 && laneid() <= 12) {
                    // Unused pages
                    s.wait_page_ready(s.pid(laneid()));
                    arrive(s.page_finished[s.pid(laneid())], Config::NUM_CONSUMER_WARPS);
                }
            }
        };
        struct launcher {
            static __device__ void run(const Globals &g, state<Config> &s) {
                if (warp::laneid() == 0)
                {
                    s.wait_tensor_ready();
                    arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
                }
            }
        };
        struct consumer {
            static __device__ void run(const Globals &g, state<Config> &s) {
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
                shared_allocator al((int*)s.scratch());
                sv_fl<Config::NUM_CONSUMER_WARPS> (&smem_rms_partial_sums) = al.template allocate<sv_fl<Config::NUM_CONSUMER_WARPS>> ();
                int group_id = warpgroup::groupid();
                int warp_id = warpgroup::warpid();

                // Step 1: Load hidden states into register
                wait(activations_arrived(s), 0);
                if (laneid() == 0)
                    s.record(32 + warpid());
                // reinterpret the activations page as sv_bf<128>[16]
                int activation_page = get_activation_page(s);
                sv_bf<128>(&activations_smem)[16] = reinterpret_cast<sv_bf<128>(&)[16]>(s.pages[activation_page]);
                warp::load(activations_vec, activations_smem[warpid()]); // 128 elements per warp
                warp::sync();
                // warp::arrive(s.page_finished[activation_page]);

                // Step 2: Apply RMS normalization
                warp::copy(fl_activations_vec, activations_vec); // cast to float      
                warp::mul(fl_activations_vec, fl_activations_vec, fl_activations_vec); // square
                float partial_sum = warp::sum(fl_activations_vec);
                
                // aggregate sums across the 16 consumer warps
                if (laneid() == 0) {
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
                if (laneid() == 0)
                    s.record(48 + warpid());
                int rms_scale_page = get_rms_scale_page(s);
                sv_bf<128>(&rms_scale_smem)[16] = reinterpret_cast<sv_bf<128>(&)[16]>(s.pages[rms_scale_page]);
                warp::load(rms_scale_vec, rms_scale_smem[warpid()]);
                warp::sync();
                // warp::arrive(s.page_finished[rms_scale_page]);
                warp::mul(activations_vec, activations_vec, rms_scale_vec);

                // Step 3: Load QKV projection weights into register
                wait(weights_arrived(s, group_id), 0);
                if (laneid() == 0)
                    s.record(64 + warpid());
                int weight_page = get_weight_page(s, group_id);
                st_bf<16, 128>(&weights_smem)[4] = reinterpret_cast<st_bf<16, 128>(&)[4]>(s.pages[weight_page]);
                warp::load(weights, weights_smem[warp_id]);
                warp::sync();
                // warp::arrive(s.page_finished[weight_page], Config::NUM_CONSUMER_WARPS / 4); // called by each warp in the warpgroup
            
                // Steo 4: Apply QKV projection
                warp::broadcast_col(broadcast_activations, activations_vec);
                warp::mul(broadcast_activations, broadcast_activations, weights);
                warp::row_sum(qkv_proj_partial_col_format, broadcast_activations);
                warp::copy(qkv_proj_partial, qkv_proj_partial_col_format);
                // now the first 16 threads have the output.
                if (laneid() < 16) {
                    // this might be a bad idea but yolo, it's probably an okay start
                    // and fortunately this is code where ncu will tell us if it's bad..
                    atomicAdd(&((float *)s.scratch())[laneid()], qkv_proj_partial[0][0]);
                }
                group<16>::sync(1); // must wait for all warps to finish atomic add

                // Step 5: Apply RoPE
                if (warpid() == 0) { // only a single warp needed from here!
                    if (inst.qkv_block_idx < V_BLK_START) { // only Q & K need RoPE
                        out_sv_fl &qkv_proj_smem = *reinterpret_cast<out_sv_fl *>(s.scratch());
                        warp::load(qkv_proj, qkv_proj_smem);

                        int rope_cos_page = get_rope_cos_page(s);
                        rope_sv &rope_cos_smem = reinterpret_cast<rope_sv &>(s.pages[rope_cos_page]);
                        wait(rope_cos_arrived(s), 0);
                        if (laneid() == 0)
                            s.record(80);
                        warp::load(rope_cos, rope_cos_smem);
                        // warp::arrive(s.page_finished[rope_cos_page], Config::NUM_CONSUMER_WARPS);
                        
                        int rope_sin_page = get_rope_sin_page(s);
                        rope_sv &rope_sin_smem = reinterpret_cast<rope_sv &>(s.pages[rope_sin_page]);
                        wait(rope_sin_arrived(s), 0);
                        if (laneid() == 0)
                            s.record(81);
                        warp::load(rope_sin, rope_sin_smem);
                        // warp::arrive(s.page_finished[rope_sin_page], Config::NUM_CONSUMER_WARPS);

                        // Fetch the neighbor values
                        int mod = (laneid() & 0b1) ? -1 : 1; // 1 for even, -1 for odd
                        warp::sync();
                        float pair_val = __shfl_sync(MASK_ALL, qkv_proj[0][0], laneid() + mod);

                        // Compute RoPE in-place
                        if (laneid() < 16) {
                            // will clean this up later
                            // qkv_proj[0][0] = __bfloat162float(float(qkv_proj[0][0]) * rope_cos[0][0] + float(-1 * mod) * float(pair_val) * rope_sin[0][0]);
                            qkv_proj[0][0] = qkv_proj[0][0] * rope_cos[0][0] + (-1 * mod) * pair_val * rope_sin[0][0];

                        }

                        // Store back to the scratch
                        warp::store(qkv_proj_smem, qkv_proj);
                        warp::sync();
                    }

                    warp::arrive(outputs_arrived(s));
                    if (kittens::group<16>::laneid() == 0)
                        s.record(96);
                }
            }
        };
        struct storer {
            // Uses 4 full pages for outputs.
            static __device__ void run(const Globals &g, state<Config> &s) {
                parsed_instruction inst{s};

                if (warp::laneid() == 0) {
                    out_sv_fl &qkv_proj_smem = *reinterpret_cast<out_sv_fl *>(s.scratch());

                    out_sv_bf &qkv_proj_smem_bf = *reinterpret_cast<out_sv_bf *>(s.scratch());

                    wait(outputs_arrived(s), 0);
                    s.record(125);

                    if (inst.qkv_block_idx < K_BLK_START) { // Q
                        tma::store_async<cache_policy::NORMAL>(g.q_post_rope, qkv_proj_smem, {0, 0, 0, inst.qkv_block_idx});
                    } else if (inst.qkv_block_idx < V_BLK_START) { // K
                        int base_index = (inst.qkv_block_idx - K_BLK_START) * Globals::matvec_block_size;
                        int head_idx = base_index / Globals::head_dim;
                        int dim_idx = (base_index % Globals::head_dim) / Globals::matvec_block_size;
                        tma::store_async<cache_policy::NORMAL>(g.k_cache, qkv_proj_smem_bf, {inst.layer_idx, static_cast<int>(g.pos_id), head_idx, dim_idx});
                    } else { // V
                        int base_index = (inst.qkv_block_idx - V_BLK_START) * Globals::matvec_block_size;
                        int head_idx = base_index / Globals::head_dim;
                        int dim_idx = (base_index % Globals::head_dim) / Globals::matvec_block_size;
                        tma::store_async<cache_policy::NORMAL>(g.v_cache, qkv_proj_smem_bf, {inst.layer_idx, static_cast<int>(g.pos_id), head_idx, dim_idx});
                    }

                    tma::store_async_wait(); // not just read wait! full wait! must be visible in global!
                    s.record(126);
                }

                warp::sync();
                asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.

                if (warp::laneid() == 0)
                    atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, inst.qkv_block_idx / 4}], 1);

                // TODO
                // I have commented out all the page_finished stuff in consumer and moved them here.
                // Currently, the way we do it in consumer is incorrect. I have moved them here 
                // so I can get the entire thing running first
                if (laneid() < 8) {
                    s.wait_page_ready(s.pid(laneid()));
                    arrive(s.page_finished[s.pid(laneid())], Config::NUM_CONSUMER_WARPS);
                }
                if (laneid() == 0)
                    s.record(127);
            }
        };
    };
}
