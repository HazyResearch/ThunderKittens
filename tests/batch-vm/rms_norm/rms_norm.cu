#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_8b_globals;

    template <
        typename Config,
        typename globals>
    struct pre_rms_norm
    {
        static constexpr int opcode = 1;
        static constexpr int REDUCTION_DIM_PER_WARP = globals::hidden_dim / Config::NUM_CONSUMER_WARPS;
        static constexpr int BATCH_SIZE_PER_PAGE = Config::PAGE_SIZE / (2 * globals::hidden_dim);
        static_assert(BATCH_SIZE_PER_PAGE == 4, "BATCH_SIZE_PER_PAGE must be 4");
        static_assert(globals::matmul_batch_block_size % BATCH_SIZE_PER_PAGE == 0, "matmul_batch_block_size must be a multiple of BATCH_SIZE_PER_PAGE");

        struct parsed_instruction
        {
            int layer_idx;
            int batch_start_idx;
            int batch_end_idx;
            int num_iters;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                layer_idx = instruction[1];
                batch_start_idx = instruction[2];
                batch_end_idx = instruction[3];
                num_iters = (batch_end_idx - batch_start_idx + BATCH_SIZE_PER_PAGE - 1) / BATCH_SIZE_PER_PAGE;
#ifndef NDEBUG
                assert(num_iters <= Config::NUM_PAGES);
#endif
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        // Semaphores
        __device__ static inline semaphore &activations_arrived(state<Config> &s, int i) { return s.semaphores()[i]; }
        __device__ static inline semaphore &outputs_arrived(state<Config> &s, int i) { return s.semaphores()[i + Config::NUM_PAGES]; }
        __device__ static inline semaphore &weights_arrived(state<Config> &s) { return s.semaphores()[2 * Config::NUM_PAGES]; }

        using activations = sv_bf<globals::hidden_dim>; 
        using warp_split_activations = sv_bf<globals::hidden_dim / Config::NUM_CONSUMER_WARPS>;

        static constexpr int ACTIVATIONS_SIZE = sizeof(activations::dtype) * globals::hidden_dim;

        using weights_vec = sv_bf<globals::hidden_dim>;
        using warp_split_weights_vec = sv_bf<globals::hidden_dim / Config::NUM_CONSUMER_WARPS>;

        // Pages layout and access methods
        static constexpr int WEIGHT_SIZE_UINT64 = sizeof(weights_vec::dtype) * weights_vec::length / sizeof(uint64_t);

        __device__ static inline activations &get_activations(state<Config> &s, int page_idx, int batch_idx){ return reinterpret_cast<activations *>(s.pages[s.pid(page_idx)].ptr())[batch_idx]; }
        __device__ static inline warp_split_activations &get_warp_split_activations(state<Config> &s, int page_idx, int batch_idx, int warp_idx){ return reinterpret_cast<warp_split_activations *>(s.pages[s.pid(page_idx)].ptr())[batch_idx * Config::NUM_CONSUMER_WARPS + warp_idx]; }
       
        __device__ static inline weights_vec &get_weights_vec(state<Config> &s) { return *reinterpret_cast<weights_vec *>(s.scratch()); }
        __device__ static inline warp_split_weights_vec &get_warp_split_weights_vec(state<Config> &s, int warp_idx){ return reinterpret_cast<warp_split_weights_vec *>(s.scratch())[warp_idx]; }

        __device__ static inline float & get_activation_accumulations_smem(state<Config> &s, int warp_idx, int idx) { return reinterpret_cast<float*>((uint64_t*) s.scratch() + WEIGHT_SIZE_UINT64)[warp_idx * BATCH_SIZE_PER_PAGE + idx]; }

        struct controller
        {
            static __device__ int release_lid(const globals &g, typename Config::instruction_t &instruction, int &query)
            {
                return query;
            }
            static __device__ int init_semaphores(const globals &g, state<Config> &s)
            {
                for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++)
                {
                    init_semaphore(activations_arrived(s, i), 1);
                    init_semaphore(outputs_arrived(s, i), Config::NUM_CONSUMER_WARPS);
                }

                init_semaphore(weights_arrived(s), 1);
                return 2 * Config::NUM_PAGES + 1;
            }
        };
        struct loader
        {
            static __device__ inline void gmem_wait(const globals &g, state<Config> &s) {}

            static __device__ void run(const globals &g, state<Config> &s)
            {
                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_LOADER_START);

                    printf("MAX_SHARED_MEMORY: %d\n", Config::MAX_SHARED_MEMORY);
                    printf("Scratch has pointer %p\n", s.scratch());
                    printf("Page 0 has pointer %p\n", s.pages[s.pid(0)].ptr());
                    printf("Page 1 has pointer %p\n", s.pages[s.pid(1)].ptr());
                    printf("Page 2 has pointer %p\n", s.pages[s.pid(2)].ptr());
                    printf("Page 3 has pointer %p\n", s.pages[s.pid(3)].ptr());
                    printf("Page 4 has pointer %p\n", s.pages[s.pid(4)].ptr());
                    printf("Page 5 has pointer %p\n", s.pages[s.pid(5)].ptr());

                    printf("Batch 0 activations have pointer %p\n", &get_activations(s, 0, 0));
                    printf("Batch 1 activations have pointer %p\n", &get_activations(s, 0, 1));
                    printf("Batch 2 activations have pointer %p\n", &get_activations(s, 0, 2));
                    printf("Batch 3 activations have pointer %p\n", &get_activations(s, 0, 3));

                    printf("Batch 4 activations have pointer %p\n", &get_activations(s, 1, 0));
                    printf("Batch 5 activations have pointer %p\n", &get_activations(s, 1, 1));
                    printf("Batch 6 activations have pointer %p\n", &get_activations(s, 1, 2));
                    printf("Batch 7 activations have pointer %p\n", &get_activations(s, 1, 3));

                    printf("Batch 8 activations have pointer %p\n", &get_activations(s, 2, 0));
                    printf("Batch 9 activations have pointer %p\n", &get_activations(s, 2, 1));
                    printf("Batch 10 activations have pointer %p\n", &get_activations(s, 2, 2));
                    printf("Batch 11 activations have pointer %p\n", &get_activations(s, 2, 3));

                    printf("Batch 12 activations have pointer %p\n", &get_activations(s, 3, 0));
                    printf("Batch 13 activations have pointer %p\n", &get_activations(s, 3, 1));
                    printf("Batch 14 activations have pointer %p\n", &get_activations(s, 3, 2));
                    printf("Batch 15 activations have pointer %p\n", &get_activations(s, 3, 3));

                    printf("Batch 16 activations have pointer %p\n", &get_activations(s, 4, 0));
                    printf("Batch 17 activations have pointer %p\n", &get_activations(s, 4, 1));
                    printf("Batch 18 activations have pointer %p\n", &get_activations(s, 4, 2));
                    printf("Batch 19 activations have pointer %p\n", &get_activations(s, 4, 3));

                    printf("Batch 20 activations have pointer %p\n", &get_activations(s, 5, 0));
                    printf("Batch 21 activations have pointer %p\n", &get_activations(s, 5, 1));
                    printf("Batch 22 activations have pointer %p\n", &get_activations(s, 5, 2));
                    printf("Batch 23 activations have pointer %p\n", &get_activations(s, 5, 3));

                    printf("Batch 24 activations have pointer %p\n", &get_activations(s, 6, 0));
                    printf("Batch 25 activations have pointer %p\n", &get_activations(s, 6, 1));
                    printf("Batch 26 activations have pointer %p\n", &get_activations(s, 6, 2));
                    printf("Batch 27 activations have pointer %p\n", &get_activations(s, 6, 3));
                }

                rv_bf<globals::hidden_dim> rms_scale;
                if (warp::laneid() == 1) printf("Storing rms_scale to page 0, batch 0\n");
                warp::store(get_activations(s, 0, 0), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 0, batch 1\n");
                warp::store(get_activations(s, 0, 1), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 0, batch 2\n");
                warp::store(get_activations(s, 0, 2), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 0, batch 3\n");
                warp::store(get_activations(s, 0, 3), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 1, batch 0\n");
                warp::store(get_activations(s, 1, 0), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 1, batch 1\n");
                warp::store(get_activations(s, 1, 1), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 1, batch 2\n");
                warp::store(get_activations(s, 1, 2), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 1, batch 3\n");
                warp::store(get_activations(s, 1, 3), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 2, batch 0\n");
                warp::store(get_activations(s, 2, 0), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 2, batch 1\n");
                warp::store(get_activations(s, 2, 1), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 2, batch 2\n");
                warp::store(get_activations(s, 2, 2), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 2, batch 3\n");
                warp::store(get_activations(s, 2, 3), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 3, batch 0\n");
                warp::store(get_activations(s, 3, 0), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 3, batch 1\n");
                warp::store(get_activations(s, 3, 1), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 3, batch 2\n");
                warp::store(get_activations(s, 3, 2), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 3, batch 3\n");
                warp::store(get_activations(s, 3, 3), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 4, batch 0\n");
                warp::store(get_activations(s, 4, 0), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 4, batch 1\n");
                warp::store(get_activations(s, 4, 1), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 4, batch 2\n");
                warp::store(get_activations(s, 4, 2), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 4, batch 3\n");
                warp::store(get_activations(s, 4, 3), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 5, batch 0\n");
                warp::store(get_activations(s, 5, 0), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 5, batch 1\n");
                warp::store(get_activations(s, 5, 1), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 5, batch 2\n");
                warp::store(get_activations(s, 5, 2), rms_scale);
                if (warp::laneid() == 1) printf("Storing rms_scale to page 5, batch 3\n");
                warp::store(get_activations(s, 5, 3), rms_scale);


                parsed_instruction inst{s};
                // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
                ((uint64_t *)s.scratch())[WEIGHT_SIZE_UINT64 + warp::laneid()] = 0;
                warp::sync(); // done, now we can proceed to other things.

                if (laneid() == 0)
                {
                    weights_vec &rms_scale = get_weights_vec(s);
                    tma::expect(weights_arrived(s), rms_scale);
                    auto &weights_global = g.attn_norm_weights; // g.*weights_ptr;
                    tma::load_async(rms_scale, weights_global, {inst.layer_idx, 0}, weights_arrived(s));

                    for (int page_idx = 0; page_idx < min(Config::NUM_PAGES, inst.num_iters); page_idx++) {
                        int batch_idx = inst.batch_start_idx + page_idx * BATCH_SIZE_PER_PAGE;
                        int batch_size = min(BATCH_SIZE_PER_PAGE, inst.batch_end_idx - batch_idx);
                        printf("LOADER page_idx: %d, batch_idx: %d, batch_size: %d\n", page_idx, batch_idx, batch_size);
                        s.wait_page_ready(s.pid(page_idx));
                        printf("LOADER past wait_page_ready\n");

                        // RMS scale
                        s.record(TEVENT_AT_GMEM_WAIT);
                        s.record(TEVENT_DONE_GMEM_WAIT);

                        tma::expect_bytes(
                            activations_arrived(s, page_idx), 
                            ACTIVATIONS_SIZE * batch_size
                        );

                        printf("Loading to page %d, which has pointer %p\n", page_idx, s.pages[s.pid(page_idx)].ptr());

                        for (int i = 0; i < batch_size; i++) {
                            printf("LOADER loading activations %d to page %d\n", batch_idx + i, page_idx);
                            tma::load_async(get_activations(s, page_idx, i), g.hidden_states, {batch_idx + i, 0}, activations_arrived(s, page_idx));
                        }

                        wait(activations_arrived(s, page_idx), 0);
                        printf("LOADER past wait\n");

                        for (int i = 0; i < batch_size; i++) {
                            printf("activations %d: ", batch_idx + i);
                            print(get_activations(s, page_idx, i));
                        }
                    }
                }

                warp::sync();
                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_LOADER_END);
                }

                if (laneid() == 0) printf("LOADER done!\n");
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

                using act_vec_fl = rv_fl<REDUCTION_DIM_PER_WARP>;

                act_vec_fl act_vecs[BATCH_SIZE_PER_PAGE];
                act_vec_fl act_vecs_copy[BATCH_SIZE_PER_PAGE];
                act_vec_fl rms_scale_vec;

                if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0) printf("CONSUMER waiting for weights_arrived\n");
                wait(weights_arrived(s), 0);
                if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0) printf("CONSUMER past wait_weights_arrived\n");

                warp::load(rms_scale_vec, get_warp_split_weights_vec(s, group<Config::NUM_CONSUMER_WARPS>::warpid()));

                for (int page_idx = 0; page_idx < min(Config::NUM_PAGES, inst.num_iters); page_idx++) {
                    int batch_size = min(BATCH_SIZE_PER_PAGE, inst.batch_end_idx - inst.batch_start_idx);
                    if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0) printf("CONSUMER waiting for activations_arrived\n");
                    wait(activations_arrived(s, page_idx), 0);
                    if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0) printf("CONSUMER past wait\n");

                    // Load the activations
                    for (int i = 0; i < batch_size; i++) {
                        // Load the activations
                        warp::load(act_vecs[i], get_warp_split_activations(s, page_idx, i, group<Config::NUM_CONSUMER_WARPS>::warpid()));

                        // if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0) {
                        //     printf("CONSUMER act_vecs[%d] from page %d: ", i, page_idx);
                        //     print(get_warp_split_activations(s, page_idx, i, group<Config::NUM_CONSUMER_WARPS>::warpid()));
                        // }

                        // Copy the activations
                        warp::copy(act_vecs_copy[i], act_vecs[i]);

                        // Square the activations
                        warp::mul(act_vecs_copy[i], act_vecs_copy[i], act_vecs_copy[i]);

                        // Sum the squared activations
                        float partial_sum = warp::sum(act_vecs_copy[i]);

                        if (warp::laneid() == 0) {
                            get_activation_accumulations_smem(s, group<Config::NUM_CONSUMER_WARPS>::warpid(), i) = partial_sum;
                        }
                    }

                    group<Config::NUM_CONSUMER_WARPS>::sync(0);

                    // Sum across consumer warps
                    for (int i = 0; i < batch_size; i++) {
                        // Compute the full sum
                        float full_sum = 0;
                        for (int j = 0; j < Config::NUM_CONSUMER_WARPS; j++) {
                            full_sum += get_activation_accumulations_smem(s, j, i);
                        }

                        // Compute the variance + rms scale
                        float variance = full_sum / (float)globals::hidden_dim;
                        float rms_scale = rsqrtf(variance + g.rms_norm_eps);

                        // Multiply by rms scale
                        warp::mul(act_vecs[i], act_vecs[i], rms_scale);

                        // Multiply by rms_scale_vec
                        warp::mul(act_vecs[i], act_vecs[i], rms_scale_vec);

                        // Store to output tile
                        warp::store(get_warp_split_activations(s, page_idx, i, group<Config::NUM_CONSUMER_WARPS>::warpid()), act_vecs[i]);
                    }

                    warp::sync();
                    if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0) printf("CONSUMER arriving at outputs_arrived\n");
                    warp::arrive(outputs_arrived(s, page_idx));
                    if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0) printf("CONSUMER past arrive\n");
                }
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
                    for (int page_idx = 0; page_idx < min(Config::NUM_PAGES, inst.num_iters); page_idx++) {
                        int batch_size = min(BATCH_SIZE_PER_PAGE, inst.batch_end_idx - inst.batch_start_idx);
                        if (laneid() == 0) printf("STORE waiting for outputs_arrived\n");
                        wait(outputs_arrived(s, page_idx), 0);
                        if (laneid() == 0) printf("STORE past wait\n");

                        for (int i = 0; i < batch_size; i++) {
                            if (laneid() == 0) {
                                printf("STORE saving stored_activations to (%d, %d): ", inst.batch_start_idx + page_idx * BATCH_SIZE_PER_PAGE + i, 0);
                                print(get_activations(s, page_idx, i));
                            }
                            tma::store_async<cache_policy::NORMAL>(
                                g.rms_rope_intermediates, 
                                get_activations(s, page_idx, i), 
                                {inst.batch_start_idx + page_idx * BATCH_SIZE_PER_PAGE + i, 0}
                            );
                        }
                        tma::store_async_read_wait();

                        s.finish_page(s.pid(page_idx), Config::NUM_CONSUMER_WARPS);
                    }
                }

                warp::sync();
                asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.

                if (warp::laneid() == 0)
                {
                    for (int page_idx = 0; page_idx < min(Config::NUM_PAGES, inst.num_iters); page_idx++) {
                        int batch_block_idx = inst.batch_start_idx / globals::matmul_batch_block_size + page_idx;
                        atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, batch_block_idx, 0}], BATCH_SIZE_PER_PAGE);
                    }
                }

                warp::sync();
                if (laneid() == 0)
                    s.record(TEVENT_STORE_END);
            }
        };
    };
};