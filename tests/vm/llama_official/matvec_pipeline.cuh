
#pragma once

#include "llama.cuh"

namespace kittens::prototype::vm
{

    template <typename Config, typename Globals, typename parsed_instruction, typename pipeline_specifics>
    struct matvec_pipeline
    {
        static constexpr int INPUT_PIPELINE_STAGES = 3;
        static constexpr int STAGE_PAGES = 4;
        static constexpr int OUTPUT_PIPELINE_STAGES = 2;
        static constexpr int ACTIVATION_PAGE = 0;
        static constexpr int WEIGHTS_START_PAGE = 1;

        static constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        static constexpr int SEM_COUNT = 1 + (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2;

        // Pages (very naive for now, no fine-grained usage)
        __device__ static inline int get_activation_page(state<Config> &s) { return s.pid(ACTIVATION_PAGE); }

        __device__ static inline int get_weight_page(state<Config> &s, int stage, int offset) { return s.pid(WEIGHTS_START_PAGE + stage * STAGE_PAGES + offset); }

        __device__ static inline semaphore &activations_arrived(state<Config> &s) { return s.semaphores()[0]; }
        __device__ static inline semaphore &weights_arrived(state<Config> &s, int stage) { return s.semaphores()[1 + stage]; }
        __device__ static inline semaphore &weights_finished(state<Config> &s, int stage) { return s.semaphores()[1 + INPUT_PIPELINE_STAGES + stage]; }
        __device__ static inline semaphore &outputs_arrived(state<Config> &s, int stage) { return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES + stage]; }
        __device__ static inline semaphore &outputs_finished(state<Config> &s, int stage) { return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES + stage]; }

        __device__ static inline int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
        {
            // NOTE: assumes a three stage pipeline

            parsed_instruction inst{instruction};
            // unused pages, then activation, then weights
            if (inst.iters % 3 == 1)
            {
                int ret_order[13] = {0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4};
                return ret_order[query];
            }
            else if (inst.iters % 3 == 2)
            {
                int ret_order[13] = {0, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8};
                return ret_order[query];
            }
            else
            {
                int ret_order[13] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                return ret_order[query];
            }
        }

        __device__ static inline int init_semaphores(state<Config> &s)
        {
            init_semaphore(activations_arrived(s), 1);
            for (int i = 0; i < INPUT_PIPELINE_STAGES; i++)
            {
                init_semaphore(weights_arrived(s, i), 1);
                init_semaphore(weights_finished(s, i), Config::NUM_CONSUMER_WARPS);
            }
            for (int i = 0; i < OUTPUT_PIPELINE_STAGES; i++)
            {
                init_semaphore(outputs_arrived(s, i), Config::NUM_CONSUMER_WARPS);
                init_semaphore(outputs_finished(s, i), 1);
            }
            return SEM_COUNT;
        }

        template <auto WeightsPtr>
        __device__ static inline void loader_loop(state<Config> &s, const Globals &g, int layer_idx)
        {
            parsed_instruction inst{s};

            auto needed_pages = 1 + min(inst.iters, INPUT_PIPELINE_STAGES) * STAGE_PAGES;

            if (laneid() == 0)
            {

                int input_stage = 0;
                for (int iter = 0; iter < inst.iters; iter++)
                {
                    wait(weights_finished(s, input_stage), (iter % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);

                    int block_idx = inst.start_block_idx + iter;

                    tma::expect_bytes(weights_arrived(s, input_stage), sizeof(bf16) * 2048 * 16);
#pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        int weight_page = get_weight_page(s, input_stage, i);
                        if (iter < INPUT_PIPELINE_STAGES)
                            s.wait_page_ready(weight_page);
                        auto &weight_chunk = reinterpret_cast<st_bf<16, 512> &>(s.pages[weight_page]);
                        tma::load_async(weight_chunk, g.*WeightsPtr, {layer_idx, block_idx, i}, weights_arrived(s, input_stage));
                    }

                    input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
                }
            }
            else if (laneid() >= needed_pages && laneid() < Config::NUM_PAGES)
            {
                auto pid = s.pid(laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
        }

        template <typename rv_t>
        __device__ static inline void consumer_loop(state<Config> &s, const Globals &g, rv_t &activations_vec)
        {
            // Setup
            parsed_instruction inst{s};

            static_assert(Config::NUM_CONSUMER_WARPS % STAGE_PAGES == 0, "NUM_CONSUMER_WARPS must be divisible by STAGE_PAGES");
            constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / STAGE_PAGES;

            int page_index = warpid() / WARPS_PER_PAGE;

            int input_stage = 0, output_stage = 0;
            for (int i = 0; i < inst.iters; i++)
            {

                int weight_page = get_weight_page(s, input_stage, page_index);
                wait(weights_arrived(s, input_stage), (i % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);
                wait(outputs_finished(s, output_stage), (i % (2 * OUTPUT_PIPELINE_STAGES)) < OUTPUT_PIPELINE_STAGES);
                st_bf<16, REDUCTION_DIM_PER_WARP> &weights = reinterpret_cast<st_bf<16, REDUCTION_DIM_PER_WARP> *>(s.pages[weight_page].ptr())[warpid() % WARPS_PER_PAGE];
                sv_fl<16> &out_smem = *reinterpret_cast<sv_fl<16> *>((float *)s.scratch() + (32 * output_stage));

                matvec(out_smem, weights, activations_vec);

                warp::sync();
                warp::arrive(outputs_arrived(s, output_stage));
                warp::arrive(weights_finished(s, input_stage));

                if (i >= inst.iters - 3)
                {
// Release pages.
#pragma unroll
                    for (int j = 0; j < STAGE_PAGES; j++)
                    {
                        s.warp_finish_page(get_weight_page(s, input_stage, j), 1);
                    }
                }

                group<Config::NUM_CONSUMER_WARPS>::sync(0);

                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
                output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
            }
        }

        __device__ static inline void storer_loop(state<Config> &s, const Globals &g)
        {
            parsed_instruction inst{s};

            int output_stage = 0;
            for (int i = 0; i < inst.iters; i++)
            {
                int block_idx = inst.start_block_idx + i;

                auto &sem = outputs_arrived(s, output_stage);
                auto bit = (i % (2 * OUTPUT_PIPELINE_STAGES)) >= OUTPUT_PIPELINE_STAGES;

                pipeline_specifics::store(s, g, inst, i, output_stage, sem, bit);

                warp::arrive(outputs_finished(s, output_stage));
                output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
            }
        }
    };
}