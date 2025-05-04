#pragma once

#include "llama.cuh"

namespace kittens::prototype::vm
{
    template <typename Config, kittens::ducks::sv::all sv_t>
    __device__ static inline auto rms_norm(const sv_t &rms_scale_smem, const sv_t &activations_smem, float rms_norm_eps, void *scratch_memory)
    {
        using rv_t = rv_fl<sv_t::length>;
        rv_t activations_vec, sq_activations_vec, rms_scale_vec;

        warp::load(activations_vec, activations_smem);
        warp::copy(sq_activations_vec, activations_vec);
        warp::mul(sq_activations_vec, sq_activations_vec, sq_activations_vec);
        float partial_sum = warp::sum(sq_activations_vec);

        float *smem_rms_partial_sums = (float *)scratch_memory;
        if (laneid() == 0)
        {
            smem_rms_partial_sums[warpid()] = partial_sum;
        }
        group<Config::NUM_CONSUMER_WARPS>::sync(0);

        float full_sum = 0;
#pragma unroll
        for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++)
        {
            full_sum += smem_rms_partial_sums[i];
        }
        float variance = full_sum / 2048.0f;
        float rms_scale = rsqrtf(variance + rms_norm_eps);

        warp::mul(activations_vec, activations_vec, rms_scale);
        warp::load(rms_scale_vec, rms_scale_smem);
        warp::mul(activations_vec, activations_vec, rms_scale_vec);

        return activations_vec;
    }

    template <kittens::ducks::st::all st_t>
    __device__ static inline void matvec(sv_fl<st_t::rows> &out_smem, st_t &weights_smem, rv_fl<st_t::cols> &activations)
    {
        using rt_t = rt_fl<st_t::rows, st_t::cols>;
        using rrv_t = typename rt_t::row_vec;
        using rcv_t = typename rt_t::col_vec;
        using rv_t = rv_fl<st_t::rows>;
        using sv_t = sv_bf<st_t::rows>;

        rrv_t row_activations;
        warp::copy(row_activations, activations);

        rt_t broadcast_activations, weights;
        warp::broadcast_col(broadcast_activations, row_activations);
        warp::load(weights, weights_smem);
        warp::mul(broadcast_activations, broadcast_activations, weights);
        rcv_t sum_col_vec;
        warp::row_sum(sum_col_vec, broadcast_activations);

        rv_t sum_vec;
        warp::copy(sum_vec, sum_col_vec);

        if (laneid() < 16)
        {
            // this might be a bad idea but yolo, it's probably an okay start
            // and fortunately this is code where ncu will tell us if it's bad..
            atomicAdd(&out_smem[laneid()], sum_vec[0][0]);
        }
        warp::sync();
    }

    template <typename Config, typename Globals, typename rv_t>
    __device__ inline void rms_norm(Globals &g, state<Config> &s, rv_t &activations_vec, int rms_scale_activation_page, semaphore &activations_arrived, semaphore &rms_scale_arrived, int scratch_offset)
    {

        constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        sv_bf<REDUCTION_DIM_PER_WARP> *rms_scale_smem = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP> *>(s.pages[rms_scale_activation_page].ptr());
        sv_bf<REDUCTION_DIM_PER_WARP> *activations_smem = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP> *>(s.pages[rms_scale_activation_page].ptr(sizeof(sv_bf<2048>)));

        // Setup
        rv_t copy_activations_vec;
        rv_t rms_scale_vec;

        wait(activations_arrived, 0);

        if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
        {
            s.record(ACT_WAIT_DONE);
        }

        warp::load(activations_vec, activations_smem[warpid()]);

        // Step 2: Apply RMS normalization
        warp::copy(copy_activations_vec, activations_vec);                           // cast to float
        warp::mul(copy_activations_vec, copy_activations_vec, copy_activations_vec); // square
        float partial_sum = warp::sum(copy_activations_vec);

        auto smem_rms_partial_sums = ((float *)s.scratch()) + scratch_offset;
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

        float variance = full_sum / 2048.0f;
        float rms_scale = rsqrtf(variance + g.rms_norm_eps);

        warp::copy(copy_activations_vec, activations_vec); // unsquare
        warp::mul(copy_activations_vec, copy_activations_vec, rms_scale);
        warp::copy(activations_vec, copy_activations_vec);

        if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
        {
            s.record(RMS_SCALE_WAIT_START);
        }

        // multiply by rms scale
        wait(rms_scale_arrived, 0);

        if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
        {
            s.record(RMS_SCALE_WAIT_DONE);
        }

        // TODO
        // if (warpid() == 0 && laneid() == 0)
        // {
        //     s.record(TEVENT_TRIPLES_END);
        // }

        warp::load(rms_scale_vec, rms_scale_smem[warpid()]);

        warp::mul(activations_vec, activations_vec, rms_scale_vec);
    }

    template <typename rt_t, int WARPS_PER_PAGE, typename Config, typename Globals, typename rv_t>
    __device__ inline void matvec(Globals &g, state<Config> &s, rv_t &activations_vec, semaphore &weights_arrived, int weight_pid, int scratch_offset)
    {

        rt_t weights, broadcast_activations;
        typename rt_t::col_vec proj_partial_col_format;
        rv<float, rt_t::rows> proj_partial;

        int page_index = warpid() / WARPS_PER_PAGE;
        int index_in_page = warpid() % WARPS_PER_PAGE;

        if (index_in_page == 0 && laneid() == 0)
        {
            s.record(WEIGHT_WAIT_START + page_index);
        }

        wait(weights_arrived, 0);

        // TODO
        if (index_in_page == 0 && laneid() == 0)
        {
            s.record(WEIGHT_WAIT_DONE + page_index);
        }

        using st_slice_t = st_bf<rt_t::rows, rt_t::cols>;

        st_slice_t(&weights_smem)[WARPS_PER_PAGE] = reinterpret_cast<st_slice_t(&)[WARPS_PER_PAGE]>(s.pages[weight_pid]);
        warp::load(weights, weights_smem[index_in_page]);
        warp::sync();

        warp::broadcast_col(broadcast_activations, activations_vec);
        warp::mul(broadcast_activations, broadcast_activations, weights);
        warp::row_sum(proj_partial_col_format, broadcast_activations);
        warp::copy(proj_partial, proj_partial_col_format);

        auto smem_proj_partials = ((float *)s.scratch()) + scratch_offset;

        // now the first 16 threads have the output.

        if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
        {
            s.record(ATOMIC_ADD_START);
        }

        if (laneid() < 16)
        {
            // this might be a bad idea but yolo, it's probably an okay start
            // and fortunately this is code where ncu will tell us if it's bad..
            atomicAdd(&smem_proj_partials[laneid()], proj_partial[0][0]);
        }

        warp::sync();

        if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
        {
            s.record(ATOMIC_ADD_END);
        }
    }

    template <typename Config, typename Globals, typename parsed_instruction, typename pipeline_specifics>
    struct matvec_pipeline
    {
        static constexpr int INPUT_PIPELINE_STAGES = 3;
        static constexpr int STAGE_PAGES = 4;
        static constexpr int OUTPUT_PIPELINE_STAGES = 2;
        static constexpr int ACTIVATION_RMS_SCALE_PAGE = 0;
        static constexpr int WEIGHTS_START_PAGE = 1;

        static constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        static constexpr int SEM_COUNT = (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2;

        // Pages (very naive for now, no fine-grained usage)
        __device__ static inline int get_rms_scale_activation_page(state<Config> &s) { return s.pid(ACTIVATION_RMS_SCALE_PAGE); }

        __device__ static inline int get_weight_page(state<Config> &s, int stage, int offset) { return s.pid(WEIGHTS_START_PAGE + stage * STAGE_PAGES + offset); }

        __device__ static inline semaphore &weights_arrived(state<Config> &s, int stage) { return s.semaphores()[stage]; }
        __device__ static inline semaphore &weights_finished(state<Config> &s, int stage) { return s.semaphores()[INPUT_PIPELINE_STAGES + stage]; }
        __device__ static inline semaphore &outputs_arrived(state<Config> &s, int stage) { return s.semaphores()[2 * INPUT_PIPELINE_STAGES + stage]; }
        __device__ static inline semaphore &outputs_finished(state<Config> &s, int stage) { return s.semaphores()[2 * INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES + stage]; }

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
        __device__ static inline void loader_loop(state<Config> &s, const Globals &g)
        {
            parsed_instruction inst{s};

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
                    tma::load_async(weight_chunk, g.*WeightsPtr, {block_idx, i}, weights_arrived(s, input_stage));
                }

                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
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

                sv_fl<16> &logits_smem = *reinterpret_cast<sv_fl<16> *>((float *)s.scratch() + (32 * output_stage));
                sv_bf<16> &logits_smem_bf = *reinterpret_cast<sv_bf<16> *>((float *)s.scratch() + (32 * output_stage));

                wait(outputs_arrived(s, output_stage), (i % (2 * OUTPUT_PIPELINE_STAGES)) >= OUTPUT_PIPELINE_STAGES);

                pipeline_specifics::store(s, g, inst, i, output_stage);

                warp::arrive(outputs_finished(s, output_stage));
                output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
            }
        }
    };

}