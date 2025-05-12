#pragma once

#include "mlp.cuh"

namespace kittens::prototype::vm {


#ifdef KITTENS_BLACKWELL
    template <kittens::ducks::st::all st_t>
    __device__ static inline void matvec(sv_fl<st_t::rows> &out_smem, st_t &weights_smem, rv_fl<st_t::cols> &activations) {
        using rt_t = rt_bf<st_t::rows, st_t::cols>;
        using rrv_t = typename rt_t::row_vec;
        using rcv_t = typename rt_fl<16,16>::col_vec;
        using rv_t = rv_fl<st_t::rows>;

        rrv_t row_activations;
        warp::copy(row_activations, activations);

        rt_t broadcast_activations, weights;
        warp::broadcast_col(broadcast_activations, row_activations);
        warp::load(weights, weights_smem);
        rt_fl<16,16> out_activations;
        warp::zero(out_activations);
        warp::mma_ABt(out_activations, weights, broadcast_activations, out_activations);
        rcv_t sum_col_vec;
        warp::row_max(sum_col_vec, out_activations);

        rv_t sum_vec;
        warp::copy(sum_vec, sum_col_vec);

        if (laneid() < 16) {
            // this might be a bad idea but yolo, it's probably an okay start
            // and fortunately this is code where ncu will tell us if it's bad..
            atomicAdd(&out_smem[laneid()], sum_vec[0][0]);
        }
        warp::sync();
    }
#else
    template <kittens::ducks::st::all st_t>
    __device__ static inline void matvec(sv_fl<st_t::rows> &out_smem, st_t &weights_smem, rv_fl<st_t::cols> &activations) {
        using rt_t = rt_fl<st_t::rows, st_t::cols>;
        using rrv_t = typename rt_t::row_vec;
        using rcv_t = typename rt_t::col_vec;
        using rv_t = rv_fl<st_t::rows>;

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

        if (laneid() < 16) {
            // this might be a bad idea but yolo, it's probably an okay start
            // and fortunately this is code where ncu will tell us if it's bad.
            atomicAdd(&out_smem[laneid()], sum_vec[0][0]);
        }
        warp::sync();
    }
#endif

    template <typename Config, typename Globals, typename parsed_instruction, typename pipeline_specifics>
    struct matvec_pipeline {
        static constexpr int INPUT_PIPELINE_STAGES = 3;
        static constexpr int OUTPUT_PIPELINE_STAGES = 3;
        static constexpr int STAGE_PAGES = 4;
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

        __device__ static inline sv_bf<Globals::hidden_dim> &get_activations(state<Config> &s) { return *reinterpret_cast<sv_bf<Globals::hidden_dim> *>(s.pages[get_activation_page(s)].ptr()); }

        __device__ static inline int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
            // NOTE: assumes a three stage pipeline

            parsed_instruction inst{instruction};
            // unused pages, then activation, then weights

            static_assert(INPUT_PIPELINE_STAGES == 3, "INPUT_PIPELINE_STAGES must be 3");

            auto iters = inst.iters;
            auto remainder = iters % INPUT_PIPELINE_STAGES;

            // special handling for 1 and 2 because only then do
            // we free pages before the activation/rms scale (page 0)
            if (iters == 1) {
                int ret_order[13] = {5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4};
                return ret_order[query];
            }
            else if (iters == 2) {
                int ret_order[13] = {9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8};
                return ret_order[query];
            }
            else if (remainder == 1) {
                int ret_order[13] = {0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4};
                return ret_order[query];
            }
            else if (remainder == 2) {
                int ret_order[13] = {0, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8};
                return ret_order[query];
            }
            else {
                int ret_order[13] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                return ret_order[query];
            }
        }

        __device__ static inline int init_semaphores(state<Config> &s) {
            init_semaphore(activations_arrived(s), 1);
            for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
                init_semaphore(weights_arrived(s, i), 1);
                init_semaphore(weights_finished(s, i), Config::NUM_CONSUMER_WARPS);
            }
            for (int i = 0; i < OUTPUT_PIPELINE_STAGES; i++) {
                init_semaphore(outputs_arrived(s, i), Config::NUM_CONSUMER_WARPS);
                init_semaphore(outputs_finished(s, i), 1);
            }
            return SEM_COUNT;
        }

        __device__ static inline void loader_loop(state<Config> &s, const Globals &g) {
            parsed_instruction inst{s};

            auto needed_pages = 1 + min(inst.iters, INPUT_PIPELINE_STAGES) * STAGE_PAGES;

            if (laneid() == 0) {

                int input_stage = 0;
                for (int iter = 0; iter < inst.iters; iter++) {
                    wait(weights_finished(s, input_stage), (iter % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);

                    auto &sem = weights_arrived(s, input_stage);
                    tma::expect_bytes(sem, sizeof(bf16) * 2048 * 16);
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int weight_page = get_weight_page(s, input_stage, i);
                        if (iter < INPUT_PIPELINE_STAGES) {
                            s.wait_page_ready(weight_page);
                        }
                        auto &weight_chunk = reinterpret_cast<st_bf<16, 512> &>(s.pages[weight_page]);

                        if (iter == 0 && i == 0) {
                            s.record(TEVENT_FIRST_LOAD);
                        }
                        else if (iter == inst.iters - 1 && i == 3) {
                            s.record(TEVENT_LAST_LOAD);
                        }

                        pipeline_specifics::load_iter(s, g, inst, iter, i, weight_chunk, sem);
                    }

                    input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
                }
            }
            else if (laneid() >= needed_pages && laneid() < Config::NUM_PAGES) {
                auto pid = s.pid(laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
        }

        template <typename rv_t>
        __device__ static inline void consumer_loop(state<Config> &s, const Globals &g, rv_t &activations_vec) {
            // Setup
            parsed_instruction inst{s};

            static_assert(Config::NUM_CONSUMER_WARPS % STAGE_PAGES == 0, "NUM_CONSUMER_WARPS must be divisible by STAGE_PAGES");
            constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / STAGE_PAGES;

            int page_index = warpid() / WARPS_PER_PAGE;

            int input_stage = 0, output_stage = 0;
            for (int i = 0; i < inst.iters; i++) {
                int weight_page = get_weight_page(s, input_stage, page_index);
                wait(weights_arrived(s, input_stage), (i % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);
                wait(outputs_finished(s, output_stage), (i % (2 * OUTPUT_PIPELINE_STAGES)) < OUTPUT_PIPELINE_STAGES);
                st_bf<16, REDUCTION_DIM_PER_WARP> &weights = reinterpret_cast<st_bf<16, REDUCTION_DIM_PER_WARP> *>(s.pages[weight_page].ptr())[warpid() % WARPS_PER_PAGE];
                sv_fl<16> &out_smem = *reinterpret_cast<sv_fl<16> *>((float *)s.scratch() + (32 * output_stage));

                if (i == 0) {
                    s.record(TEVENT_FIRST_USE);
                }
                else if (i == inst.iters - 1) {
                    s.record(TEVENT_LAST_USE);
                }

                matvec(out_smem, weights, activations_vec);

                warp::sync();
                warp::arrive(outputs_arrived(s, output_stage));
                warp::arrive(weights_finished(s, input_stage));

                if (i >= inst.iters - INPUT_PIPELINE_STAGES) {
// Release pages.
#pragma unroll
                    for (int j = 0; j < STAGE_PAGES; j++) {
                        s.warp_finish_page(get_weight_page(s, input_stage, j), 1);
                    }
                }

                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
                output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
            }
        }

        __device__ static inline void storer_loop(state<Config> &s, const Globals &g) {
            parsed_instruction inst{s};

            int output_stage = 0;
            for (int i = 0; i < inst.iters; i++) {
                auto &sem = outputs_arrived(s, output_stage);
                auto bit = (i % (2 * OUTPUT_PIPELINE_STAGES)) >= OUTPUT_PIPELINE_STAGES;

                wait(sem, bit);

                if (i == 0) {
                    s.record(TEVENT_FIRST_STORE);
                }
                else if (i == inst.iters - 1) {
                    s.record(TEVENT_LAST_STORE);
                }

                pipeline_specifics::store(s, g, inst, i, output_stage);

                warp::arrive(outputs_finished(s, output_stage));
                output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
            }
        }
    };
}