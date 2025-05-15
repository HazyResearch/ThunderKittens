#pragma once

#include "llama.cuh"

namespace kittens::prototype::vm {

static constexpr int PIPELINE_K_DIM = 64;

template <typename Config, typename Globals, typename parsed_instruction, typename gmem_waiter, auto A_Ptr, auto B_Ptr, int Num_Iters>
struct matmul_pipeline {
    static_assert(Config::NUM_CONSUMER_WARPS == 16);
    static_assert(Config::PAGE_SIZE == 32768);
    static_assert(Config::SCRATCH_BYTES >= 8192);

    static constexpr int INPUT_PIPELINE_STAGES = 3;

    using a_st = st_bf<128, PIPELINE_K_DIM>;
    using b_st = st_bf<256, PIPELINE_K_DIM>;

    static constexpr int SEM_COUNT = 2 * INPUT_PIPELINE_STAGES + 1;

    // Pages (very naive for now, no fine-grained usage)
    __device__ static inline int get_a_page(state<Config> &s, int stage) { return s.pid(2*stage); }
    __device__ static inline int get_b_page(state<Config> &s, int stage) { return s.pid(2*stage + 1); }

    __device__ static inline semaphore &inputs_arrived(state<Config> &s, int stage) { return s.semaphores()[stage]; }
    __device__ static inline semaphore &inputs_finished(state<Config> &s, int stage) { return s.semaphores()[INPUT_PIPELINE_STAGES + stage]; }
    __device__ static inline semaphore &outputs_arrived(state<Config> &s) { return s.semaphores()[2 * INPUT_PIPELINE_STAGES]; }

    // Helper to get what page is released at certain stage 
    __device__ static inline int get_used_page_at(int idx) {
        auto iters = Num_Iters;
        auto remainder = iters % INPUT_PIPELINE_STAGES;

        if(remainder == 0) {
            int ret_order[6] = {0,1,2,3,4,5};
            return ret_order[idx];
        }
        else if (remainder == 1) {
            int ret_order[6] = {2,3,4,5,0,1};
            return ret_order[idx];
        }
        else if (remainder == 2) {
            int ret_order[6] = {4,5,0,1,2,3};
            return ret_order[idx];
        }
        else {
            asm volatile("trap;");
            return -1;
        }
    }

    __device__ static inline int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
        static_assert(INPUT_PIPELINE_STAGES == 3, "INPUT_PIPELINE_STAGES must be 3");

        parsed_instruction inst{instruction};
        
        auto remainder = Num_Iters % INPUT_PIPELINE_STAGES;

        if(remainder == 0) {
            int ret_order[6] = {0,1,2,3,4,5};
            return ret_order[query];
        }
        else if (remainder == 1) {
            int ret_order[6] = {2,3,4,5,0,1};
            return ret_order[query];
        }
        else if (remainder == 2) {
            int ret_order[6] = {4,5,0,1,2,3};
            return ret_order[query];
        }
        else {
            asm volatile("trap;");
            return -1;
        }
    }

    __device__ static inline int init_semaphores(state<Config> &s) {
        for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
            init_semaphore(inputs_arrived(s, i), 1);
            init_semaphore(inputs_finished(s, i), 2);
        }
        init_semaphore(outputs_arrived(s), 1);
        return SEM_COUNT;
    }

    template <int _stages_to_not_release = 0>
    __device__ static inline void loader_loop(state<Config> &s, const Globals &g, int layer_idx = 0) {
        parsed_instruction inst{s};
        auto needed_pages = min(Num_Iters, INPUT_PIPELINE_STAGES) * 2;

        if (laneid() == 0) {

            int input_stage = 0;
            for (int iter = 0; iter < Num_Iters; iter++) {
                wait(inputs_finished(s, input_stage), (iter % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);

                auto &sem = inputs_arrived(s, input_stage);
                tma::expect_bytes(sem, sizeof(bf16) * 512 * 64);

                int a_page = get_a_page(s, input_stage), b_page = get_b_page(s, input_stage);
                auto (&a_smem)[2] = reinterpret_cast<a_st(&)[2]>(s.pages[a_page]);
                auto &b_smem = reinterpret_cast<b_st&>(s.pages[b_page]);

                if (iter < INPUT_PIPELINE_STAGES) s.wait_page_ready(a_page); // Stall until A is ready.
                gmem_waiter::gmem_wait(g, s, inst);

                // Load A
                tma::load_async(a_smem[0], g.*A_Ptr, {2*inst.row+0, iter}, sem);
                tma::load_async(a_smem[1], g.*A_Ptr, {2*inst.row+1, iter}, sem);

                if (iter < INPUT_PIPELINE_STAGES) s.wait_page_ready(b_page); // Stall until B is ready.
                // Load B
                tma::load_async(b_smem, g.*B_Ptr, {layer_idx, inst.col, iter}, sem);

                // Advance
                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            }
            for(int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
                wait(inputs_finished(s, input_stage), ((Num_Iters+i) % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);
                if(i == INPUT_PIPELINE_STAGES-1) arrive(outputs_arrived(s)); // signal done with matmuls.
                int a_page = get_a_page(s, input_stage), b_page = get_b_page(s, input_stage);
                if(i < INPUT_PIPELINE_STAGES-Num_Iters) {
                    s.wait_page_ready(a_page);
                    s.wait_page_ready(b_page);
                }
                
                if (i < INPUT_PIPELINE_STAGES - _stages_to_not_release) {
                    s.finish_page(a_page, Config::NUM_CONSUMER_WARPS);
                    s.finish_page(b_page, Config::NUM_CONSUMER_WARPS);
                }

                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES; // Advance input stage.
            }
        }
    }
    __device__ static inline void launcher_loop(state<Config> &s, const Globals &g) {
        parsed_instruction inst{s};

        s.wait_tensor_ready();

        auto dt = s.tensor_alloc.template allocate<tt<float, 128, 256>>(laneid() * 256);

        if(laneid() < 2) {
            int input_stage = 0;
            for (int i = 0; i < Num_Iters; i++) {
                int a_page = get_a_page(s, input_stage), b_page = get_b_page(s, input_stage);
                wait(inputs_arrived(s, input_stage), (i % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);
                a_st (&a_smem)[2] = reinterpret_cast<a_st(&)[2]>(s.pages[a_page]);
                b_st &b_smem = reinterpret_cast<b_st&>(s.pages[b_page]);

                if(i == 0) kittens::mm <transpose::N, transpose::T>(dt, a_smem[laneid()], b_smem, inputs_finished(s, input_stage));
                else       kittens::mma<transpose::N, transpose::T>(dt, a_smem[laneid()], b_smem, inputs_finished(s, input_stage));
                // printf("launched lane %d mma targeting col %d, inputs_finished(%d)\n", laneid(), laneid()*256, input_stage);

                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            }
        }
    }
};
}