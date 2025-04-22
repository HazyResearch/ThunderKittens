#pragma once

#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using llama_vm_config = default_config;
struct llama_globals {
    static constexpr int MAX_SEQ_LEN = 4096;
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<llama_vm_config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<llama_vm_config>;
    // All weights are fused here.
    using up_weights_t         = gl<bf16,  1, 16, 8192, 2048>;
    using down_weights_t       = gl<bf16,  1, 16, 2048, 8192>;
    using qkv_weights_t        = gl<bf16,  1, 16, 2048+512+512, 2048>; // Q + K + V output dimensions are fused here.
    using o_weights_t          = gl<bf16,  1, 16, 2048, 2048>;
    using layernorm_weights_t  = gl<bf16,  1, 16, 2, 2048>; // both layernorms are fused here.
    // Two types of activations: for the residual stream, and for the hidden activations.
    using activations_t        = gl<bf16,  1, 1, 1, 2048>;
    using hidden_activations_t = gl<bf16,  1, 1, 1, 8192>;
    // I am making a deliberate choice to fuse nheads & headdim to make it easier to batch later.
    using k_cache_t            = gl<bf16,  1, 16, MAX_SEQ_LEN, 512>; 
    using v_cache_t            = gl<bf16,  1, 16, MAX_SEQ_LEN, 512>;
    // Never expecting to have more partials for a given sequence than SM's.
    using o_scratch_t          = gl<float, 1, 16, 160, 2048>;
    using lvec_scratch_t       = gl<float, 1, 1, 16, 160>;
    // Barriers are per-layer, per-op, per-head.
    using barriers_t           = gl<bf16,  1, 16, 6, 32>;
    // instructions
    instruction_layout instructions;
    timing_layout timings;
    // model weights
    up_weights_t up_weights;
    down_weights_t down_weights;
    qkv_weights_t qkv_weights;
    o_weights_t o_weights;
    layernorm_weights_t layernorm_weights;
    // activations
    activations_t activations;
    activations_t attention_output;
    hidden_activations_t hidden_activations;
    // caches
    k_cache_t k_cache;
    v_cache_t v_cache;
    // scratch for ThunderGQA
    o_scratch_t o_scratch;
    lvec_scratch_t lvec_scratch;
    // barriers
    barriers_t barriers;
    // grid & block sizes
    dim3 grid() { return dim3(148); }
    dim3 block() { return dim3(llama_vm_config::NUM_THREADS); }
    int dynamic_shared_memory() { return llama_vm_config::DYNAMIC_SHARED_MEMORY; }
};

template<
    int _EXPECTED_ARRIVAL_COUNT,
    typename Weights,
    typename InputActivations,
    typename OutputActivations,
    int _opcode,
    int _OP_IDX=0,
    typename config=llama_vm_config
>
struct AddMatvecOp {
    static constexpr int opcode = _opcode;
    static constexpr int EXPECTED_ARRIVAL_COUNT = _EXPECTED_ARRIVAL_COUNT;
    static constexpr int OP_IDX = _OP_IDX; // Op index within the layer -- controls which barrier to listen to.
    struct parsed_instruction {
        int layer, start_col;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            layer = instruction[1]; // in units of 1
            start_col = instruction[2]; // in units of 1
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };
    static __device__ inline parsed_instruction parse_instruction(const globals &g, state<config> &s) {
        return parsed_instruction{s.instruction()[1], s.instruction()[2]};
    }
    __device__ static inline semaphore &inputs_arrived(state<config> &s, int id) {
        return s.semaphores()[id];
    }
    __device__ static inline semaphore &outputs_arrived(state<config> &s) {
        return s.semaphores()[4];
    }
    __device__ static inline semaphore &activations_arrived(state<config> &s) {
        return s.semaphores()[5];
    }
    __device__ static inline int get_weight_page(state<config> &s, int offset) {
        return s.pid(offset + 1);
    }
    __device__ static inline int get_activation_page(state<config> &s) {
        return s.pid(0);
    }
    
    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            int ret_order[] = {5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for(int i = 0; i < 4; i++) {
                init_semaphore(inputs_arrived(s, i), 1); // Inputs arrived.
            }
            init_semaphore(outputs_arrived(s), 16); // outputs arrived.
            init_semaphore(activations_arrived(s), 1);
            s.record(1);
            return 6;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
            ((int*)s.scratch())[laneid()] = 0;
            warp::sync(); // done, now we can proceed to other things.
            if(laneid() < 4) {
                s.wait_page_ready(get_weight_page(s, laneid()));
                s.record(16+laneid());
                auto &weight_chunk = reinterpret_cast<st_bf<16, 512> &>(s.pages[get_weight_page(s, laneid())]);
                tma::expect(inputs_arrived(s, laneid()), weight_chunk);
                tma::load_async(weight_chunk, Weights, {inst.layer, inst.start_col/16, laneid()}, inputs_arrived(s, laneid()));
            }
            else if(laneid() == 31) {
                int activation_page = get_activation_page(s);
                s.wait_page_ready(activation_page);
                while(*(volatile int *)&g.Bar[{inst.layer, OP_IDX, 0}] != EXPECTED_ARRIVAL_COUNT) __nanosleep(20);
                s.record(24);
                auto &activations = reinterpret_cast<sv_bf<2048> &>(s.pages[activation_page]);
                tma::expect(activations_arrived(s), activations);
                tma::load_async(activations, InputActivations, {}, activations_arrived(s));
            }
            else if(laneid() >= 5 && laneid() <= 12) {
                int unused_page = s.pid(laneid());
                s.wait_page_ready(unused_page);
                arrive(s.page_finished[unused_page], config::NUM_CONSUMER_WARPS); // Release the unused pages immediately.
            }
        }
    };
    struct launcher { // launches mma's
        // launcher does nothing here, since this doesn't use tensor cores.
        static __device__ void run(const globals &g, state<config> &s) {
            s.wait_tensor_ready();
            if(laneid() == 0) arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            rt_bf<16, 128> weights, broadcast_activations;
            typename rt_bf<16, 128>::row_vec activations_vec;
            typename rt_bf<16, 128>::col_vec output_col_format;
            rv_bf<16> output;
            int group_id = warpgroup::groupid();
            int warp_id = warpgroup::warpid(); // id within the warpgroup
            wait(inputs_arrived(s, group_id), 0);
            if(laneid() == 0) s.record(32+warpid());
            // Reinterpret the page as a st_bf<16, 128>[4], which turns out to be a valid recast of the layout.
            int weight_page = get_weight_page(s, group_id);
            st_bf<16, 128> (&weights_smem)[4] = reinterpret_cast<st_bf<16, 128>(&)[4]>(s.pages[weight_page]);
            warp::load(weights, weights_smem[warp_id]);
            warp::sync();
            warp::arrive(s.page_finished[weight_page], config::NUM_CONSUMER_WARPS/4); // this is called by each warp in the warpgroup
            // Next we need to load the activations
            wait(activations_arrived(s), 0);
            if(laneid() == 0) s.record(64+warpid());
            // reinterpret the activations page as sv_bf<128>[16]
            int activation_page = get_activation_page(s);
            sv_bf<128> (&activations_smem)[16] = reinterpret_cast<sv_bf<128>(&)[16]>(s.pages[activation_page]);
            warp::load(activations_vec, activations_smem[warpid()]);
            warp::sync();
            warp::arrive(s.page_finished[activation_page]); // just 1 is sufficient
            // broadcast this into a tile
            warp::broadcast_col(broadcast_activations, activations_vec);
            warp::mul(broadcast_activations, broadcast_activations, weights);
            warp::row_sum(output_col_format, broadcast_activations);
            warp::copy(output, output_col_format);
            // Now the first 16 threads have the output.
            if(laneid() < 16) { // this might be a bad idea but yolo, it's probably an okay start
                // and fortunately this is code where ncu will tell us if it's bad..
                atomicAdd(&((bf16*)s.scratch())[laneid()], output[0][0]);
            }
            warp::sync();
            warp::arrive(outputs_arrived(s));
            if(group<16>::laneid() == 0) s.record(124);
        }
    };
    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            if(laneid() == 0) {
                wait(outputs_arrived(s), 0);
                s.record(125);
                void *scratch = s.scratch();
                sv_bf<16> &output = *reinterpret_cast<sv_bf<16>*>(scratch);
                tma::store_add_async(OutputActivations, output, {inst.start_col/16});
                tma::store_async_wait(); // not just read wait! full wait! must be visible in global!
                s.record(126);
            }
            warp::sync();
            asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.
            if(laneid() == 0) {
                if constexpr (OP_IDX == g.Bar.rows()-1) atomicAdd(&g.Bar[{inst.layer+1, 0, 0}], 1);
                else atomicAdd(&g.Bar[{inst.layer, OP_IDX+1, 0}], 1);
            }
            if(laneid() == 0) s.record(127);
        }
    };
};
using DownOp = AddMatvecOp<128, &llama_globals::down_weights, &llama_globals::hidden_activations, &llama_globals::activations, 2, 1>;
using OOp = AddMatvecOp<128, &llama_globals::o_weights, &llama_globals::attention_output, &llama_globals::activations, 6, 5>;