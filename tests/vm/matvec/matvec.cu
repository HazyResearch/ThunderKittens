#define RED_TEXT "\033[31m"
#define GREEN_TEXT "\033[32m"
#define YELLOW_TEXT "\033[33m"
#define BLUE_TEXT "\033[34m"
#define MAGENTA_TEXT "\033[35m"
#define CYAN_TEXT "\033[36m"
#define WHITE_TEXT "\033[37m"
#define RESET_TEXT "\033[0m"

#include "kittens.cuh"
// #define KVM_DEBUG
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using weights = gl<bf16, 1, -1, -1, 2048, st_bf<16, 512>>; // assumed to be N by 2048 (X@W.T).
    using activations = gl<bf16, 1, 1, 1, 2048, sv_bf<2048>, sv_bf<16>>;
    using barriers = gl<bf16, 1, -1, 6, 32>; // num_layers by 6 ops per layer by up to 32 heads. 
    instruction_layout instructions;
    timing_layout timings;
    weights W;
    activations A;
    activations O;
    barriers Bar;
    dim3 grid() { return dim3(148); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config, int _OP_IDX=0> struct MatvecOp {
    static constexpr int opcode = 2;
    static constexpr int OP_IDX = _OP_IDX; // Op index within the layer -- controls which barrier to listen to.
    struct parsed_instruction {
        int layer, start_col, expected_arrival_count;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            layer = instruction[1]; // in units of 1
            start_col = instruction[2]; // in units of 1
            expected_arrival_count = instruction[3];
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };
    static __device__ inline parsed_instruction parse_instruction(const globals &g, state<config> &s) {
        return parsed_instruction{s.instruction()[1], s.instruction()[2], s.instruction()[3]};
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
                tma::load_async(weight_chunk, g.W, {inst.layer, inst.start_col/16, laneid()}, inputs_arrived(s, laneid()));
            }
            else if(laneid() == 31) {
                int activation_page = get_activation_page(s);
                s.wait_page_ready(activation_page);
                while(*(volatile int *)&g.Bar[{inst.layer, OP_IDX, 0}] != inst.expected_arrival_count) __nanosleep(20);
                s.record(24);
                auto &activations = reinterpret_cast<sv_bf<2048> &>(s.pages[activation_page]);
                tma::expect(activations_arrived(s), activations);
                tma::load_async(activations, g.A, {}, activations_arrived(s));
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
                tma::store_async(g.O, output, {inst.start_col/16});
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

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(matvec, m) {
    m.doc() = "matvec python module";
    kittens::py::bind_kernel<kvm<config, globals, MatvecOp<config>>>(m, "matvec",
        &globals::instructions,
        &globals::timings,
        &globals::W,
        &globals::A,
        &globals::O,
        &globals::Bar
    );
}