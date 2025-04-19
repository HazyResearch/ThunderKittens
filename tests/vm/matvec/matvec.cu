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

/*
Instruction format:
[0] = opcode
[1] = Row offset of C, in units of 128
[2] = Col offset of C, in units of 128
[3] = K reduction dimension, in units of 128
*/

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using weights = gl<bf16, 1, -1, -1, 2048, st_bf<16, 512>>; // assumed to be N by 2048 (X@W.T).
    using activations = gl<bf16, 1, 1, 1, 2048, sv_bf<512>, sv_bf<16>>;
    using barriers = gl<bf16, 1, -1, 6, 32, sv_bf<512>, sv_bf<16>>; // num_layers by 6 ops per layer by up to 32 heads. 
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

template<typename config=config, int _OP_IDX> struct MatmulOp {
    static constexpr int opcode = 2;
    constexpr int OP_IDX = _OP_IDX; // Op index within the layer -- controls which barrier to listen to.
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
            if(laneid() >= 5 && laneid() <= 12) {
                arrive(s.page_finished[s.pid(laneid())], config::NUM_CONSUMER_WARPS); // Release the unused pages immediately.
            }
            parsed_instruction inst{s};
            // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
            ((bf16*)s.scratch())[laneid()] = 0;
            warp::sync(); // done, now we can proceed to other things.
            if(laneid() < 4) {
                s.wait_page_ready(get_weight_page(s, laneid()));
                auto &weight_chunk = reinterpret_cast<st_bf<16, 512> &>(s.pages[get_weight_page(s, laneid())]);
                tma::expect(inputs_arrived(s, laneid()), weight_chunk);
                tma::load_async(weight_chunk, g.W, {inst.layer, inst.start_col/16, laneid()}, inputs_arrived(s, laneid()));
            }
            else if(laneid() == 5) {
                s.wait_page_ready(get_barriers_page(s, laneid()));
                while(*(volatile int *)&g.Bar[{inst.layer, OP_IDX, 0}] == 0) __nanosleep(20);
                auto &activations = reinterpret_cast<sv_bf<2048> &>(s.pages[get_activation_page(s)]);
                tma::expect(activations_arrived(s), activations);
                tma::load_async(activations, g.A, {}, activations_arrived(s));
            }
            warp::sync();
        }
    };
    struct launcher { // launches mma's
        // launcher does nothing here, since this doesn't use tensor cores.
        static __device__ void run(const globals &g, state<config> &s) {}
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
            if(group<16>::laneid() == 0) s.record(125);
            // Reinterpret the page as a st_bf<16, 128>[4], which turns out to be a valid recast of the layout.
            int weight_page = get_weight_page(s, group_id);
            st_bf<16, 128> (&weights_smem)[4] = reinterpret_cast<(st_bf<16, 128>&)[4]>(s.pages[weight_page]);
            warp::load(weights, weights_smem[warp_id]);
            warp::sync();
            warp::arrive(s.page_finished[weight_page], config::NUM_CONSUMER_WARPS/4); // this is called by each warp in the warpgroup
            // Next we need to load the activations
            wait(activations_arrived(s), 0);
            // reinterpret the activations page as sv_bf<128>[16]
            int activation_page = get_activation_page(s);
            sv_bf<128> (&activations_smem)[16] = reinterpret_cast<(sv_bf<128>&)[16]>(s.pages[activation_page]);
            warp::load(activations_vec, activations_smem[warpid()]);
            warp::sync();
            warp::arrive(s.page_finished[activation_page]); // just 1 is sufficient
            // broadcast this into a tile
            broadcast_col(broadcast_activations, activations_vec);
            mul(broadcast_activations, weights);
            row_sum(output_col_format, broadcast_activations);
            copy(output, output_col_format);
            // Now the first 16 threads have the output.
            if(laneid() < 16) { // this might be a bad idea but yolo, it's probably an okay start
                // and fortunately this is code where ncu will tell us if it's bad..
                atomicAdd(((bf16*)s.scratch())[laneid()], output[0]);
            }
            warp::sync();
            warp::arrive(outputs_arrived(s));
            if(laneid() == 0) s.record(127);
        }
    };
    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            if(laneid() == 0) {
                wait(outputs_arrived(s), 0);
                s.record(125);
                st_bf<16> &output = reinterpret_cast<st_bf<16>&>(s.scratch());
                tma::store_async(g.O, output, {g.start_col/16});
                tma::store_async_wait(); // not just read wait! full wait! must be visible in global!
                s.record(126);
            }
            warp::sync();
            asm volatile("fence.gpu"); // possible we need sc here but I don't think so.
            if(laneid() == 0) {
                if constexpr (OP_IDX == Bar.rows()-1) g.Bar[{inst.layer+1, 0, 0}] = 1;
                else g.Bar[{inst.layer, OP_IDX+1, 0}] = 1;
            }
            if(laneid() == 0) s.record(127);
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(matmul, m) {
    m.doc() = "matmul python module";
    kittens::py::bind_kernel<kernel<config, globals, MatmulOp<config>>>(m, "matmul",
        &globals::instructions,
        &globals::timings,
        &globals::A,
        &globals::B,
        &globals::C
    );
}