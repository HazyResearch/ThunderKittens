#pragma once

#include "../include/kittens.cuh"

/*
int INPUT_PIPE_STAGES
int OUTPUT_PIPE_STAGES
int NUM_CONSUMER_WARPS

struct globals {}
struct input_block {}
struct output_block {}

struct producer {
    struct state {};
    __device__ static void setup(state& s, globals& g) {}
    __device__ static bool load(state& s, input_block& b, globals& g, barrier& inputs_ready, int idx) {} // more to do?
    __device__ static bool store(state& s, output_block& b, globals& g, barrier& outputs_finished, int idx) {} // more to do?
}

struct consumer {
    struct state {};
    __device__ static void setup(state& s, globals& g) {}
    __device__ static bool work(state& s, input_block& b, output_block& o, barrier& inputs_finished, barrier& outputs_ready, int idx) {} // more to do?
    __device__ static void finish(state& s, globals& g, int idx) {}
}
*/

namespace kittens {
namespace prototype {

template<typename pct> concept pc_template = requires {
    pct::INPUT_PIPE_STAGES;
    pct::OUTPUT_PIPE_STAGES;
    pct::NUM_CONSUMER_WARPS;
    typename pct::globals;
    typename pct::input_block;
    typename pct::output_block;
    typename pct::scratch_block;
    typename pct::finish_block;
    typename pct::producer;
    typename pct::consumer;
    typename pct::producer::state;
    typename pct::consumer::state;
    pct::producer::setup;
    pct::producer::load;
    pct::consumer::setup; 
    pct::consumer::work;
    pct::consumer::finish;
} && (pct::OUTPUT_PIPE_STAGES == 0 || requires { pct::producer::store; });

template<typename T> constexpr int num_threads = T::NUM_CONSUMER_WARPS * 32 + 128;
template<typename T> constexpr int num_warps = T::NUM_CONSUMER_WARPS + 4;
template<typename T> constexpr int num_consumer_warpgroups = T::NUM_CONSUMER_WARPS / 4;

template<int N> __device__ static inline int ring_advance(int ring, int distance=1) { return (ring + distance) % N; }
template<int N> __device__ static inline int ring_retreat(int ring) { return (ring + N-1) % N; }

template<typename pct>
__global__ __launch_bounds__(num_threads<pct>, 1)
void pc(typename pct::globals g) {
    static_assert(pc_template<pct>, "pc template parameter does not satisfy concept requirements");
    constexpr int INPUT_PIPE_STAGES = pct::INPUT_PIPE_STAGES;
    static_assert(INPUT_PIPE_STAGES >= 1 && INPUT_PIPE_STAGES <= 32, "Invalid number of input pipe stages");
    constexpr int OUTPUT_PIPE_STAGES = pct::OUTPUT_PIPE_STAGES == 0 ? 1 : pct::OUTPUT_PIPE_STAGES;
    static_assert(OUTPUT_PIPE_STAGES >= 1 && OUTPUT_PIPE_STAGES <= 32, "Invalid number of output pipe stages");
    constexpr int NUM_CONSUMER_WARPS = pct::NUM_CONSUMER_WARPS;
    constexpr int NUM_WARPS = num_warps<pct>;
    constexpr int NUM_CONSUMER_WARPGROUPS = num_consumer_warpgroups<pct>;
    constexpr int NUM_THREADS = num_threads<pct>;
    using globals = typename pct::globals;
    using input_block = typename pct::input_block;
    using output_block = typename pct::output_block;
    using scratch_block = typename pct::scratch_block;
    using finish_block = typename pct::finish_block;

    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    input_block   (&input_smem) [INPUT_PIPE_STAGES]  = alloc.allocate<input_block,  INPUT_PIPE_STAGES >();
    output_block  (&output_smem)[OUTPUT_PIPE_STAGES] = alloc.allocate<output_block, OUTPUT_PIPE_STAGES>();
    scratch_block (&scratch_smem)                    = alloc.allocate<scratch_block>();
    finish_block  (&finish_smem)                     = reinterpret_cast<finish_block&>(input_smem);

    // Initialize barriers. This is constant for all two-stage producer-consumer kernels.
    __shared__ kittens::barrier inputs_arrived[INPUT_PIPE_STAGES],   inputs_finished[INPUT_PIPE_STAGES];
    __shared__ kittens::barrier outputs_arrived[OUTPUT_PIPE_STAGES], outputs_finished[OUTPUT_PIPE_STAGES];
    if (warpid() == 0) { // a single warp (in fact a single thread) does these.
        for(int i = 0; i < INPUT_PIPE_STAGES; i++) {
            init_barrier(inputs_arrived[i], 0, 4); // needs to wait on each producer warp
            init_barrier(inputs_finished[i], NUM_CONSUMER_WARPS, 0); // needs to wait on one thread from each consumer warp
        }
        for(int i = 0; i < OUTPUT_PIPE_STAGES; i++) {
            init_barrier(outputs_arrived[i], NUM_CONSUMER_WARPS, 0); // needs to wait on one thread from each consumer warp
            init_barrier(outputs_finished[i], 0, 4); // needs to wait on each producer warp
        }
    }

    int input_ring  = 0; // tracking which input block is being loaded
    int output_ring = 0; // tracking which output block is being written

    __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

    if(warpgroup::groupid() == NUM_CONSUMER_WARPGROUPS) { // last warpgroup is a producer
        typename pct::producer::state s;
        pct::producer::setup(s, g);
        int load_more = true, load_iter = 0, store_iter = 0;
        #pragma unroll
        for(int i = 0; i < INPUT_PIPE_STAGES && load_more; i++) { // fill the pipeline
            load_more = pct::producer::load(
                s,
                input_smem[input_ring],
                g,
                inputs_arrived[input_ring],
                load_iter
            );
            input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            load_iter++;
        }
        while(load_more || (pct::OUTPUT_PIPE_STAGES != 0 && store_iter < load_iter)) {
            if constexpr (pct::OUTPUT_PIPE_STAGES != 0) {
                if(store_iter < load_iter && test_wait(outputs_arrived[output_ring], (store_iter/OUTPUT_PIPE_STAGES)%2)) {
                    pct::producer::store(
                        s,
                        output_smem[output_ring],
                        g,
                        outputs_finished[output_ring],
                        store_iter
                    );
                    output_ring=ring_advance<OUTPUT_PIPE_STAGES>(output_ring);
                    store_iter++;
                }
                // need to wait for the next stage to be available to write to.
                if(load_more && test_wait(inputs_finished[input_ring], ((load_iter/INPUT_PIPE_STAGES)%2)^1)) {
                    load_more = pct::producer::load(
                        s,
                        input_smem[input_ring],
                        g,
                        inputs_arrived[input_ring],
                        load_iter
                    );
                    input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
                    load_iter++;
                }
                __nanosleep(5);
            }
            else { // just do the load
                wait(inputs_finished[input_ring], ((load_iter/INPUT_PIPE_STAGES)%2)^1);
                load_more = pct::producer::load(
                    s,
                    input_smem[input_ring],
                    g,
                    inputs_arrived[input_ring],
                    load_iter
                );
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
                load_iter++;
            }
        }
    }
    else { // other warpgroups are consumers
        typename pct::consumer::state s;
        pct::consumer::setup(s, scratch_smem, g);
        int work_more = true, iter = 0;
        while(work_more) {
            if constexpr (pct::OUTPUT_PIPE_STAGES != 0) {
                wait(outputs_finished[output_ring], ((iter/OUTPUT_PIPE_STAGES)%2)^1); // wait for memory to arrive, phase changes at half the rate of the ring
            }
            wait(inputs_arrived[input_ring], (iter/INPUT_PIPE_STAGES)%2); // wait for memory to arrive, phase changes at half the rate of the ring
            work_more = pct::consumer::work(s, input_smem[input_ring], scratch_smem, output_smem[output_ring], inputs_finished[input_ring], outputs_arrived[output_ring], iter);
            iter++;
            input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            if constexpr (pct::OUTPUT_PIPE_STAGES != 0) {
                output_ring=ring_advance<OUTPUT_PIPE_STAGES>(output_ring);
            }
        }
        group<NUM_CONSUMER_WARPS>::sync(0);
        pct::consumer::finish(s, finish_smem, scratch_smem, g, iter);
    }
}

}
}