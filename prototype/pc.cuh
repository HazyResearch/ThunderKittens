#pragma once

#include "../include/kittens.cuh"

/*
int INPUT_PIPE_STAGES
int OUTPUT_PIPE_STAGES
int NUM_CONSUMER_WARPS

struct globals {}
struct input_block {}
struct output_block {}
struct scratch_block {}
struct finish_block {}

struct producer {
    struct state {};
    __device__ static void setup(state& s, scratch_block &scratch, globals& g) {}
    __device__ static bool load(state& s, input_block& b, scratch_block &scratch, globals& g, barrier& inputs_ready, int idx) {} // more to do?
    __device__ static bool store(state& s, output_block& b, scratch_block &scratch, globals& g, barrier& outputs_finished, int idx) {} // more to do?
}

struct consumer {
    struct state {};
    __device__ static void setup(state& s, scratch_block &scratch, globals& g) {}
    __device__ static bool work(state& s, input_block& b, scratch_block &scratch, output_block& o, barrier& inputs_finished, barrier& outputs_ready, int idx) {} // more to do?
    __device__ static void finish(state& s, finish_block &f, scratch_block &scratch, globals& g, int idx) {}
}
*/

namespace kittens {
namespace prototype {

template<typename T> concept pc_layout = requires {
    typename T::globals;
    typename T::input_block;
    typename T::producer_state;
    typename T::consumer_state;    
};
namespace detail {
template<typename T> concept has_output_block  = pc_layout<T> && requires { typename T::output_block;  };
template<typename T> struct output_block_getter { using type = empty; };
template<has_output_block T> struct output_block_getter<T> { using type = typename T::output_block; };
template<typename T> concept has_scratch_block = pc_layout<T> && requires { typename T::scratch_block; };
template<typename T> struct scratch_block_getter { using type = empty; };
template<has_scratch_block T> struct scratch_block_getter<T> { using type = typename T::scratch_block; };
template<typename T> concept has_finish_block  = pc_layout<T> && requires { typename T::finish_block;  };
template<typename T> struct finish_block_getter { using type = empty; };
template<has_finish_block T> struct finish_block_getter<T> { using type = typename T::finish_block; };
}
template<pc_layout T> struct complete_pc_layout : T {
    using output_block_type  = typename detail::output_block_getter<T>::type;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using finish_block_type  = typename detail::finish_block_getter<T>::type;
};

template<pc_layout T> struct producer_setup_args {
    using globals_type = typename T::globals;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using producer_state_type = typename T::producer_state;
    globals_type& globals;
    producer_state_type& state;
    scratch_block_type& scratch;
    __device__ producer_setup_args(globals_type& _globals, producer_state_type& _state, scratch_block_type& _scratch)
        : globals(_globals), state(_state), scratch(_scratch) {}
};
template<pc_layout T> struct producer_load_args {
    using input_block_type = typename T::input_block;
    using producer_state_type = typename T::producer_state;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using globals_type = typename T::globals;
    input_block_type& input;
    producer_state_type& state;
    scratch_block_type& scratch;
    globals_type& globals;
    barrier& inputs_arrived;
    int iter;
    __device__ producer_load_args(input_block_type& _input, producer_state_type& _state, scratch_block_type& _scratch, globals_type& _globals, barrier& _inputs_arrived, int _iter)
        : input(_input), state(_state), scratch(_scratch), globals(_globals), inputs_arrived(_inputs_arrived), iter(_iter) {}
};
template<pc_layout T> struct producer_store_args {
    using output_block_type = typename detail::output_block_getter<T>::type;
    using producer_state_type = typename T::producer_state;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using globals_type = typename T::globals;
    output_block_type& output;
    producer_state_type& state;
    scratch_block_type& scratch;
    globals_type& globals;
    barrier& outputs_finished;
    int iter;
    __device__ producer_store_args(output_block_type& _output, producer_state_type& _state, scratch_block_type& _scratch, globals_type& _globals, barrier& _outputs_finished, int _iter)
        : output(_output), state(_state), scratch(_scratch), globals(_globals), outputs_finished(_outputs_finished), iter(_iter) {}
};
template<pc_layout T> struct consumer_setup_args {
    using globals_type = typename T::globals;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using consumer_state_type = typename T::consumer_state;
    globals_type& globals;
    consumer_state_type& state;
    scratch_block_type& scratch;
    __device__ consumer_setup_args(globals_type& _globals, consumer_state_type& _state, scratch_block_type& _scratch)
        : globals(_globals), state(_state), scratch(_scratch) {}
};
template<pc_layout T> struct consumer_work_args {
    using input_block_type = typename T::input_block;
    using output_block_type = typename detail::output_block_getter<T>::type;
    using consumer_state_type = typename T::consumer_state;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using globals_type = typename T::globals;
    input_block_type& input;
    output_block_type& output;
    consumer_state_type& state;
    scratch_block_type& scratch;
    globals_type& globals;
    barrier& inputs_finished;
    barrier& outputs_arrived;
    int iter;
    __device__ consumer_work_args(input_block_type& _input, output_block_type& _output, consumer_state_type& _state, scratch_block_type& _scratch, globals_type& _globals, barrier& _inputs_finished, barrier& _outputs_arrived, int _iter)
        : input(_input), output(_output), state(_state), scratch(_scratch), globals(_globals), inputs_finished(_inputs_finished), outputs_arrived(_outputs_arrived), iter(_iter) {}
};
template<pc_layout T> struct consumer_finish_args {
    using finish_block_type = typename detail::finish_block_getter<T>::type;
    using consumer_state_type = typename T::consumer_state;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using globals_type = typename T::globals;
    finish_block_type& finish;
    consumer_state_type& state;
    scratch_block_type& scratch;
    globals_type& globals;
    int iter;
    __device__ consumer_finish_args(finish_block_type& _finish, consumer_state_type& _state, scratch_block_type& _scratch, globals_type& _globals, int _iter)
        : finish(_finish), state(_state), scratch(_scratch), globals(_globals), iter(_iter) {}
};
template<typename pct> concept pc_template = requires {
    pct::INPUT_PIPE_STAGES;
    pct::OUTPUT_PIPE_STAGES;
    pct::NUM_CONSUMER_WARPS;
    typename pct::layout;
    typename pct::producer;
    typename pct::consumer;
    pct::producer::setup;
    pct::producer::load;
    pct::consumer::setup; 
    pct::consumer::work;
} && pc_layout<typename pct::layout>;
template<typename pct> concept has_store  = requires { pct::producer::store; };
template<typename pct> concept has_finish = requires { pct::consumer::finish; };

template<typename T> constexpr int num_threads = T::NUM_CONSUMER_WARPS * 32 + 128;
template<typename T> constexpr int num_warps = T::NUM_CONSUMER_WARPS + 4;
template<typename T> constexpr int num_consumer_warpgroups = T::NUM_CONSUMER_WARPS / 4;

template<int N> __device__ static inline int ring_advance(int ring, int distance=1) { return (ring + distance) % N; }
template<int N> __device__ static inline int ring_retreat(int ring) { return (ring + N-1) % N; }

template<typename pct>
__global__ __launch_bounds__(num_threads<pct>, 1)
void pc(typename pct::layout::globals g) {
    static_assert(pc_template<pct>, "pc template parameter does not satisfy concept requirements");
    using layout = complete_pc_layout<typename pct::layout>; // complete the layout by filling in the optional types with empty
    constexpr int INPUT_PIPE_STAGES = pct::INPUT_PIPE_STAGES;
    static_assert(INPUT_PIPE_STAGES >= 1 && INPUT_PIPE_STAGES <= 32, "Invalid number of input pipe stages");
    constexpr int OUTPUT_PIPE_STAGES = pct::OUTPUT_PIPE_STAGES == 0 ? 1 : pct::OUTPUT_PIPE_STAGES; // Even if the producer doesn't have a store, we need to allocate the empty struct
    static_assert(OUTPUT_PIPE_STAGES >= 1 && OUTPUT_PIPE_STAGES <= 32, "Invalid number of output pipe stages");
    constexpr int NUM_CONSUMER_WARPS = pct::NUM_CONSUMER_WARPS;
    constexpr int NUM_WARPS = num_warps<pct>;
    constexpr int NUM_CONSUMER_WARPGROUPS = num_consumer_warpgroups<pct>;
    constexpr int NUM_THREADS = num_threads<pct>;
    using globals        = typename layout::globals;
    using input_block    = typename layout::input_block;
    using producer_state = typename layout::producer_state;
    using consumer_state = typename layout::consumer_state;
    using output_block   = typename layout::output_block_type;
    using scratch_block  = typename layout::scratch_block_type;
    using finish_block   = typename layout::finish_block_type;

    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    input_block   (&input_smem)  [INPUT_PIPE_STAGES]  = alloc.allocate<input_block,  INPUT_PIPE_STAGES >();
    output_block  (&output_smem) [OUTPUT_PIPE_STAGES] = alloc.allocate<output_block, OUTPUT_PIPE_STAGES>();
    scratch_block (&scratch_smem)                     = alloc.allocate<scratch_block>();
    finish_block  (&finish_smem)                      = reinterpret_cast<finish_block&>(input_smem);

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
        producer_state s;
        pct::producer::setup({g, s, scratch_smem});
        int load_more = true, load_iter = 0, store_iter = 0;
        #pragma unroll
        for(int i = 0; i < INPUT_PIPE_STAGES && load_more; i++) { // fill the pipeline
            load_more = pct::producer::load({
                input_smem[input_ring],
                s,
                scratch_smem,
                g,
                inputs_arrived[input_ring],
                load_iter
            });
            input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            load_iter++;
        }
        if constexpr (has_store<pct>) {
            while(load_more || store_iter < load_iter) {
                if(store_iter < load_iter && test_wait(outputs_arrived[output_ring], (store_iter/OUTPUT_PIPE_STAGES)%2)) {
                    pct::producer::store({
                        output_smem[output_ring],
                        s,
                        scratch_smem,
                        g,
                        outputs_finished[output_ring],
                        store_iter
                    });
                    output_ring=ring_advance<OUTPUT_PIPE_STAGES>(output_ring);
                    store_iter++;
                }
                // need to wait for the next stage to be available to write to.
                if(load_more && test_wait(inputs_finished[input_ring], ((load_iter/INPUT_PIPE_STAGES)%2)^1)) {
                    load_more = pct::producer::load({
                        input_smem[input_ring],
                        s,
                        scratch_smem,
                        g,
                        inputs_arrived[input_ring],
                        load_iter
                    });
                    input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
                    load_iter++;
                }
                __nanosleep(5);
            }
        }
        else { // just do the load
            while(load_more) {
                wait(inputs_finished[input_ring], ((load_iter/INPUT_PIPE_STAGES)%2)^1);
                load_more = pct::producer::load({
                    input_smem[input_ring],
                    s,
                    scratch_smem,
                    g,
                    inputs_arrived[input_ring],
                    load_iter
                });
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
                load_iter++;
            }
        }
    }
    else { // other warpgroups are consumers
        consumer_state s;
        pct::consumer::setup({g, s, scratch_smem});
        int work_more = true, iter = 0;
        while(work_more) {
            if constexpr (has_store<pct>) {
                wait(outputs_finished[output_ring], ((iter/OUTPUT_PIPE_STAGES)%2)^1); // wait for memory to arrive, phase changes at half the rate of the ring
            }
            wait(inputs_arrived[input_ring], (iter/INPUT_PIPE_STAGES)%2); // wait for memory to arrive, phase changes at half the rate of the ring
            work_more = pct::consumer::work({
                input_smem[input_ring],
                output_smem[output_ring],
                s,
                scratch_smem,
                g,
                inputs_finished[input_ring],
                outputs_arrived[output_ring],
                iter
            });
            iter++;
            input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            if constexpr (has_store<pct>) {
                output_ring=ring_advance<OUTPUT_PIPE_STAGES>(output_ring);
            }
        }
        if constexpr (has_finish<pct>) {
            group<NUM_CONSUMER_WARPS>::sync(0); // cannot overwrite finish block until all consumer warps are done.
            pct::consumer::finish({
                finish_smem,
                s,
                scratch_smem,
                g,
                iter
            });
        }
    }
}

}
}