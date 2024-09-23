#pragma once

#include "../include/kittens.cuh"

namespace kittens {
namespace prototype {

template<typename T> concept pc_layout = requires {
    typename T::globals;
    typename T::input_block;
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
template<typename T> concept has_producer_state = pc_layout<T> && requires { typename T::producer_state; };
template<typename T> struct producer_state_getter { using type = empty; };
template<has_producer_state T> struct producer_state_getter<T> { using type = typename T::producer_state; };
template<typename T> concept has_consumer_state = pc_layout<T> && requires { typename T::consumer_state; };
template<typename T> struct consumer_state_getter { using type = empty; };
template<has_consumer_state T> struct consumer_state_getter<T> { using type = typename T::consumer_state; };
}
template<pc_layout T> struct complete_pc_layout : T {
    using output_block_type  = typename detail::output_block_getter<T>::type;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using finish_block_type  = typename detail::finish_block_getter<T>::type;
    using producer_state_type = typename detail::producer_state_getter<T>::type;
    using consumer_state_type = typename detail::consumer_state_getter<T>::type;
};

template<pc_layout T> struct producer_setup_args {
    using globals_type = typename T::globals;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using producer_state_type = typename detail::producer_state_getter<T>::type;
    globals_type & globals;
    producer_state_type & state;
    scratch_block_type & scratch;
    barrier * all_inputs_arrived;
    barrier * all_inputs_finished;
    barrier * all_outputs_arrived;
    barrier * all_outputs_finished;
    __device__ producer_setup_args(globals_type& _globals, producer_state_type& _state, scratch_block_type& _scratch, barrier * _all_inputs_arrived, barrier * _all_inputs_finished, barrier * _all_outputs_arrived, barrier * _all_outputs_finished)
        : globals(_globals), state(_state), scratch(_scratch), all_inputs_arrived(_all_inputs_arrived), all_inputs_finished(_all_inputs_finished), all_outputs_arrived(_all_outputs_arrived), all_outputs_finished(_all_outputs_finished) {}
};
template<pc_layout T> struct producer_load_args {
    using input_block_type = typename T::input_block;
    using producer_state_type = typename detail::producer_state_getter<T>::type;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using globals_type = typename T::globals;
    input_block_type & input;
    producer_state_type & state;
    scratch_block_type & scratch;
    globals_type & globals;
    barrier & inputs_arrived;
    barrier * all_inputs_arrived;
    barrier * all_inputs_finished;
    barrier * all_outputs_arrived;
    barrier * all_outputs_finished;
    int & iter;
    __device__ producer_load_args(input_block_type& _input, producer_state_type& _state, scratch_block_type& _scratch, globals_type& _globals, barrier& _inputs_arrived, barrier * _all_inputs_arrived, barrier * _all_inputs_finished, barrier * _all_outputs_arrived, barrier * _all_outputs_finished, int & _iter)
        : input(_input), state(_state), scratch(_scratch), globals(_globals), inputs_arrived(_inputs_arrived), all_inputs_arrived(_all_inputs_arrived), all_inputs_finished(_all_inputs_finished), all_outputs_arrived(_all_outputs_arrived), all_outputs_finished(_all_outputs_finished), iter(_iter) {}
};
template<pc_layout T> struct producer_store_args {
    using output_block_type = typename detail::output_block_getter<T>::type;
    using producer_state_type = typename detail::producer_state_getter<T>::type;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using globals_type = typename T::globals;
    output_block_type & output;
    producer_state_type & state;
    scratch_block_type & scratch;
    globals_type & globals;
    barrier & outputs_finished;
    barrier * all_inputs_arrived;
    barrier * all_inputs_finished;
    barrier * all_outputs_arrived;
    barrier * all_outputs_finished;
    int & iter;
    __device__ producer_store_args(output_block_type& _output, producer_state_type& _state, scratch_block_type& _scratch, globals_type& _globals, barrier& _outputs_finished, barrier * _all_inputs_arrived, barrier * _all_inputs_finished, barrier * _all_outputs_arrived, barrier * _all_outputs_finished, int & _iter)
        : output(_output), state(_state), scratch(_scratch), globals(_globals), outputs_finished(_outputs_finished), all_inputs_arrived(_all_inputs_arrived), all_inputs_finished(_all_inputs_finished), all_outputs_arrived(_all_outputs_arrived), all_outputs_finished(_all_outputs_finished), iter(_iter) {}
};
template<pc_layout T> struct consumer_setup_args {
    using globals_type = typename T::globals;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using consumer_state_type = typename detail::consumer_state_getter<T>::type;
    globals_type & globals;
    consumer_state_type & state;
    scratch_block_type & scratch;
    barrier * all_inputs_arrived;
    barrier * all_inputs_finished;
    barrier * all_outputs_arrived;
    barrier * all_outputs_finished;
    __device__ consumer_setup_args(globals_type& _globals, consumer_state_type& _state, scratch_block_type& _scratch, barrier * _all_inputs_arrived, barrier * _all_inputs_finished, barrier * _all_outputs_arrived, barrier * _all_outputs_finished)
        : globals(_globals), state(_state), scratch(_scratch), all_inputs_arrived(_all_inputs_arrived), all_inputs_finished(_all_inputs_finished), all_outputs_arrived(_all_outputs_arrived), all_outputs_finished(_all_outputs_finished) {}
};
template<pc_layout T> struct consumer_work_args {
    using input_block_type = typename T::input_block;
    using output_block_type = typename detail::output_block_getter<T>::type;
    using consumer_state_type = typename detail::consumer_state_getter<T>::type;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using globals_type = typename T::globals;
    input_block_type & input;
    output_block_type & output;
    consumer_state_type & state;
    scratch_block_type & scratch;
    globals_type & globals;
    barrier & inputs_finished;
    barrier & outputs_arrived;
    barrier * all_inputs_arrived;
    barrier * all_inputs_finished;
    barrier * all_outputs_arrived;
    barrier * all_outputs_finished;
    int & iter;
    __device__ consumer_work_args(input_block_type& _input, output_block_type& _output, consumer_state_type& _state, scratch_block_type& _scratch, globals_type& _globals, barrier& _inputs_finished, barrier& _outputs_arrived, barrier * _all_inputs_arrived, barrier * _all_inputs_finished, barrier * _all_outputs_arrived, barrier * _all_outputs_finished, int & _iter)
        : input(_input), output(_output), state(_state), scratch(_scratch), globals(_globals), inputs_finished(_inputs_finished), outputs_arrived(_outputs_arrived), all_inputs_arrived(_all_inputs_arrived), all_inputs_finished(_all_inputs_finished), all_outputs_arrived(_all_outputs_arrived), all_outputs_finished(_all_outputs_finished), iter(_iter) {}
};
template<pc_layout T> struct consumer_finish_args {
    using finish_block_type = typename detail::finish_block_getter<T>::type;
    using consumer_state_type = typename detail::consumer_state_getter<T>::type;
    using scratch_block_type = typename detail::scratch_block_getter<T>::type;
    using globals_type = typename T::globals;
    finish_block_type & finish;
    consumer_state_type & state;
    scratch_block_type & scratch;
    globals_type & globals;
    barrier * all_inputs_arrived;
    barrier * all_inputs_finished;
    barrier * all_outputs_arrived;
    barrier * all_outputs_finished;
    int & iter;
    __device__ consumer_finish_args(finish_block_type& _finish, consumer_state_type& _state, scratch_block_type& _scratch, globals_type& _globals, barrier * _all_inputs_arrived, barrier * _all_inputs_finished, barrier * _all_outputs_arrived, barrier * _all_outputs_finished, int & _iter)
        : finish(_finish), state(_state), scratch(_scratch), globals(_globals), all_inputs_arrived(_all_inputs_arrived), all_inputs_finished(_all_inputs_finished), all_outputs_arrived(_all_outputs_arrived), all_outputs_finished(_all_outputs_finished), iter(_iter) {}
};
template<typename pct> concept pc_template = requires {
    typename pct::layout;
    typename pct::producer;
    typename pct::consumer;
    pct::iters;
    pct::producer::setup;
    pct::producer::load;
    pct::consumer::setup; 
    pct::consumer::work;
} && pc_layout<typename pct::layout>;
namespace detail {
template<typename pct> concept has_store  = requires { pct::producer::store; };
template<typename pct> concept has_finish = requires { pct::consumer::finish; };
template<typename pct> concept has_num_consumer_warps = requires { pct::NUM_CONSUMER_WARPS; };
template<typename pct> concept has_num_producer_warps = requires { pct::NUM_PRODUCER_WARPS; };
template<typename pct> concept has_enable_explicit_barriers = requires { pct::ENABLE_EXPLICIT_BARRIERS; };
template<typename pct> concept has_input_pipe_stages = requires { pct::INPUT_PIPE_STAGES; };
template<typename pct> concept has_output_pipe_stages = requires { pct::producer::store; };
template<typename pct> concept has_num_blocks = requires { pct::NUM_BLOCKS; };
template<typename pct> concept has_debug = requires { pct::DEBUG; };
}
template<typename pct> constexpr bool enable_explicit_barriers = false;
template<detail::has_enable_explicit_barriers pct> constexpr bool enable_explicit_barriers<pct> = pct::ENABLE_EXPLICIT_BARRIERS;
template<typename pct> constexpr int output_pipe_stages = 1;
template<detail::has_output_pipe_stages pct> constexpr int output_pipe_stages<pct> = pct::OUTPUT_PIPE_STAGES;
template<typename pct> constexpr int num_consumer_warps = 8;
template<detail::has_num_consumer_warps pct> constexpr int num_consumer_warps<pct> = pct::NUM_CONSUMER_WARPS;
template<typename pct> constexpr int num_producer_warps = 4;
template<detail::has_num_producer_warps pct> constexpr int num_producer_warps<pct> = pct::NUM_PRODUCER_WARPS;
template<typename pct> constexpr int num_blocks = 1;
template<detail::has_num_blocks pct> constexpr int num_blocks<pct> = pct::NUM_BLOCKS;
template<typename pct> constexpr int input_pipe_stages = (
    (MAX_SHARED_MEMORY-2048)/num_blocks<pct> - 
    sizeof(typename complete_pc_layout<typename pct::layout>::scratch_block_type)
    - output_pipe_stages<pct>*sizeof(typename complete_pc_layout<typename pct::layout>::output_block_type)
) / sizeof(typename pct::layout::input_block);
template<detail::has_input_pipe_stages pct> constexpr int input_pipe_stages<pct> = pct::INPUT_PIPE_STAGES;
template<typename pct> constexpr bool debug_status = false;
template<detail::has_debug pct> constexpr int debug_status<pct> = pct::DEBUG;

template<pc_template T> constexpr int num_threads = (num_consumer_warps<T> + num_producer_warps<T>) * 32;
template<pc_template T> constexpr int num_warps = num_consumer_warps<T> + num_producer_warps<T>;
template<pc_template T> constexpr int num_consumer_warpgroups = num_consumer_warps<T> / 4;

template<int N> __device__ static inline int ring_advance(int ring, int distance=1) { return (ring + distance) % N; }
template<int N> __device__ static inline int ring_retreat(int ring) { return (ring + N-1) % N; }

template<typename pct>
__global__ __launch_bounds__(num_threads<pct>, num_blocks<pct>)
void pc(typename pct::layout::globals g) {
    static_assert(pc_template<pct>, "pc template parameter does not satisfy concept requirements");
    using layout         = complete_pc_layout<typename pct::layout>; // complete the layout by filling in the optional types with empty
    using globals        = typename layout::globals;
    using input_block    = typename layout::input_block;
    using producer_state = typename layout::producer_state_type;
    using consumer_state = typename layout::consumer_state_type;
    using output_block   = typename layout::output_block_type;
    using scratch_block  = typename layout::scratch_block_type;
    using finish_block   = typename layout::finish_block_type;
    constexpr int OUTPUT_PIPE_STAGES = output_pipe_stages<pct> < 1 ? 1 : output_pipe_stages<pct>; // Even if the producer doesn't have a store, we need to allocate the empty struct
    static_assert(OUTPUT_PIPE_STAGES >= 1 && OUTPUT_PIPE_STAGES <= 32, "Invalid number of output pipe stages");
    constexpr int NUM_BLOCKS = num_blocks<pct>;
    constexpr int INPUT_PIPE_STAGES = input_pipe_stages<pct>;
    static_assert(INPUT_PIPE_STAGES >= 1 && INPUT_PIPE_STAGES <= 32, "Invalid number of input pipe stages");
    constexpr int NUM_CONSUMER_WARPS = num_consumer_warps<pct>;
    constexpr int NUM_PRODUCER_WARPS = num_producer_warps<pct>;

    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    input_block   (&input_smem)  [INPUT_PIPE_STAGES]  = alloc.allocate<input_block,  INPUT_PIPE_STAGES >();
    output_block  (&output_smem) [OUTPUT_PIPE_STAGES] = alloc.allocate<output_block, OUTPUT_PIPE_STAGES>();
    scratch_block (&scratch_smem)                     = alloc.allocate<scratch_block>();
    finish_block  (&finish_smem)                      = reinterpret_cast<finish_block&>(input_smem);

    if constexpr (debug_status<pct>) {
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            printf("DEBUG REPORT FOR PRODUCER TEMPLATE KERNEL:\n");
            printf("    BLOCK INFORMATION\n");
            printf("        gridDim.x:                        %d\n", gridDim.x);
            printf("        gridDim.y:                        %d\n", gridDim.y);
            printf("        gridDim.z:                        %d\n", gridDim.z);
            printf("        blockDim.x:                       %d\n", blockDim.x);
            printf("        blockDim.y:                       %d\n", blockDim.y);
            printf("        blockDim.z:                       %d\n", blockDim.z);
            printf("        num_blocks per SM:                %d\n", num_blocks<pct>);
            printf("        num_threads per SM:               %d\n", num_threads<pct>);
            printf("        num_warps per SM:                 %d\n", num_warps<pct>);
            printf("        num_consumer_warpgroups:          %d\n", num_consumer_warpgroups<pct>);
            printf("        num_consumer_warps:               %d\n", num_consumer_warps<pct>);
            printf("        num_producer_warps:               %d\n", num_producer_warps<pct>);
            printf("        has_store:                        %d\n", detail::has_store<pct>);
            printf("        has_finish:                       %d\n", detail::has_finish<pct>);
            printf("    PIPELINE INFORMATION\n");
            printf("        input_pipe_stages:                %d\n", INPUT_PIPE_STAGES);
            printf("        output_pipe_stages:               %d\n", OUTPUT_PIPE_STAGES);
            printf("    SHARED MEMORY INFORMATION\n");
            printf("        input_smem block size:            %llu\n", sizeof(input_block));
            printf("        input_smem:                       %p\n", (void*)&input_smem);
            printf("        input_smem size:                  %llu\n", INPUT_PIPE_STAGES*sizeof(input_block));
            printf("        output_smem block size:           %llu\n", sizeof(output_block));
            printf("        output_smem:                      %p\n", (void*)&output_smem);
            printf("        output_smem size:                 %llu\n", OUTPUT_PIPE_STAGES*sizeof(output_block));
            printf("        scratch_smem:                     %p\n", (void*)&scratch_smem);
            printf("        scratch_smem size:                %llu\n", sizeof(scratch_block));
            printf("        finish_smem:                      %p\n", (void*)&finish_smem);
            printf("        finish_smem size:                 %llu\n", sizeof(finish_block));
            printf("        dynamic shared memory usage:      %llu\n", sizeof(scratch_block) + uint64_t(&scratch_smem) - uint64_t(&__shm[0]));
        }
        __syncthreads();
    }

    // Initialize barriers. This is constant for all two-stage producer-consumer kernels.
    __shared__ kittens::barrier inputs_arrived[INPUT_PIPE_STAGES],   inputs_finished[INPUT_PIPE_STAGES];
    __shared__ kittens::barrier outputs_arrived[OUTPUT_PIPE_STAGES], outputs_finished[OUTPUT_PIPE_STAGES];
    if (warpid() == 0) { // a single warp (in fact a single thread) does these.
        for(int i = 0; i < INPUT_PIPE_STAGES; i++) {
            init_barrier(inputs_arrived[i], NUM_PRODUCER_WARPS, 0); // needs to wait on each producer warp
            init_barrier(inputs_finished[i], NUM_CONSUMER_WARPS, 0); // needs to wait on one thread from each consumer warp
        }
        if constexpr (detail::has_store<pct>) {
            for(int i = 0; i < OUTPUT_PIPE_STAGES; i++) {
                init_barrier(outputs_arrived[i], NUM_CONSUMER_WARPS, 0); // needs to wait on one thread from each consumer warp
                init_barrier(outputs_finished[i], NUM_PRODUCER_WARPS, 0); // needs to wait on each producer warp
            }
        }
    }
    int iters = pct::iters(g);
    int input_ring  = 0; // tracking which input block is being loaded
    int output_ring = 0; // tracking which output block is being written

    __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

    if(warpid() >= NUM_CONSUMER_WARPS) { // last warpgroup is a producer
        producer_state s;
        pct::producer::setup({
            g,
            s,
            scratch_smem,
            &inputs_arrived[0],
            &inputs_finished[0],
            &outputs_arrived[0],
            &outputs_finished[0]
        });
        int load_iter = 0, store_iter = 0;
        #pragma unroll
        for(int i = 0; i < INPUT_PIPE_STAGES && load_iter<iters; i++) { // fill the pipeline
            pct::producer::load({
                input_smem[input_ring],
                s,
                scratch_smem,
                g,
                inputs_arrived[input_ring],
                &inputs_arrived[0],
                &inputs_finished[0],
                &outputs_arrived[0],
                &outputs_finished[0],
                load_iter
            });
            input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            load_iter++;
        }
        if constexpr (detail::has_store<pct>) {
            while(load_iter<iters || store_iter<load_iter) {
                if(store_iter<load_iter && (enable_explicit_barriers<pct> || test_wait(outputs_arrived[output_ring], (store_iter/OUTPUT_PIPE_STAGES)%2))) {
                    pct::producer::store({
                        output_smem[output_ring],
                        s,
                        scratch_smem,
                        g,
                        outputs_finished[output_ring],
                        &inputs_arrived[0],
                        &inputs_finished[0],
                        &outputs_arrived[0],
                        &outputs_finished[0],
                        store_iter
                    });
                    output_ring=ring_advance<OUTPUT_PIPE_STAGES>(output_ring);
                    store_iter++;
                }
                // need to wait for the next stage to be available to write to.
                if(load_iter<iters && (enable_explicit_barriers<pct> || test_wait(inputs_finished[input_ring], ((load_iter/INPUT_PIPE_STAGES)%2)^1))) {
                    pct::producer::load({
                        input_smem[input_ring],
                        s,
                        scratch_smem,
                        g,
                        inputs_arrived[input_ring],
                        &inputs_arrived[0],
                        &inputs_finished[0],
                        &outputs_arrived[0],
                        &outputs_finished[0],
                        load_iter
                    });
                    input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
                    load_iter++;
                }
                __nanosleep(5);
            }
        }
        else { // just do the load
            while(load_iter<iters) {
                if constexpr (!enable_explicit_barriers<pct>) {
                    wait(inputs_finished[input_ring], ((load_iter/INPUT_PIPE_STAGES)%2)^1);
                }
                pct::producer::load({
                    input_smem[input_ring],
                    s,
                    scratch_smem,
                    g,
                    inputs_arrived[input_ring],
                    &inputs_arrived[0],
                    &inputs_finished[0],
                    &outputs_arrived[0],
                    &outputs_finished[0],
                    load_iter
                });
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
                load_iter++;
            }
        }
    }
    else { // other warpgroups are consumers
        consumer_state s;
        pct::consumer::setup({
            g,
            s,
            scratch_smem,
            &inputs_arrived[0],
            &inputs_finished[0],
            &outputs_arrived[0],
            &outputs_finished[0]
        });
        for(int it = 0; it < iters; it++) {
            if constexpr(!enable_explicit_barriers<pct>) {
                if constexpr (detail::has_store<pct>) {
                    wait(outputs_finished[output_ring], ((it/OUTPUT_PIPE_STAGES)%2)^1); // wait for output to free up
                }
                wait(inputs_arrived[input_ring], (it/INPUT_PIPE_STAGES)%2); // wait for memory to arrive, phase changes at half the rate of the ring
            }
            pct::consumer::work({
                input_smem[input_ring],
                output_smem[output_ring],
                s,
                scratch_smem,
                g,
                inputs_finished[input_ring],
                outputs_arrived[output_ring],
                &inputs_arrived[0],
                &inputs_finished[0],
                &outputs_arrived[0],
                &outputs_finished[0],
                it
            });
            input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            if constexpr (detail::has_store<pct>) {
                output_ring=ring_advance<OUTPUT_PIPE_STAGES>(output_ring);
            }
        }
        if constexpr (detail::has_finish<pct>) {
            group<NUM_CONSUMER_WARPS>::sync(0); // cannot overwrite finish block until all consumer warps are done.
            pct::consumer::finish({
                finish_smem,
                s,
                scratch_smem,
                g,
                &inputs_arrived[0],
                &inputs_finished[0],
                &outputs_arrived[0],
                &outputs_finished[0],
                iters
            });
        }
    }
}

}
}