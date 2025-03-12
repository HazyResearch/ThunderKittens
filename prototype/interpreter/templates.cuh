#pragma once

#include "../common/common.cuh"

namespace kittens {
namespace prototype {
namespace interpreter {

struct persistent_state {
    int task_iter;
    int *shmem;
    int max_finish_offset;
    kittens::semaphore *inputs_arrived, *outputs_arrived, *inputs_finished, *outputs_finished, *finish_finished;
    int *instruction;
    uint32_t semaphore_bitfield;
#ifdef KITTENS_TIMINGS
    uint64_t *timings;
#endif
};

// All template functions take these args
template<kittens_layout T> struct uniform_args {
    using CKL = complete_kittens_layout<T>;
    typename CKL::common_state_t & common; // scratch for the coordinates of the task.
    int & task_iter; // which task are we on?
    int & num_iters; // how many iters are there for this task?
    const typename CKL::globals_t & globals;
    typename CKL::scratch_block_t & scratch;
    int *instruction;
#ifdef KITTENS_TIMINGS
    uint64_t *timings;
#endif
    __device__ uniform_args(
        typename CKL::common_state_t & _common,
        int & _task_iter,
        int & _num_iters,
        const typename CKL::globals_t& _globals,
        typename CKL::scratch_block_t& _scratch,
        int * _instruction
    ) : common(_common),
        task_iter(_task_iter),
        num_iters(_num_iters),
        globals(_globals),
        scratch(_scratch),
        instruction(_instruction)
    {}
    __device__ uniform_args(uniform_args<T> &_args) :
        common(_args.common),
        task_iter(_args.task_iter),
        num_iters(_args.num_iters),
        globals(_args.globals),
        scratch(_args.scratch),
        instruction(_args.instruction)
#ifdef KITTENS_TIMINGS
        , timings(_args.timings)
#endif
    {}
};

// Setup args are the same as uniform args
template<kittens_layout T> using common_setup_args = uniform_args<T>;

// Producer init args
template<kittens_layout T> struct producer_setup_args : uniform_args<T> {
    using CKL = complete_kittens_layout<T>;
    typename CKL::producer_state_t & state;
    __device__ producer_setup_args(
        typename CKL::producer_state_t& _state,
        uniform_args<T> &_args
    ) : uniform_args<T>(_args), state(_state) {}
};

// Producer load args
template<kittens_layout T> struct producer_load_args : uniform_args<T> {
    using CKL = complete_kittens_layout<T>;
    typename CKL::producer_state_t & state;
    typename CKL::input_block_t & input;
    kittens::semaphore & inputs_arrived;
    int iter;
    __device__ producer_load_args(
        typename CKL::producer_state_t& _state,
        typename CKL::input_block_t& _input,
        semaphore& _inputs_arrived,
        int _iter,
        uniform_args<T> &_args
    ) : uniform_args<T>(_args), input(_input), state(_state), inputs_arrived(_inputs_arrived), iter(_iter) {}
};

// Producer store args
template<store_kittens_layout T> struct producer_store_args : uniform_args<T> {
    using CKL = complete_kittens_layout<T>;
    typename CKL::producer_state_t & state;
    typename CKL::output_block_t & output;
    kittens::semaphore & outputs_finished;
    int iter;
    __device__ producer_store_args(
        typename CKL::producer_state_t& _state,
        typename CKL::output_block_t& _output,
        semaphore& _outputs_finished,
        int _iter,
        uniform_args<T> &_args
    ) : uniform_args<T>(_args), output(_output), state(_state), outputs_finished(_outputs_finished), iter(_iter) {}
};

// Consumer init args
template<kittens_layout T> struct consumer_setup_args : uniform_args<T> {
    using CKL = complete_kittens_layout<T>;
    typename CKL::consumer_state_t & state;
    __device__ consumer_setup_args(
        typename CKL::consumer_state_t& _state,
        uniform_args<T> &_args
    ) : uniform_args<T>(_args), state(_state) {}
};

// Consumer compute args
template<kittens_layout T> struct consumer_compute_args : uniform_args<T> {
    using CKL = complete_kittens_layout<T>;
    typename CKL::consumer_state_t & state;
    typename CKL::input_block_t & input;
    kittens::semaphore & inputs_finished;
    int iter;
    __device__ consumer_compute_args(
        typename CKL::consumer_state_t& _state,
        typename CKL::input_block_t& _input,
        semaphore& _inputs_finished,
        int _iter,
        uniform_args<T> &_args
    ) : uniform_args<T>(_args), input(_input), state(_state), inputs_finished(_inputs_finished), iter(_iter) {}
};
template<store_kittens_layout T> struct consumer_compute_args<T> : uniform_args<T> {
    using CKL = complete_kittens_layout<T>;
    typename CKL::consumer_state_t & state;
    typename CKL::input_block_t & input;
    typename CKL::output_block_t & output;
    kittens::semaphore & inputs_finished;
    kittens::semaphore & outputs_arrived;
    int iter;
    __device__ consumer_compute_args(
        typename CKL::consumer_state_t& _state,
        typename CKL::input_block_t& _input,
        typename CKL::output_block_t& _output,
        semaphore& _inputs_finished,
        semaphore& _outputs_arrived,
        int _iter,
        uniform_args<T> &_args
    ) : uniform_args<T>(_args), input(_input), output(_output), state(_state), inputs_finished(_inputs_finished), outputs_arrived(_outputs_arrived), iter(_iter) {}
};

// Consumer finish args
template<kittens_layout T> struct consumer_finish_args : uniform_args<T> {
    using CKL = complete_kittens_layout<T>;
    typename CKL::consumer_state_t & state;
    typename CKL::finish_block_t & finish;
    kittens::semaphore & finish_finished;
    __device__ consumer_finish_args(
        typename CKL::consumer_state_t& _state,
        typename CKL::finish_block_t& _finish,
        semaphore& _finish_finished,
        uniform_args<T> &_args
    ) : uniform_args<T>(_args), finish(_finish), state(_state), finish_finished(_finish_finished) {}
};

} // namespace interp
} // namespace prototype
} // namespace kittens
