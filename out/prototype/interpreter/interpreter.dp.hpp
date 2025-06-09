#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kittens.dp.hpp"
#include "../common/common.dp.hpp"
#include "templates.dp.hpp"
#include <time.h>

namespace kittens {
namespace prototype {
namespace interpreter {

static constexpr int NUM_PRODUCER_WARPS = 4;
static constexpr int NUM_CONSUMER_WARPS = 8;
static constexpr int NUM_WARPS = NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS;
static constexpr int NUM_BLOCKS = 1;

template<typename T> concept kernel_template = requires {
    typename T::layout;
    typename T::producer;
    typename T::consumer;
    T::common_setup;
    T::producer::setup;
    T::producer::load;
    T::consumer::setup; 
    T::consumer::compute;
    T::consumer::finish;
} && kittens_layout<typename T::layout>;
template<typename T> concept has_store = requires {
    T::producer::store;
};

template<typename Op> inline void run_op_producer(const typename Op::config::globals &globals, persistent_state &ps) {
    static_assert(kernel_template<Op>, "interpreter producer kernel template parameter does not satisfy concept requirements");
    using L              = typename Op::layout;
    using CKL            = complete_kittens_layout<L>; // complete the layout by filling in the optional types with empty
    using common_state   = typename CKL::common_state_t;
    using producer_state = typename CKL::producer_state_t;
    using consumer_state = typename CKL::consumer_state_t;
    using input_block    = typename CKL::input_block_t;
    using output_block   = typename CKL::output_block_t;
    using scratch_block  = typename CKL::scratch_block_t;
    using finish_block   = typename CKL::finish_block_t;
    using input_alloc_block   = typename CKL::input_alloc_block_t;
    using output_alloc_block  = typename CKL::output_alloc_block_t;
    using scratch_alloc_block = typename CKL::scratch_alloc_block_t;
    constexpr int MAX_BLOCK_SHARED_MEMORY = detail::MAX_SHARED_MEMORY_v<Op>;
    constexpr int INPUT_PIPE_STAGES   = detail::INPUT_PIPE_STAGES_v<Op>;
    static_assert(INPUT_PIPE_STAGES  >= 1 && INPUT_PIPE_STAGES <= 8, "Invalid number of input pipe stages");
    if constexpr (has_store<Op>) {
        constexpr int OUTPUT_PIPE_STAGES  = detail::OUTPUT_PIPE_STAGES_v<Op>;
        static_assert(OUTPUT_PIPE_STAGES >= 1 && OUTPUT_PIPE_STAGES <= 8, "Invalid number of output pipe stages");
        static_assert(
            INPUT_PIPE_STAGES*sizeof(input_alloc_block) +
            OUTPUT_PIPE_STAGES*sizeof(output_alloc_block) +
            sizeof(scratch_alloc_block)
            <= MAX_BLOCK_SHARED_MEMORY, "Shared memory usage exceeds limits"
        );
        
        shared_allocator alloc(ps.shmem); // allocate shared memory
        input_alloc_block   (&input_smem)  [INPUT_PIPE_STAGES]  = alloc.allocate<input_alloc_block,  INPUT_PIPE_STAGES >();
        output_alloc_block  (&output_smem) [OUTPUT_PIPE_STAGES] = alloc.allocate<output_alloc_block, OUTPUT_PIPE_STAGES>();

        // Put the scratch block at the very end of shared memory.
        scratch_alloc_block (*scratch_smem) = reinterpret_cast<scratch_alloc_block*>((((uint64_t)(ps.shmem) + (MAX_BLOCK_SHARED_MEMORY - sizeof(scratch_alloc_block)))/1024)*1024);
        // Finish block will come right before the scratch block.
        finish_block  (*finish_smem) = reinterpret_cast<finish_block*>((((uint64_t)(scratch_smem) - sizeof(finish_block))/1024)*1024);
        static_assert(sizeof(scratch_alloc_block) + sizeof(finish_block) <= MAX_BLOCK_SHARED_MEMORY, "Finish block is too large for shared memory.");

        int danger_stage = ps.max_finish_offset/sizeof(input_alloc_block);
        ps.max_finish_offset = (uint64_t)(finish_smem) - (uint64_t)(ps.shmem);
        common_state common;

        using producers = group<NUM_PRODUCER_WARPS>;
        producer_state p_state;
        int num_iters = 0;
        common_setup_args<L> unif{common, ps.task_iter, num_iters, globals, **scratch_smem, ps.instruction};
#ifdef KITTENS_TIMINGS
        unif.timings = ps.timings;
#endif
        Op::common_setup(unif);
        if(num_iters < 0) return; // no work to do
        int input_ring  = 0; // tracking which input block is being loaded
        int output_ring = 0; // tracking which output block is being written
        int load_iter, store_iter = 0;
        Op::producer::setup({p_state, unif});
        for(load_iter = 0; load_iter<INPUT_PIPE_STAGES && load_iter<num_iters; load_iter++) { // fill the pipeline
            wait(ps.inputs_finished[input_ring], get_phasebit<1>(ps.semaphore_bitfield, input_ring));
            if(load_iter == danger_stage) wait(*ps.finish_finished, (ps.task_iter%2)^1);
            update_phasebit<1>(ps.semaphore_bitfield, input_ring);
            Op::producer::load({p_state, *input_smem[input_ring], ps.inputs_arrived[input_ring], load_iter, unif});
            input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
        }
        while(load_iter<num_iters || store_iter<load_iter) {
            if(store_iter<load_iter && test_wait(ps.outputs_arrived[output_ring], get_phasebit<0>(ps.semaphore_bitfield, output_ring))) {
                update_phasebit<0>(ps.semaphore_bitfield, output_ring); // second half
                Op::producer::store({p_state, *output_smem[output_ring], ps.outputs_finished[output_ring], store_iter, unif});
                output_ring=ring_advance<OUTPUT_PIPE_STAGES>(output_ring);
                store_iter++;
            } // poll store, do store
            // need to wait for the next stage to be available to write to.
            if(load_iter<num_iters && test_wait(ps.inputs_finished[input_ring], get_phasebit<1>(ps.semaphore_bitfield, input_ring))) {
                update_phasebit<1>(ps.semaphore_bitfield, input_ring);
                Op::producer::load({p_state, *input_smem[input_ring], ps.inputs_arrived[input_ring], load_iter, unif});
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
                load_iter++;
            } // poll load, do load
            /*
            DPCT1008:486: __nanosleep function is not defined in SYCL. This is a
            hardware-specific feature. Consult with your hardware vendor to find
            a replacement.
            */
            __nanosleep(10); // relinquish for a little while
        } // load and store loop
        producers::sync(14); // producer warps must finish before proceeding
    }
    else {
        static_assert(
            INPUT_PIPE_STAGES*sizeof(input_alloc_block) + sizeof(scratch_alloc_block)
            <= MAX_SHARED_MEMORY-1024, "Shared memory usage exceeds limits"
        );
        
        shared_allocator alloc(ps.shmem); // allocate shared memory
        input_alloc_block   (&input_smem)  [INPUT_PIPE_STAGES]  = alloc.allocate<input_alloc_block,  INPUT_PIPE_STAGES>();

        // Put the scratch block at the very end of shared memory.
        scratch_alloc_block (*scratch_smem) = reinterpret_cast<scratch_alloc_block*>((((uint64_t)(ps.shmem) + (MAX_BLOCK_SHARED_MEMORY - sizeof(scratch_alloc_block)))/1024)*1024);
        // Finish block will come right before the scratch block.
        finish_block  (*finish_smem) = reinterpret_cast<finish_block*>((((uint64_t)(scratch_smem) - sizeof(finish_block))/1024)*1024);
        static_assert(sizeof(scratch_alloc_block) + sizeof(finish_block) <= MAX_BLOCK_SHARED_MEMORY, "Finish block is too large for shared memory.");
        
        int danger_stage = ps.max_finish_offset/sizeof(input_alloc_block);
        ps.max_finish_offset = (uint64_t)(finish_smem) - (uint64_t)(ps.shmem);
        common_state common;

        using producers = group<NUM_PRODUCER_WARPS>;
        producer_state p_state;
        int num_iters = -1;
        common_setup_args<L> unif{common, ps.task_iter, num_iters, globals, **scratch_smem, ps.instruction};
#ifdef KITTENS_TIMINGS
        unif.timings = ps.timings;
#endif
        Op::common_setup(unif);
        if(num_iters < 0) return; // no work to do
        int input_ring = 0; // tracking which input block is being loaded
        int load_iter;
        Op::producer::setup({p_state, unif});
        for(load_iter = 0; load_iter < INPUT_PIPE_STAGES && load_iter<num_iters; load_iter++) { // fill the pipeline
            wait(ps.inputs_finished[input_ring], get_phasebit<1>(ps.semaphore_bitfield, input_ring));
            if(load_iter == danger_stage) wait(*ps.finish_finished, (ps.task_iter%2)^1);
            update_phasebit<1>(ps.semaphore_bitfield, input_ring);
            Op::producer::load({p_state, *input_smem[input_ring], ps.inputs_arrived[input_ring], load_iter, unif});
            input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
        }
        for(; load_iter<num_iters; load_iter++) { // fill the pipeline
            wait(ps.inputs_finished[input_ring], get_phasebit<1>(ps.semaphore_bitfield, input_ring));
            update_phasebit<1>(ps.semaphore_bitfield, input_ring);
            Op::producer::load({p_state, *input_smem[input_ring], ps.inputs_arrived[input_ring], load_iter, unif});
            input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
        }
        producers::sync(14); // producer warps must finish before consumer warps can proceed
    }
}
template<typename Op> inline void run_op_consumer(const typename Op::config::globals &globals, persistent_state &ps) {
    static_assert(kernel_template<Op>, "interpreter consumer kernel template parameter does not satisfy concept requirements");
    using L              = typename Op::layout;
    using CKL            = complete_kittens_layout<L>; // complete the layout by filling in the optional types with empty
    using common_state   = typename CKL::common_state_t;
    using producer_state = typename CKL::producer_state_t;
    using consumer_state = typename CKL::consumer_state_t;
    using input_block    = typename CKL::input_block_t;
    using output_block   = typename CKL::output_block_t;
    using scratch_block  = typename CKL::scratch_block_t;
    using finish_block   = typename CKL::finish_block_t;
    using input_alloc_block   = typename CKL::input_alloc_block_t;
    using output_alloc_block  = typename CKL::output_alloc_block_t;
    using scratch_alloc_block = typename CKL::scratch_alloc_block_t;
    constexpr int MAX_BLOCK_SHARED_MEMORY = detail::MAX_SHARED_MEMORY_v<Op>;
    constexpr int INPUT_PIPE_STAGES   = detail::INPUT_PIPE_STAGES_v<Op>;
    static_assert(INPUT_PIPE_STAGES  >= 1 && INPUT_PIPE_STAGES <= 8, "Invalid number of input pipe stages");
    if constexpr (has_store<Op>) {
        constexpr int OUTPUT_PIPE_STAGES  = detail::OUTPUT_PIPE_STAGES_v<Op>;
        static_assert(OUTPUT_PIPE_STAGES >= 1 && OUTPUT_PIPE_STAGES <= 8, "Invalid number of output pipe stages");
        static_assert(
            INPUT_PIPE_STAGES*sizeof(input_alloc_block) +
            OUTPUT_PIPE_STAGES*sizeof(output_alloc_block) +
            sizeof(scratch_alloc_block)
            <= MAX_BLOCK_SHARED_MEMORY, "Shared memory usage exceeds limits"
        );
        
        shared_allocator alloc(ps.shmem); // allocate shared memory
        input_alloc_block   (&input_smem)  [INPUT_PIPE_STAGES]  = alloc.allocate<input_alloc_block,  INPUT_PIPE_STAGES >();
        output_alloc_block  (&output_smem) [OUTPUT_PIPE_STAGES] = alloc.allocate<output_alloc_block, OUTPUT_PIPE_STAGES>();

        // Put the scratch block at the very end of shared memory.
        scratch_alloc_block (*scratch_smem) = reinterpret_cast<scratch_alloc_block*>((((uint64_t)(ps.shmem) + (MAX_BLOCK_SHARED_MEMORY - sizeof(scratch_alloc_block)))/1024)*1024);
        // Finish block will come right before the scratch block.
        finish_block  (*finish_smem) = reinterpret_cast<finish_block*>((((uint64_t)(scratch_smem) - sizeof(finish_block))/1024)*1024);
        static_assert(sizeof(scratch_alloc_block) + sizeof(finish_block) <= MAX_BLOCK_SHARED_MEMORY, "Finish block is too large for shared memory.");

        common_state common;

        using consumers = group<NUM_CONSUMER_WARPS>;
        consumer_state c_state;
        int num_iters = 0;
        common_setup_args<L> unif{common, ps.task_iter, num_iters, globals, **scratch_smem, ps.instruction};
#ifdef KITTENS_TIMINGS
        unif.timings = ps.timings;
#endif
        Op::common_setup(unif);
        if(num_iters < 0) return; // no work to do
        int input_ring  = 0; // tracking which input block is being loaded
        int output_ring = 0; // tracking which output block is being written
        Op::consumer::setup({c_state, unif});
#ifdef CONSUMER_UNROLL
        #pragma unroll CONSUMER_UNROLL_VALUE
#endif
        for(int it = 0; it < num_iters; it++) {
            wait(ps.outputs_finished[output_ring], get_phasebit<1>(ps.semaphore_bitfield, output_ring)); // wait for output to free up
            update_phasebit<1>(ps.semaphore_bitfield, output_ring);
            wait(ps.inputs_arrived[input_ring], get_phasebit<0>(ps.semaphore_bitfield, input_ring)); // wait for memory to arrive, phase changes at half the rate of the ring
            update_phasebit<0>(ps.semaphore_bitfield, input_ring);
            Op::consumer::compute({c_state, *input_smem[input_ring], *output_smem[output_ring], ps.inputs_finished[input_ring], ps.outputs_arrived[output_ring], it, unif});
            input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            output_ring=ring_advance<OUTPUT_PIPE_STAGES>(output_ring);
        } // compute loop
        // ensure the outputs are all written before overwriting that memory
        #pragma unroll
        for(int i = 0; i < OUTPUT_PIPE_STAGES; i++) {
            wait(ps.outputs_finished[i], get_phasebit<1>(ps.semaphore_bitfield, i));
        }
        // no need to update phase bit, as nothing is actually changing before the next wait starts.
        consumers::sync(15); // cannot overwrite finish block until all consumer warps are done.
        Op::consumer::finish({c_state, *finish_smem, *ps.finish_finished, unif});
        consumers::sync(15); // cannot overwrite finish block until all consumer warps are done.
    }
    else {
        static_assert(
            INPUT_PIPE_STAGES*sizeof(input_alloc_block) + sizeof(scratch_alloc_block)
            <= MAX_SHARED_MEMORY-1024, "Shared memory usage exceeds limits"
        );
        
        shared_allocator alloc(ps.shmem); // allocate shared memory
        input_alloc_block   (&input_smem)  [INPUT_PIPE_STAGES]  = alloc.allocate<input_alloc_block,  INPUT_PIPE_STAGES>();

        // Put the scratch block at the very end of shared memory.
        scratch_alloc_block (*scratch_smem) = reinterpret_cast<scratch_alloc_block*>((((uint64_t)(ps.shmem) + (MAX_BLOCK_SHARED_MEMORY - sizeof(scratch_alloc_block)))/1024)*1024);
        // Finish block will come right before the scratch block.
        finish_block  (*finish_smem) = reinterpret_cast<finish_block*>((((uint64_t)(scratch_smem) - sizeof(finish_block))/1024)*1024);
        static_assert(sizeof(scratch_alloc_block) + sizeof(finish_block) <= MAX_BLOCK_SHARED_MEMORY, "Finish block is too large for shared memory.");
        
        common_state common;

        using consumers = group<NUM_CONSUMER_WARPS>;
        consumer_state c_state;
        int num_iters = -1;
        common_setup_args<L> unif{common, ps.task_iter, num_iters, globals, **scratch_smem, ps.instruction};
#ifdef KITTENS_TIMINGS
        unif.timings = ps.timings;
#endif
        Op::common_setup(unif);
        if(num_iters < 0) return; // no work to do
        int input_ring = 0; // tracking which input block is being loaded
        Op::consumer::setup({c_state, unif});
#ifdef CONSUMER_UNROLL
        #pragma unroll CONSUMER_UNROLL_VALUE
#endif
        for(int it = 0; it < num_iters; it++) {
            wait(ps.inputs_arrived[input_ring], get_phasebit<0>(ps.semaphore_bitfield, input_ring)); // wait for memory to arrive, phase changes at half the rate of the ring
            update_phasebit<0>(ps.semaphore_bitfield, input_ring);
            Op::consumer::compute({c_state, *input_smem[input_ring], ps.inputs_finished[input_ring], it, unif});
            input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
        } // work loop
        consumers::sync(15); // cannot overwrite finish block until all consumer warps are done.
        Op::consumer::finish({c_state, *finish_smem, *ps.finish_finished, unif});
        consumers::sync(15); // cannot overwrite finish block until all consumer warps are done.
    }
}

template<typename config, typename... ops>
struct dispatch_producer {
    static inline void run(int opcode, const typename config::globals &globals, persistent_state &ps) {} // do nothing, base case
};
template<typename config, typename op, typename... ops>
struct dispatch_producer<config, op, ops...> {
    static inline void run(int opcode, const typename config::globals &globals, persistent_state &ps) {
        if(opcode == op::opcode) run_op_producer<op>(globals, ps);
        else dispatch_producer<config, ops...>::run(opcode, globals, ps);
    }
};
template<typename config, typename... ops>
struct dispatch_consumer {
    static inline void run(int opcode, const typename config::globals &globals, persistent_state &ps) {} // do nothing, base case
};
template<typename config, typename op, typename... ops>
struct dispatch_consumer<config, op, ops...> {
    static inline void run(int opcode, const typename config::globals &globals, persistent_state &ps) {
        if(opcode == op::opcode) run_op_consumer<op>(globals, ps);
        else dispatch_consumer<config, ops...>::run(opcode, globals, ps);
    }
};

template<typename config> void inline load_instructions(int *instructions, int task_iter, const typename config::globals &globals, kittens::semaphore &bar) {
    if(laneid() == 0) {
        constexpr int ints_per_instruction = globals.instructions.cols();
        constexpr int bytes = ints_per_instruction*sizeof(int);
        tma::expect_bytes(bar, bytes);
        auto mbar_ptr = &bar;
        auto dst_ptr = instructions;
        uint64_t src_ptr = (uint64_t)(&globals.instructions[kittens::coord<>{
            (int)(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(
                2)),
            task_iter, 0}]);
        /*
        DPCT1053:315: Migration of device assembly code is not supported.
        */
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1], %3, [%2];"
            :
            : "r"(dst_ptr), "l"(src_ptr), "r"(mbar_ptr), "n"(bytes)
            : "memory");
    }
}

template<typename config, typename... ops>

__cluster_dims__(detail::CLUSTER_BLOCKS_v<config>)
void kernel(const typename config::globals globals,
            uint8_t *dpct_local) {
#ifdef KITTENS_TIMINGS
    uint64_t kernel_start = clock64();
    __shared__ uint64_t timings[2][64]; // We'll allow 64 separate timing events, per instruction.
    if(threadIdx.x < 64) { // 0 them all to start.
        timings[0][threadIdx.x] = 0;
        timings[1][threadIdx.x] = 0;
    }
#endif
#ifdef KITTENS_BLACKWELL
    auto tmem_alloc = allocate_tmem<1>(); // NUM_BLOCKS is hardcoded as 1 for the interpreter.
    auto &all_tmem = reinterpret_cast<tmem<float, 128, 512>&>(tmem_alloc);
#endif
    auto &instructions = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[2][globals.instructions.cols()]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &inputs_arrived = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        kittens::semaphore[8]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &inputs_finished =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::semaphore[8]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &outputs_arrived =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::semaphore[8]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &outputs_finished =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::semaphore[8]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &finish_finished =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::semaphore>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &instruction_arrived =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::semaphore[2]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &instruction_finished =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::semaphore[2]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if(warpid() == 0) {
        init_semaphore(instruction_arrived[0], 0, 1);
        init_semaphore(instruction_arrived[1], 0, 1);
        init_semaphore(instruction_finished[0], NUM_WARPS, 0);
        init_semaphore(instruction_finished[1], NUM_WARPS, 0);
    }
    auto __shm = (int *)dpct_local;
    persistent_state ps;
    ps.shmem = &__shm[0];
    ps.max_finish_offset = 1000000; // out of bounds, which is to say, no need to stall on the first instruction
    // Initialize semaphores. This is constant for all two-stage producer-consumer kernels.
    ps.semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
    if(warpid() < 8) {
        init_semaphore(inputs_arrived[warpid()], NUM_PRODUCER_WARPS, 0);
        init_semaphore(inputs_finished[warpid()], NUM_CONSUMER_WARPS, 0);
        init_semaphore(outputs_arrived[warpid()], NUM_CONSUMER_WARPS, 0);
        init_semaphore(outputs_finished[warpid()], NUM_PRODUCER_WARPS, 0);
    }
    if(warpid() == 0) init_semaphore(finish_finished, NUM_CONSUMER_WARPS, 0);
    ps.inputs_arrived = &inputs_arrived[0];
    ps.inputs_finished = &inputs_finished[0];
    ps.outputs_arrived = &outputs_arrived[0];
    ps.outputs_finished = &outputs_finished[0];
    ps.finish_finished = &finish_finished;
    if(detail::CLUSTER_BLOCKS_v<config> == 1) group<NUM_WARPS>::sync(15); // all warps must arrive here, confirming semaphore initialization is visible to all threads.
    else tma::cluster::sync();
    if(warpid() < NUM_CONSUMER_WARPS) { // CONSUMER WARPS
        warpgroup::increase_registers<232>();
        for(ps.task_iter = 0; ps.task_iter < globals.instructions.rows(); ps.task_iter++) {
            ps.instruction = &instructions[ps.task_iter%2][0];
#ifdef KITTENS_TIMINGS
            ps.timings = &timings[ps.task_iter%2][0];
#endif
            wait(instruction_arrived[ps.task_iter%2], ((ps.task_iter/2)%2));
            int opcode = ps.instruction[0];
            if(opcode == 0) return; // Stop Op
            dispatch_consumer<config, ops...>::run(opcode, globals, ps);
#ifdef KITTENS_TIMINGS
            if(threadIdx.x < 64) {
                if(ps.timings[threadIdx.x] != 0) {
                    globals.timings[kittens::coord<>{(int)(blockIdx.x), ps.task_iter, (int)(threadIdx.x)}] = (int)(ps.timings[threadIdx.x] - kernel_start);
                    ps.timings[threadIdx.x] = 0;
                }
            }
#endif
            if(laneid() == 0) arrive(instruction_finished[ps.task_iter%2]);
        }
    }
    else { // PRODUCER WARPS
        warpgroup::decrease_registers<40>();
        if(warpgroup::warpid() == 2) {
            load_instructions<config>(&instructions[0][0], 0, globals, instruction_arrived[0]); // load the instructions for the first task
        }
        ps.instruction = &instructions[1][0];
        for(ps.task_iter = 0; ps.task_iter < globals.instructions.rows(); ps.task_iter++) {
            if(warpgroup::warpid() == 2 && ps.task_iter+1 < globals.instructions.rows()) { // warp #2 doesn't get enough love
                wait(instruction_finished[(ps.task_iter+1)%2], ((ps.task_iter+3)/2)%2);
                sycl::group_barrier(
                    sycl::ext::oneapi::this_work_item::get_sub_group());
                load_instructions<config>(ps.instruction, ps.task_iter+1, globals, instruction_arrived[(ps.task_iter+1)%2]); // load next instruction into previous location
            }
            ps.instruction = &instructions[ps.task_iter%2][0]; // Update to the new one
#ifdef KITTENS_TIMINGS
            ps.timings = &timings[ps.task_iter%2][0];
#endif
            wait(instruction_arrived[ps.task_iter%2], ((ps.task_iter/2)%2));
            int opcode = ps.instruction[0];
            if(opcode == 0) break; // Stop Op
            dispatch_producer<config, ops...>::run(opcode, globals, ps);
            if(laneid() == 0) arrive(instruction_finished[ps.task_iter%2]);
        }
    }
}
template<typename config, typename... ops>
void run(typename config::globals &globals) {
    cudaFuncSetAttribute(kernel<config, ops...>, cudaFuncAttributeMaxDynamicSharedMemorySize, globals.dynamic_shared_memory());
    /*
    DPCT1049:316: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(globals.dynamic_shared_memory()), cgh);

            cgh.depends_on(
                dpct::get_current_device().get_in_order_queues_last_events());

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, globals.grid()) *
                                      sycl::range<3>(1, 1, globals.block()),
                                  sycl::range<3>(1, 1, globals.block())),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    kernel<config, ops...>(
                        globals,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
}

} // namespace interpreter
} // namespace prototype
} // namespace kittens