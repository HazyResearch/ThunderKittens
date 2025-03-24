#pragma once

#include "../include/kittens.cuh"
#include "../common/common.cuh"
#include "templates.cuh"

namespace kittens {
namespace prototype {
namespace lcf {

template<typename lcft> concept kernel_template = requires {
    typename lcft::layout;
    typename lcft::producer;
    typename lcft::consumer;
    lcft::common_setup;
    lcft::producer::setup;
    lcft::producer::load;
    lcft::consumer::setup; 
    lcft::consumer::compute;
    lcft::consumer::finish;
} && kittens_layout<typename lcft::layout>;

template<typename lcft> // load-compute-store-finish template
__global__ __launch_bounds__(detail::NUM_THREADS_v<lcft>, detail::NUM_BLOCKS_v<lcft>)
void kernel(const __grid_constant__ typename lcft::layout::globals globals) {
    static_assert(kernel_template<lcft>, "lcf kernel template parameter does not satisfy concept requirements");
    using L              = typename lcft::layout;
    using CKL            = complete_kittens_layout<L>; // complete the layout by filling in the optional types with empty
    using common_state   = typename CKL::common_state_t;
    using producer_state = typename CKL::producer_state_t;
    using consumer_state = typename CKL::consumer_state_t;
    using input_block    = typename CKL::input_block_t;
    using scratch_block  = typename CKL::scratch_block_t;
    using finish_block   = typename CKL::finish_block_t;
    using input_alloc_block   = typename CKL::input_alloc_block_t;
    using scratch_alloc_block = typename CKL::scratch_alloc_block_t;
    constexpr int MAX_SHARED_MEMORY = detail::MAX_SHARED_MEMORY_v<lcft>;
    constexpr int INPUT_PIPE_STAGES = detail::INPUT_PIPE_STAGES_v<lcft>;
    static_assert(INPUT_PIPE_STAGES >= 1 && INPUT_PIPE_STAGES <= 16, "Invalid number of input pipe stages");
    static_assert(
        INPUT_PIPE_STAGES*sizeof(input_alloc_block) + sizeof(scratch_alloc_block)
        <= MAX_SHARED_MEMORY-1024, "Shared memory usage exceeds limits"
    );
    constexpr int NUM_CONSUMER_WARPS = detail::NUM_CONSUMER_WARPS_v<lcft>;
    constexpr int NUM_PRODUCER_WARPS = detail::NUM_PRODUCER_WARPS_v<lcft>;
    
    using everyone = group<detail::NUM_WARPS_v<lcft>>;
    
    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    scratch_alloc_block (&scratch_smem)                     = alloc.allocate<scratch_alloc_block>();
    input_alloc_block   (&input_smem)  [INPUT_PIPE_STAGES]  = alloc.allocate<input_alloc_block,  INPUT_PIPE_STAGES>();
    
    // figure out where we're going to put the finish block
    constexpr int FINISH_BLOCK_OFFSET = (MAX_SHARED_MEMORY-1024)/detail::NUM_BLOCKS_v<lcft> - sizeof(finish_block);
    static_assert(FINISH_BLOCK_OFFSET >= 0, "Finish block is too large for shared memory.");
    constexpr int NON_FINISH_BLOCK_SPACE = FINISH_BLOCK_OFFSET - 1024 - sizeof(scratch_alloc_block); // including the losses from alignment
    constexpr int SAFE_STAGES_BETWEEN_BLOCKS = NON_FINISH_BLOCK_SPACE/sizeof(input_alloc_block)<INPUT_PIPE_STAGES?NON_FINISH_BLOCK_SPACE/sizeof(input_alloc_block):INPUT_PIPE_STAGES;
    finish_block  (*finish_smem) = reinterpret_cast<finish_block*>((((uint64_t)&__shm[0] + FINISH_BLOCK_OFFSET)/1024)*1024); // alignment

    if constexpr (detail::DEBUG_v<lcft>) {
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            printf("DEBUG REPORT FOR PRODUCER TEMPLATE KERNEL:\n");
            printf("    BLOCK INFORMATION\n");
            printf("        gridDim.x:                         %d\n", gridDim.x);
            printf("        gridDim.y:                         %d\n", gridDim.y);
            printf("        gridDim.z:                         %d\n", gridDim.z);
            printf("        blockDim.x:                        %d\n", blockDim.x);
            printf("        blockDim.y:                        %d\n", blockDim.y);
            printf("        blockDim.z:                        %d\n", blockDim.z);
            printf("        num_blocks per SM:                 %d\n", detail::NUM_BLOCKS_v<lcft>);
            printf("        num_threads per SM:                %d\n", detail::NUM_THREADS_v<lcft>);
            printf("        num_warps per SM:                  %d\n", detail::NUM_WARPS_v<lcft>);
            printf("        num_consumer_warpgroups:           %d\n", detail::NUM_CONSUMER_WARPGROUPS_v<lcft>);
            printf("        num_consumer_warps:                %d\n", detail::NUM_CONSUMER_WARPS_v<lcft>);
            printf("        num_producer_warps:                %d\n", detail::NUM_PRODUCER_WARPS_v<lcft>);
            printf("    PIPELINE INFORMATION\n"); 
            printf("        input_pipe_stages:                 %d\n", INPUT_PIPE_STAGES);
            printf("        safe_stages_between_blocks:        %d\n", SAFE_STAGES_BETWEEN_BLOCKS);
            printf("    SHARED MEMORY INFORMATION\n"); 
            printf("        input_smem block size:             %llu\n", sizeof(input_block));
            printf("        input_smem block size (aligned):   %llu\n", sizeof(input_alloc_block));
            printf("        input_smem:                        %p\n", (void*)&input_smem);
            printf("        input_smem size:                   %llu\n", INPUT_PIPE_STAGES*sizeof(input_alloc_block));
            printf("        scratch_smem block size:           %llu\n", sizeof(scratch_block));
            printf("        scratch_smem block size (aligned): %llu\n", sizeof(scratch_alloc_block));
            printf("        scratch_smem:                      %p\n", (void*)&scratch_smem);
            printf("        finish_smem:                       %p\n", (void*)finish_smem);
            printf("        finish_smem size:                  %llu\n", sizeof(finish_block));
            printf("        dynamic shared memory usage:       %llu\n", sizeof(scratch_alloc_block) + uint64_t(&scratch_smem) - uint64_t(&__shm[0]));
        }
        everyone::sync(15);
    }

    // Initialize semaphores. This is constant for all two-stage producer-consumer kernels.
    __shared__ kittens::semaphore inputs_arrived[INPUT_PIPE_STAGES], inputs_finished[INPUT_PIPE_STAGES];
    __shared__ kittens::semaphore finish_finished;
    uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
    common_state common;

    // if (laneid() == 0) printf("Warp %d entering %s path\n", warpid(), warpid() >= NUM_CONSUMER_WARPS ? "producer" : "consumer");

    if(warpid() >= NUM_CONSUMER_WARPS) { // code path for producer warps
        using producers = group<NUM_PRODUCER_WARPS>;
        if (warpid() == NUM_CONSUMER_WARPS) { // a single warp (in fact a single thread) does these.
            for(int i = 0; i < INPUT_PIPE_STAGES; i++) {
                init_semaphore(inputs_arrived[i], detail::PRODUCER_BARRIER_ARRIVALS_v<lcft>, 0); // needs to wait on each producer warp
                init_semaphore(inputs_finished[i], detail::CONSUMER_BARRIER_ARRIVALS_v<lcft>, 0); // needs to wait on one thread from each consumer warp
            }
            init_semaphore(finish_finished, detail::CONSUMER_BARRIER_ARRIVALS_v<lcft>, 0); // consumer warps must say they are done with the finish block
        }
        everyone::sync(15); // all warps must arrive here, confirming semaphore initialization is visible to all threads.
        producer_state p_state;
        for(int task_iter = 0; true; task_iter++) {
            int num_iters = -1;
            common_setup_args<L> unif{common, task_iter, num_iters, globals, *scratch_smem};
            lcft::common_setup(unif);
            if(num_iters <= 0) return; // no work to do
            int input_ring = 0; // tracking which input block is being loaded
            int load_iter;
            lcft::producer::setup({p_state, unif});
            for(load_iter = 0; load_iter < SAFE_STAGES_BETWEEN_BLOCKS && load_iter<num_iters; load_iter++) { // fill the pipeline
                wait(inputs_finished[input_ring], get_phasebit<1>(semaphore_bitfield, input_ring));
                update_phasebit<1>(semaphore_bitfield, input_ring);
                lcft::producer::load({p_state, *input_smem[input_ring], inputs_arrived[input_ring], load_iter, unif});
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            }
            wait(finish_finished, (task_iter%2)^1); // wait for consumer to finish their finish stage before we can do the rest.
            for(; load_iter<num_iters; load_iter++) { // fill the pipeline
                wait(inputs_finished[input_ring], get_phasebit<1>(semaphore_bitfield, input_ring));
                update_phasebit<1>(semaphore_bitfield, input_ring);
                lcft::producer::load({p_state, *input_smem[input_ring], inputs_arrived[input_ring], load_iter, unif});
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            }
            producers::sync(13); // producer warps must finish before consumer warps can proceed
        } // task iter loop
    } // producer warpgroup
    else { // code path for consumer warps
        using consumers = group<NUM_CONSUMER_WARPS>;
        everyone::sync(15); // all warps must arrive here, confirming semaphore initialization is visible to all threads.
        consumer_state c_state;
        for(int task_iter = 0; true; task_iter++) {
            int num_iters = -1;
            common_setup_args<L> unif{common, task_iter, num_iters, globals, *scratch_smem};
            lcft::common_setup(unif);
            if(num_iters <= 0) return; // no work to do
            int input_ring = 0; // tracking which input block is being loaded
            lcft::consumer::setup({c_state, unif});
#ifdef CONSUMER_UNROLL
            #pragma unroll CONSUMER_UNROLL_VALUE
#endif
            for(int it = 0; it < num_iters; it++) {
                wait(inputs_arrived[input_ring], get_phasebit<0>(semaphore_bitfield, input_ring)); // wait for memory to arrive, phase changes at half the rate of the ring
                update_phasebit<0>(semaphore_bitfield, input_ring);
                lcft::consumer::compute({c_state, *input_smem[input_ring], inputs_finished[input_ring], it, unif});
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            } // work loop
            consumers::sync(14); // cannot overwrite finish block until all consumer warps are done.
            lcft::consumer::finish({c_state, *finish_smem, finish_finished, unif});
            consumers::sync(14); // cannot overwrite finish block until all consumer warps are done.
        } // task iter loop
    } // consumer warpgroup
}

} // namespace lcf
} // namespace prototype
} // namespace kittens
