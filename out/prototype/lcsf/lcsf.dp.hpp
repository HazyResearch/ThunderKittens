#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../include/kittens.dp.hpp"
#include "../common/common.dp.hpp"
#include "templates.dp.hpp"
#include <time.h>

namespace kittens {
namespace prototype {
namespace lcsf {

template<typename lcsft> concept kernel_template = requires {
    typename lcsft::layout;
    typename lcsft::producer;
    typename lcsft::consumer;
    lcsft::common_setup;
    lcsft::producer::setup;
    lcsft::producer::load;
    lcsft::producer::store;
    lcsft::consumer::setup; 
    lcsft::consumer::compute;
    lcsft::consumer::finish;
} && kittens_layout<typename lcsft::layout>;

template <typename lcsft> // load-compute-store-finish template
/*
DPCT1110:314: The total declared local variable size in device function kernel
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/

void kernel(const typename lcsft::layout::globals globals,
            uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    static_assert(
        kernel_template<lcsft>,
        "lcsf kernel template parameter does not satisfy concept requirements");
    using L              = typename lcsft::layout;
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
    constexpr int MAX_BLOCK_SHARED_MEMORY = detail::MAX_SHARED_MEMORY_v<lcsft>;
    constexpr int OUTPUT_PIPE_STAGES  = detail::OUTPUT_PIPE_STAGES_v<lcsft>;
    static_assert(OUTPUT_PIPE_STAGES >= 1 && OUTPUT_PIPE_STAGES <= 16, "Invalid number of output pipe stages");
    constexpr int INPUT_PIPE_STAGES   = detail::INPUT_PIPE_STAGES_v<lcsft>;
    static_assert(INPUT_PIPE_STAGES  >= 1 && INPUT_PIPE_STAGES <= 16, "Invalid number of input pipe stages");
    static_assert(
        INPUT_PIPE_STAGES*sizeof(input_alloc_block) +
        OUTPUT_PIPE_STAGES*sizeof(output_alloc_block) +
        sizeof(scratch_alloc_block)
        <= MAX_BLOCK_SHARED_MEMORY, "Shared memory usage exceeds limits"
    );
    constexpr int NUM_CONSUMER_WARPS = detail::NUM_CONSUMER_WARPS_v<lcsft>;
    constexpr int NUM_PRODUCER_WARPS = detail::NUM_PRODUCER_WARPS_v<lcsft>;
    using everyone = group<detail::NUM_WARPS_v<lcsft>>;

    auto __shm = (int *)dpct_local;
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    scratch_alloc_block (&scratch_smem)                     = alloc.allocate<scratch_alloc_block>();
    input_alloc_block   (&input_smem)  [INPUT_PIPE_STAGES]  = alloc.allocate<input_alloc_block,  INPUT_PIPE_STAGES >();
    output_alloc_block  (&output_smem) [OUTPUT_PIPE_STAGES] = alloc.allocate<output_alloc_block, OUTPUT_PIPE_STAGES>();
    
    // figure out where we're going to put the finish block
    constexpr int FINISH_BLOCK_OFFSET = (MAX_SHARED_MEMORY-1024)/detail::NUM_BLOCKS_v<lcsft> - sizeof(finish_block);
    static_assert(sizeof(scratch_alloc_block) + sizeof(finish_block) <= MAX_BLOCK_SHARED_MEMORY, "Finish block is too large for shared memory.");
    constexpr int NON_FINISH_BLOCK_SPACE = FINISH_BLOCK_OFFSET - 1024 - sizeof(scratch_alloc_block); // including the losses from alignment
    constexpr int SAFE_STAGES_BETWEEN_BLOCKS = NON_FINISH_BLOCK_SPACE/sizeof(input_alloc_block)<INPUT_PIPE_STAGES?NON_FINISH_BLOCK_SPACE/sizeof(input_alloc_block):INPUT_PIPE_STAGES;
    finish_block  (*finish_smem) = reinterpret_cast<finish_block*>((((uint64_t)&__shm[0] + FINISH_BLOCK_OFFSET)/1024)*1024); // alignment

    if constexpr (detail::DEBUG_v<lcsft>) {
        if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0 &&
            item_ct1.get_local_id(0) == 0 && item_ct1.get_group(2) == 0 &&
            item_ct1.get_group(1) == 0 && item_ct1.get_group(0) == 0) {
            sycl::ext::oneapi::experimental::printf(
                "DEBUG REPORT FOR PRODUCER TEMPLATE KERNEL:\n");
            sycl::ext::oneapi::experimental::printf("    BLOCK INFORMATION\n");
            sycl::ext::oneapi::experimental::printf(
                "        gridDim.x:                         %d\n",
                item_ct1.get_group_range(2));
            sycl::ext::oneapi::experimental::printf(
                "        gridDim.y:                         %d\n",
                item_ct1.get_group_range(1));
            sycl::ext::oneapi::experimental::printf(
                "        gridDim.z:                         %d\n",
                item_ct1.get_group_range(0));
            sycl::ext::oneapi::experimental::printf(
                "        blockDim.x:                        %d\n",
                item_ct1.get_local_range(2));
            sycl::ext::oneapi::experimental::printf(
                "        blockDim.y:                        %d\n",
                item_ct1.get_local_range(1));
            sycl::ext::oneapi::experimental::printf(
                "        blockDim.z:                        %d\n",
                item_ct1.get_local_range(0));
            printf("        num_blocks per SM:                 %d\n", detail::NUM_BLOCKS_v<lcsft>);
            printf("        num_threads per SM:                %d\n", detail::NUM_THREADS_v<lcsft>);
            printf("        num_warps per SM:                  %d\n", detail::NUM_WARPS_v<lcsft>);
            printf("        num_consumer_warpgroups:           %d\n", detail::NUM_CONSUMER_WARPGROUPS_v<lcsft>);
            printf("        num_consumer_warps:                %d\n", detail::NUM_CONSUMER_WARPS_v<lcsft>);
            printf("        num_producer_warps:                %d\n", detail::NUM_PRODUCER_WARPS_v<lcsft>);
            sycl::ext::oneapi::experimental::printf(
                "    PIPELINE INFORMATION\n");
            sycl::ext::oneapi::experimental::printf(
                "        input_pipe_stages:                 %d\n",
                INPUT_PIPE_STAGES);
            sycl::ext::oneapi::experimental::printf(
                "        output_pipe_stages:                %d\n",
                OUTPUT_PIPE_STAGES);
            sycl::ext::oneapi::experimental::printf(
                "        safe_stages_between_blocks:        %d\n",
                SAFE_STAGES_BETWEEN_BLOCKS);
            sycl::ext::oneapi::experimental::printf(
                "    SHARED MEMORY INFORMATION\n");
            sycl::ext::oneapi::experimental::printf(
                "        input_smem block size:             %llu\n",
                sizeof(input_block));
            sycl::ext::oneapi::experimental::printf(
                "        input_smem block size (aligned):   %llu\n",
                sizeof(input_alloc_block));
            sycl::ext::oneapi::experimental::printf(
                "        input_smem:                        %p\n",
                (void *)&input_smem);
            sycl::ext::oneapi::experimental::printf(
                "        input_smem size:                   %llu\n",
                INPUT_PIPE_STAGES * sizeof(input_alloc_block));
            sycl::ext::oneapi::experimental::printf(
                "        output_smem block size:            %llu\n",
                sizeof(output_block));
            sycl::ext::oneapi::experimental::printf(
                "        output_smem block size (aligned):  %llu\n",
                sizeof(output_alloc_block));
            sycl::ext::oneapi::experimental::printf(
                "        output_smem:                       %p\n",
                (void *)&output_smem);
            sycl::ext::oneapi::experimental::printf(
                "        output_smem size:                  %llu\n",
                OUTPUT_PIPE_STAGES * sizeof(output_alloc_block));
            sycl::ext::oneapi::experimental::printf(
                "        scratch_smem block size:           %llu\n",
                sizeof(scratch_block));
            sycl::ext::oneapi::experimental::printf(
                "        scratch_smem block size (aligned): %llu\n",
                sizeof(scratch_alloc_block));
            sycl::ext::oneapi::experimental::printf(
                "        scratch_smem:                      %p\n",
                (void *)&scratch_smem);
            sycl::ext::oneapi::experimental::printf(
                "        finish_smem:                       %p\n",
                (void *)finish_smem);
            sycl::ext::oneapi::experimental::printf(
                "        finish_smem size:                  %llu\n",
                sizeof(finish_block));
            sycl::ext::oneapi::experimental::printf(
                "        dynamic shared memory usage:       %llu\n",
                sizeof(output_block) +
                    uint64_t(&(*output_smem[OUTPUT_PIPE_STAGES - 1])) -
                    uint64_t(&__shm[0]));
        }
        everyone::sync(15);
    }

    // Initialize semaphores. This is constant for all two-stage producer-consumer kernels.
    auto &inputs_arrived = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        kittens::semaphore[INPUT_PIPE_STAGES]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &inputs_finished =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::semaphore[INPUT_PIPE_STAGES]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &outputs_arrived =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::semaphore[OUTPUT_PIPE_STAGES]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &outputs_finished =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::semaphore[OUTPUT_PIPE_STAGES]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &finish_finished =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::semaphore>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
    common_state common;

    if (item_ct1.get_local_id(2) ==
        0) { // a single warp (in fact a single thread) does these.
        for(int i = 0; i < INPUT_PIPE_STAGES; i++) {
            init_semaphore(inputs_arrived[i], detail::PRODUCER_BARRIER_ARRIVALS_v<lcsft>, 0); // needs to wait on each producer warp
            init_semaphore(inputs_finished[i], detail::CONSUMER_BARRIER_ARRIVALS_v<lcsft>, 0); // needs to wait on one thread from each consumer warp
        }
        for(int i = 0; i < OUTPUT_PIPE_STAGES; i++) {
            init_semaphore(outputs_arrived[i], detail::CONSUMER_BARRIER_ARRIVALS_v<lcsft>, 0); // needs to wait on one thread from each consumer warp
            init_semaphore(outputs_finished[i], detail::PRODUCER_BARRIER_ARRIVALS_v<lcsft>, 0); // needs to wait on each producer warp
        }
        init_semaphore(finish_finished, detail::CONSUMER_BARRIER_ARRIVALS_v<lcsft>, 0); // consumer warps must say they are done with the finish block
    }
    everyone::sync(15); // all warps must arrive here, confirming semaphore initialization is visible to all threads.

    if(warpid() >= NUM_CONSUMER_WARPS) { // code path for producer warps
        using producers = group<NUM_PRODUCER_WARPS>;
        producer_state p_state;
        for(int task_iter = 0; true; task_iter++) {
            int num_iters = 0;
            common_setup_args<L> unif{common, task_iter, num_iters, globals, *scratch_smem};
            lcsft::common_setup(unif);
            if(num_iters <= 0) {
                /*
                DPCT1118:312: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065:484: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                return; // no work to do
            }
            int input_ring  = 0; // tracking which input block is being loaded
            int output_ring = 0; // tracking which output block is being written
            int load_iter, store_iter = 0;
            lcsft::producer::setup({p_state, unif});
            for(load_iter = 0; load_iter<SAFE_STAGES_BETWEEN_BLOCKS && load_iter<num_iters; load_iter++) { // fill the pipeline
                wait(inputs_finished[input_ring], get_phasebit<1>(semaphore_bitfield, input_ring));
                update_phasebit<1>(semaphore_bitfield, input_ring);
                lcsft::producer::load({p_state, *input_smem[input_ring], inputs_arrived[input_ring], load_iter, unif});
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            }
            wait(finish_finished, (task_iter%2)^1); // wait for consumer to finish their finish stage before we can do the rest.
            for(; load_iter<INPUT_PIPE_STAGES && load_iter<num_iters; load_iter++) { // fill the pipeline
                wait(inputs_finished[input_ring], get_phasebit<1>(semaphore_bitfield, input_ring));
                update_phasebit<1>(semaphore_bitfield, input_ring);
                lcsft::producer::load({p_state, *input_smem[input_ring], inputs_arrived[input_ring], load_iter, unif});
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
            }
            while(load_iter<num_iters || store_iter<load_iter) {
                if(store_iter<load_iter && test_wait(outputs_arrived[output_ring], get_phasebit<0>(semaphore_bitfield, output_ring))) {
                    update_phasebit<0>(semaphore_bitfield, output_ring); // second half
                    lcsft::producer::store({p_state, *output_smem[output_ring], outputs_finished[output_ring], store_iter, unif});
                    output_ring=ring_advance<OUTPUT_PIPE_STAGES>(output_ring);
                    store_iter++;
                } // poll store, do store
                // need to wait for the next stage to be available to write to.
                if(load_iter<num_iters && test_wait(inputs_finished[input_ring], get_phasebit<1>(semaphore_bitfield, input_ring))) {
                    update_phasebit<1>(semaphore_bitfield, input_ring);
                    lcsft::producer::load({p_state, *input_smem[input_ring], inputs_arrived[input_ring], load_iter, unif});
                    input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
                    load_iter++;
                } // poll load, do load
                /*
                DPCT1008:483: __nanosleep function is not defined in SYCL. This
                is a hardware-specific feature. Consult with your hardware
                vendor to find a replacement.
                */
                __nanosleep(10); // relinquish for a little while
            } // load and store loop
            producers::sync(13); // producer warps must finish before consumer warps can proceed
        } // task iter loop
    } // producer warpgroup
    else { // code path for consumer warps
        using consumers = group<NUM_CONSUMER_WARPS>;
        consumer_state c_state;
        for(int task_iter = 0; true; task_iter++) {
            int num_iters = 0;
            common_setup_args<L> unif{common, task_iter, num_iters, globals, *scratch_smem};
            lcsft::common_setup(unif);
            if(num_iters <= 0) {
                /*
                DPCT1118:313: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065:485: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                return; // no work to do
            }
            int input_ring  = 0; // tracking which input block is being loaded
            int output_ring = 0; // tracking which output block is being written
            lcsft::consumer::setup({c_state, unif});
#ifdef CONSUMER_UNROLL
            #pragma unroll CONSUMER_UNROLL_VALUE
#endif
            for(int it = 0; it < num_iters; it++) {
                wait(outputs_finished[output_ring], get_phasebit<1>(semaphore_bitfield, output_ring)); // wait for output to free up
                update_phasebit<1>(semaphore_bitfield, output_ring);
                wait(inputs_arrived[input_ring], get_phasebit<0>(semaphore_bitfield, input_ring)); // wait for memory to arrive, phase changes at half the rate of the ring
                update_phasebit<0>(semaphore_bitfield, input_ring);
                lcsft::consumer::compute({c_state, *input_smem[input_ring], *output_smem[output_ring], inputs_finished[input_ring], outputs_arrived[output_ring], it, unif});
                input_ring=ring_advance<INPUT_PIPE_STAGES>(input_ring);
                output_ring=ring_advance<OUTPUT_PIPE_STAGES>(output_ring);
            } // compute loop
            // ensure the outputs are all written before overwriting that memory
            #pragma unroll
            for(int i = 0; i < OUTPUT_PIPE_STAGES; i++) {
                wait(outputs_finished[i], get_phasebit<1>(semaphore_bitfield, i));
            }
            // no need to update phase bit, as nothing is actually changing before the next wait starts.
            consumers::sync(14); // cannot overwrite finish block until all consumer warps are done.
            lcsft::consumer::finish({c_state, *finish_smem, finish_finished, unif});
            consumers::sync(14); // cannot overwrite finish block until all consumer warps are done.
        } // task iter loop
    } // consumer warpgroup
}

} // namespace lcsf
} // namespace prototype
} // namespace kittens