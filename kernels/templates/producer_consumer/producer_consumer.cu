#include "kittens.cuh"
using namespace kittens;

constexpr int NUM_CONSUMER_WARPGROUPS = 3; // 2 and 3 are likely cases for real kernels.
static_assert(NUM_CONSUMER_WARPGROUPS >= 2 && NUM_CONSUMER_WARPGROUPS <= 6); // The register alloc is only set up for this range.
constexpr int NUM_CONSUMER_WARPS = NUM_CONSUMER_WARPGROUPS * WARPGROUP_WARPS;
constexpr int NUM_WARPS = NUM_CONSUMER_WARPS + WARPGROUP_WARPS; // producers, too
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

// This is a producer+consumer copy kernel that demonstrates the use of TMA to implement a two-stage pipeline.
__global__ __launch_bounds__(NUM_THREADS, 1)
void producer_consumer_template(int n_blocks, const CUtensorMap* example_input_global, CUtensorMap* example_output_global)  {

    int laneid = kittens::laneid(), warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
    int tic = 0, toc = 1; // these are used to track the two-stage pipeline.

    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    st_fl_4x4 (&example_input_smem) [2] = alloc.allocate<st_fl_4x4, 2>();
    st_fl_4x4 (&example_output_smem)[2] = alloc.allocate<st_fl_4x4, 2>();

    // Initialize barriers
    __shared__ kittens::barrier producer_arrived[2], consumer_arrived[2];
    if (warpid == 0) {
        init_barrier(producer_arrived[0], 0, 1); // needs to wait on just one memory transaction, each
        init_barrier(producer_arrived[1], 0, 1);
        init_barrier(consumer_arrived[0], NUM_CONSUMER_WARPS, 0); // needs to wait on one thread from each consumer warp
        init_barrier(consumer_arrived[1], NUM_CONSUMER_WARPS, 0);
    }
    // Launch first load. No sync needed since thread 0 is doing these, too.
    if(warpid == 0) {
        tma::expect<st_fl_4x4>(producer_arrived[0]); // register a transaction of a full st_fl_4x4.
        int block_idx = 0;
        tma::load_async(example_input_smem[tic], example_input_global, producer_arrived[0], block_idx); // launch the initial load
    }

    __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

    if(warpgroupid == NUM_CONSUMER_WARPGROUPS) { // last warpgroup is a producer
        warpgroup::decrease_registers<24>();
   
        if(warpid == NUM_CONSUMER_WARPS) { // just need a single warp to do this
            for (int block_idx = 1; block_idx < n_blocks; block_idx++, tic=tic^1, toc=toc^1) {
                tma::expect<st_fl_4x4>(producer_arrived[toc]); // register that another block is coming in
                tma::load_async(example_input_smem[toc], example_input_global, producer_arrived[toc], block_idx); // load that block
                wait(consumer_arrived[tic], ((block_idx-1)/2)%2); // phase changes at half the rate of the tic/toc
            }
        }
    }
    else { // other warpgroups are consumers
        constexpr int n_reg = NUM_CONSUMER_WARPGROUPS <= 3 ? 480/NUM_CONSUMER_WARPGROUPS : 480/NUM_CONSUMER_WARPGROUPS - 8;
        warpgroup::increase_registers<n_reg>(); // valid up to 6 consumer warpgroups, and I can't imagine wanting more

        for (int block_idx = 0; block_idx < n_blocks; block_idx++, tic^=1, toc^=1) {
            
            wait(producer_arrived[tic], (block_idx/2)%2); // wait for memory to arrive

            // do work. in this case, we'll just have the first consumer warpgroup copy the input to the output

            tma::store_async_read_wait(); // this will cause thread 1 to wait for previous stores to complete
            warpgroup::sync(); // no thread may start to store until thread 1 confirms previous store has completed.
            if(warpgroupid == 0) {
                rt_fl_1x4 tmp;
                warpgroup::load(tmp, example_input_smem[tic]);
                warpgroup::store(example_output_smem[tic], tmp);
            }
            warpgroup::sync(); // writes to shared memory are now visible

            if(laneid == 0) arrive(consumer_arrived[tic]); // work is complete, signal to the producer that it may start the next load.

            // This particular example has consumers launch an async store to global memory, which is usually the right approach from our experience.
            // Sometimes it does make sense have the producer do the store (such as on pre-Hopper GPUs). In those cases, additional barrier signals
            // will be necessary to ensure the store memory is synchronized, too.

            if(warpid == 0) { // first warp stores
                tma::store_async(example_output_global, example_output_smem[tic], block_idx);
                tma::store_commit_group();
            }
        }
    }
}

#include <iostream>
int main() {
    // Create arrays
    float* input = new float[64*64 * 100];
    float* output = new float[64*64 * 100];

    for(int i = 0; i < 64*64 * 100; i++) { // Initialize memory with something memorable.
        input[i] = float(i);
    }

    // Create CUDA tensors
    float *d_input, *d_output;
    cudaMalloc(&d_input, 64*64 * 100 * sizeof(float));
    cudaMalloc(&d_output, 64*64 * 100 * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_input, input, 64*64 * 100 * sizeof(float), cudaMemcpyHostToDevice);

    // Create tma descriptors on both
    CUtensorMap* input_tma  = tma::allocate_and_create_tensor_map<st_fl_4x4>(d_input, 100);
    CUtensorMap* output_tma = tma::allocate_and_create_tensor_map<st_fl_4x4>(d_output, 100);

    // Launch kernel
    cudaFuncSetAttribute(producer_consumer_template, cudaFuncAttributeMaxDynamicSharedMemorySize, 100000);
    producer_consumer_template<<<1, NUM_THREADS, 100000>>>(100, input_tma, output_tma);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy data from GPU
    cudaMemcpy(output, d_output, 64*64 * 100 * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    for(int i = 0; i < 64*64 * 100; i++) {
        if(output[i] != input[i]) {
            std::cout << "Failed at index " << i << " with input " << input[i] << " and output " << output[i] << std::endl;
            return 1;
        }
    }

    std::cout << "Passed" << std::endl;
    return 0;
}