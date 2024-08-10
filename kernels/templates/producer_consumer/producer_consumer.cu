#include "kittens.cuh"
using namespace kittens;

template<int _NUM_CONSUMER_WARPGROUPS>
struct producer_consumer_parameters {
    static constexpr int NUM_CONSUMER_WARPGROUPS = _NUM_CONSUMER_WARPGROUPS;
    static_assert(NUM_CONSUMER_WARPGROUPS >= 2 && NUM_CONSUMER_WARPGROUPS <= 6); // The register alloc is only set up for this range.
    static constexpr int NUM_CONSUMER_WARPS      = NUM_CONSUMER_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_WARPS               = NUM_CONSUMER_WARPS + WARPGROUP_WARPS; // producers, too
    static constexpr int NUM_THREADS             = NUM_WARPS * WARP_THREADS;
    static constexpr int NUM_PRODUCER_REG        = NUM_CONSUMER_WARPGROUPS == 2 ? 32 : 24;
    static constexpr int NUM_CONSUMER_REG        = NUM_CONSUMER_WARPGROUPS <= 3 ? 480/NUM_CONSUMER_WARPGROUPS : 480/NUM_CONSUMER_WARPGROUPS-8; // valid up to 6 consumer warpgroups
};

struct globals {
    int n_blocks;
    const CUtensorMap* input_global;
    CUtensorMap* output_global;
    __host__ __device__ inline globals(int n_blocks, const CUtensorMap* input_global, CUtensorMap* output_global) :
        n_blocks(n_blocks), input_global(input_global), output_global(output_global) {}
};

struct block { // the chunk of data that the producer and consumer are working on
    st_fl_4x4 &input;
    st_fl_4x4 &output;
    __device__ inline block(st_fl_4x4 &input, st_fl_4x4 &output) : input(input), output(output) {}
};

template<int NUM_CONSUMER_WARPGROUPS>
struct producer_consumer {
    using params = producer_consumer_parameters<NUM_CONSUMER_WARPGROUPS>;

    struct producer {
        struct state {}; // persistent registers
        __device__ static void setup(state &s, globals &g) { // setup and load the first iteration
            warpgroup::decrease_registers<params::NUM_PRODUCER_REG>(); // decrease registers for the producer warpgroup
        }
        __device__ static void load(state &s, block &b, globals &g, kittens::barrier &bar, int iter) { // barrier for the producer to load into
            if(warpgroup::warpid() == 0) {
                tma::expect<st_fl_4x4>(bar);
                tma::load_async(b.input, g.input_global, bar, iter);
            }
        }
        __device__ static void finish(state &s, globals &g) {}
    };

    struct consumer {
        struct state {}; // persistent registers
        __device__ static void setup(state &s, globals &g) { // setup locals for before the first iteration
            warpgroup::increase_registers<params::NUM_CONSUMER_REG>();
        }
        __device__ static void compute(state &s, block &b, globals &g, int iter) {
            // do work. in this case, we'll just have the first consumer warpgroup copy the input to the output, and then launch a tma store to global memory.
            if(warpgroup::groupid() == 0) {
                rt_fl_1x4 tmp;
                warpgroup::load(tmp, b.input);
                warpgroup::store(b.output, tmp);
            }
            tma::store_async_read_wait(); // this will cause thread 1 to wait for previous stores to complete, so that there's never more than one active.
            warpgroup::sync(); // writes to shared memory are now visible
            if(warpid() == 0) { // first warp stores
                tma::store_async(g.output_global, b.output, iter);
                tma::store_commit_group();
            }
        }
        __device__ static void finish(state &s, globals &g) {
            tma::store_async_read_wait(); // this isn't really necessary, but it illustrates the principle.
            warpgroup::sync();
        }
    };
};

// This is a producer+consumer copy kernel that demonstrates the use of TMA to implement a two-stage pipeline.
template<int NUM_CONSUMER_WARPGROUPS>
__global__ __launch_bounds__(producer_consumer<NUM_CONSUMER_WARPGROUPS>::params::NUM_THREADS, 1)
void producer_consumer_template(globals g) {
    using pc = producer_consumer<NUM_CONSUMER_WARPGROUPS>;

    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    st_fl_4x4 (&example_input_smem) [2] = alloc.allocate<st_fl_4x4, 2>();
    st_fl_4x4 (&example_output_smem)[2] = alloc.allocate<st_fl_4x4, 2>();
    block blocks[] = {
        block(example_input_smem[0], example_output_smem[0]),
        block(example_input_smem[1], example_output_smem[1])
    };

    // Initialize barriers. This is constant for all two-stage producer-consumer kernels.
    __shared__ kittens::barrier producer_arrived[2], consumer_arrived[2];
    int tic = 0, toc = 1; // these are used to track the two-stage pipeline.
    if (warpid() == 0) { // a single warp (in fact a single thread) does these.
        init_barrier(producer_arrived[tic], 0, 1); // needs to wait on just one memory transaction, each
        init_barrier(producer_arrived[toc], 0, 1);
        init_barrier(consumer_arrived[tic], pc::params::NUM_CONSUMER_WARPS, 0); // needs to wait on one thread from each consumer warp
        init_barrier(consumer_arrived[toc], pc::params::NUM_CONSUMER_WARPS, 0);
    }

    __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

    if(warpgroup::groupid() == pc::params::NUM_CONSUMER_WARPGROUPS) { // last warpgroup is a producer
        typename pc::producer::state s;
        pc::producer::setup(s, g);
        pc::producer::load(s, blocks[tic], g, producer_arrived[tic], 0); // load initial block
        for (int block_idx = 1; block_idx < g.n_blocks; block_idx++, tic=tic^1, toc=toc^1) {
            pc::producer::load(s, blocks[toc], g, producer_arrived[toc], block_idx);
            wait(consumer_arrived[tic], ((block_idx-1)/2)%2); // phase changes at half the rate of the tic/toc
        }
        pc::producer::finish(s, g);
    }
    else { // other warpgroups are consumers
        typename pc::consumer::state s;
        pc::consumer::setup(s, g);
        for (int block_idx = 0; block_idx < g.n_blocks; block_idx++, tic^=1, toc^=1) {
            wait(producer_arrived[tic], (block_idx/2)%2); // wait for memory to arrive
            pc::consumer::compute(s, blocks[tic], g, block_idx);
            if(laneid() == 0) arrive(consumer_arrived[tic]); // work is complete, signal to the producer that it may start the next load.
        }
        pc::consumer::finish(s, g);
    }
}

// This is just a demo function to call the kernel with some test data and sanity-check.
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
    constexpr int NUM_CONSUMER_WARPGROUPS = 4;
    cudaFuncSetAttribute(producer_consumer_template<NUM_CONSUMER_WARPGROUPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100000);
    std::cout << "Launching kernel" << std::endl;
    producer_consumer_template<NUM_CONSUMER_WARPGROUPS><<<1, producer_consumer_parameters<NUM_CONSUMER_WARPGROUPS>::NUM_THREADS, 100000>>>(
        globals{
            100,
            input_tma,
            output_tma
        }
    );

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