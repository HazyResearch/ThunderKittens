#include "kittens.cuh"
// #define KVM_DEBUG
#include "vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    instruction_layout instructions;
    timing_layout timings;
};

template<typename config=config> struct TestOp {
    static constexpr int opcode = 1;
    static __device__ inline int num_pages(const globals &g, state<config> &s) { return 0; }
    static __device__ inline int num_mini_pages(const globals &g, state<config> &s) { return 0; }
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) {}
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {}
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {}
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {}
    };
};

int main() {

    vm::print_config<config>();

    // Initialize a vector of one 1 and 31 0's
    int instruction[config::INSTRUCTION_WIDTH] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Create a device array for the instruction
    int *d_instruction;
    cudaMalloc(&d_instruction, config::INSTRUCTION_WIDTH * sizeof(int));
    cudaMemcpy(d_instruction, instruction, config::INSTRUCTION_WIDTH * sizeof(int), cudaMemcpyHostToDevice);

    // Create a device array for timing data
    int *d_timing;
    cudaMalloc(&d_timing, config::TIMING_WIDTH * sizeof(int));
    cudaMemset(d_timing, 0, config::TIMING_WIDTH * sizeof(int));
    
    // Use the device array
    typename globals::instruction_layout instructions{d_instruction, nullptr, 1, 1, nullptr};
    typename globals::timing_layout timings{d_timing, nullptr, 1, 1, nullptr};
    globals g{instructions, timings};
    ::kittens::prototype::vm::kernel<config, globals, TestOp<config>><<<1, config::NUM_THREADS>>>(g);
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    // Synchronize device to ensure all operations are complete
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "CUDA synchronize error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    // Clean up
    cudaFree(d_instruction);
    cudaFree(d_timing);

    return 0;
}