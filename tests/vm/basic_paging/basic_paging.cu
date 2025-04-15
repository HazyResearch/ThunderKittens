#define RED_TEXT "\033[31m"
#define GREEN_TEXT "\033[32m"
#define YELLOW_TEXT "\033[33m"
#define BLUE_TEXT "\033[34m"
#define MAGENTA_TEXT "\033[35m"
#define CYAN_TEXT "\033[36m"
#define WHITE_TEXT "\033[37m"
#define RESET_TEXT "\033[0m"

#include "kittens.cuh"
// #define KVM_DEBUG
#include "vm/vm.cuh"
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
    static constexpr int PAGE_REQUESTS = 48;
    static constexpr int opcode = 1;
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) {}
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {}
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            s.record(0);
            int pages_received[PAGE_REQUESTS];
            for(int i = 0; i < PAGE_REQUESTS; i++) {
                pages_received[i] = s.get_page();
                // printf(BLUE_TEXT "Loader received page %d for iteration %d\n" RESET_TEXT, pages_received[i], i);
                s.record(16+i);
                if(laneid() == 0) {
                    // printf(BLUE_TEXT "Loader setting page arrived for page %d\n" RESET_TEXT, pages_received[i]);
                    // __nanosleep(10000);
                    kittens::arrive(s.page_arrived[pages_received[i]], config::NUM_CONSUMER_WARPS);
                }
                warp::sync();
            }
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            for(int i = 0; i < PAGE_REQUESTS; i++) {
                int page_id = s.get_page();
                s.record(72+i);
                s.wait_page_arrived(page_id);
                if(threadIdx.x == 0) {
                    // printf("Consumer setting page finished for page %d\n", page_id);
                    // Let's simulate reserving the first 8 pages as persistent scratch.
                    if(page_id >= 8) {
                        printf(RED_TEXT "Consumer setting page finished for page %d\n" RESET_TEXT, page_id);
                        kittens::arrive(s.page_finished[page_id], config::NUM_CONSUMER_WARPS);
                    }
                }
                warp::sync();
            }
        }
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

    // Copy timing data back to host
    int h_timing[config::TIMING_WIDTH];
    cudaMemcpy(h_timing, d_timing, config::TIMING_WIDTH * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print all timing data
    std::cout << "Timing data:" << std::endl;
    for (int i = 0; i < config::TIMING_WIDTH; i++) {
        std::cout << "timing[" << i << "] = " << h_timing[i] << std::endl;
    }

    // Clean up
    cudaFree(d_instruction);
    cudaFree(d_timing);

    std::cout << "Test passed!" << std::endl;

    return 0;
}