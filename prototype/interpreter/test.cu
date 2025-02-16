#include "interpreter.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;
struct config {
    struct globals {
        using instructions_global = kittens::gl<int, 1, -1, -1, 1>;
        instructions_global instructions;
        int dynamic_shared_memory() { return 224000; }
        dim3 grid()  { return dim3(132); }
        dim3 block() { return dim3((
            kittens::prototype::interpreter::NUM_CONSUMER_WARPS+kittens::prototype::interpreter::NUM_PRODUCER_WARPS)
            * kittens::WARP_THREADS
        ); }
    };
};
template<typename _config> struct OpA {
    using config = _config;
    static constexpr int opcode = 1;
    struct layout {
        using globals = config::globals;
        struct input_block { st_bf<64, 64> tile; };
    };
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        args.num_iters = -1;
        if(threadIdx.x == 0) printf("block %d running op A\n", blockIdx.x);
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            if(laneid() == 0) arrive(args.inputs_arrived);
            warpgroup::sync(warpgroup::groupid());
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<4>();
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            if(laneid() == 0) arrive(args.inputs_finished);
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};
template<typename _config> struct OpB {
    using config = _config;
    static constexpr int opcode = 2;
    struct layout {
        using globals = config::globals;
        struct input_block { st_bf<64, 64> tile; };
    };
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        args.num_iters = -1;
        if(threadIdx.x == 0) printf("block %d running op B\n", blockIdx.x);
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            if(laneid() == 0) arrive(args.inputs_arrived);
            warpgroup::sync(warpgroup::groupid());
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<4>();
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            if(laneid() == 0) arrive(args.inputs_finished);
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};


int main() {
    int instructions[3] = {1, 2, 0};
    int *instructions_d;
    cudaMalloc(&instructions_d, sizeof(int) * 3);
    cudaMemcpy(instructions_d, instructions, sizeof(int) * 3, cudaMemcpyHostToDevice);
    kittens::gl<int, 1, -1, -1, 1> instructions_gl{instructions_d, nullptr, 1, 3, nullptr};
    config::globals G{instructions_gl};
    kittens::prototype::interpreter::run<config, OpA<config>, OpB<config>>(G);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError(); 
    if (err != cudaSuccess) {
        printf("CUDA error after synchronize: %s\n", cudaGetErrorString(err));
        return 1;
    }
}