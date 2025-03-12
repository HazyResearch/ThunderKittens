#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;
struct config {
    struct globals {
        using instructions_global = kittens::gl<int, 1, -1, -1, 4>;
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
        if(threadIdx.x == 0) printf("block %d running op A (%d)\n", blockIdx.x, opcode);
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
        if(threadIdx.x == 0) printf("block %d running op B (%d)\n", blockIdx.x, opcode);
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
    constexpr int NUM_INSTRUCTIONS = 5;
    int instructions[NUM_INSTRUCTIONS] = {1, 2, 1, 0, 2}; // last 2 should not execute.
    std::vector<int> instructions_vec(132*NUM_INSTRUCTIONS);
    for(int i = 0; i < 132*NUM_INSTRUCTIONS; i++) {
        instructions_vec[i] = instructions[i % NUM_INSTRUCTIONS];
    }
    int *instructions_d;
    cudaMalloc(&instructions_d, sizeof(int) * NUM_INSTRUCTIONS*132);
    cudaMemcpy(instructions_d, instructions_vec.data(), sizeof(int) * NUM_INSTRUCTIONS*132, cudaMemcpyHostToDevice);
    kittens::gl<int, 1, -1, -1, 4> instructions_gl{instructions_d, nullptr, 132, NUM_INSTRUCTIONS, nullptr};
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

// #include "kittens.cuh"
// #include "prototype.cuh"
// #include "pyutils/pyutils.cuh"

// using namespace kittens;
// using namespace kittens::prototype;
// using namespace kittens::prototype::interpreter;

// using instructions_global = kittens::gl<int, 1, -1, -1, 1>;

// struct config {
//     struct globals {
//         using instructions_global = instructions_global;
//         instructions_global instructions;
        
//         int dynamic_shared_memory() { return 226000; }
//         dim3 grid()                 { return dim3(132); }
//         dim3 block()                { return dim3((NUM_CONSUMER_WARPS + NUM_PRODUCER_WARPS) * WARP_THREADS); }
//     };
// };

// struct OpA_layout{
//     using globals = config::globals;
//     struct input_block {};
//     struct scratch_block {};
//     struct finish_block {};
//     struct common_state {};
//     struct consumer_state {};
// };
// struct OpA {
//     using config = config;
//     using layout = OpA_layout;
//     static constexpr int opcode = 1;
//     __device__ static inline void common_setup(common_setup_args<layout> args) {
//         if(threadIdx.x == 0) printf("block %d running op A (%d)\n", blockIdx.x, opcode);
//     }
//     struct producer {
//         __device__ static inline void setup(producer_setup_args<layout> args) {
//             warpgroup::producer_registers();
//         }
//         __device__ static inline void load(producer_load_args<layout> args) {
//             if(laneid() == 0) arrive(args.inputs_arrived);
//             warpgroup::sync(warpgroup::groupid());
//         }
//     };
//     struct consumer {
//         __device__ static inline void setup(consumer_setup_args<layout> args) {
//             warpgroup::consumer_registers<4>();
//         }
//         __device__ static inline void compute(consumer_compute_args<layout> args) {
//             if(laneid() == 0) arrive(args.inputs_finished);
//             warpgroup::sync(warpgroup::groupid());
//         }
//         __device__ static inline void finish(consumer_finish_args<layout> args) {
//             if(laneid() == 0) arrive(args.finish_finished);
//         }
//     };
// };

// int main() {
//     constexpr int NUM_INSTRUCTIONS = 5;
//     int instructions[NUM_INSTRUCTIONS] = {1, 0, 1, 0, 2}; // last 2 should not execute.
//     std::vector<int> instructions_vec(132*NUM_INSTRUCTIONS);
//     for(int i = 0; i < 132*NUM_INSTRUCTIONS; i++) {
//         instructions_vec[i] = instructions[i % NUM_INSTRUCTIONS];
//     }
//     int *instructions_d;
//     cudaMalloc(&instructions_d, sizeof(int) * NUM_INSTRUCTIONS*132);
//     cudaMemcpy(instructions_d, instructions_vec.data(), sizeof(int) * NUM_INSTRUCTIONS*132, cudaMemcpyHostToDevice);

//     kittens::gl<int, 1, -1, -1, 1> instructions_gl{instructions_d, nullptr, 132, NUM_INSTRUCTIONS, nullptr};
//     config::globals G{instructions_gl};
//     kittens::prototype::interpreter::run<config, OpA>(G);
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA error: %s\n", cudaGetErrorString(err));
//         return 1;
//     }
//     cudaDeviceSynchronize();
//     err = cudaGetLastError(); 
//     if (err != cudaSuccess) {
//         printf("CUDA error after synchronize: %s\n", cudaGetErrorString(err));
//         return 1;
//     }
// }