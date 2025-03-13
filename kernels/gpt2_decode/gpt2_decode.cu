#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

static constexpr int BATCH_SIZE = 64;
static constexpr int DIM = 64;
using head_vec   = sv_bf<DIM>; 
using h_global = gl<bf16, 1, 1, BATCH_SIZE, DIM, tma::descriptor<head_vec>>; // B * DIM

struct config {
    struct globals {
        
        using instructions_global = kittens::gl<int, 1, -1, -1, 4>;
        instructions_global instructions;

        h_global H;

        int dynamic_shared_memory() { return 224000; }
        dim3 grid()  { return dim3(1); }
        dim3 block() { return dim3((NUM_CONSUMER_WARPS + NUM_PRODUCER_WARPS) * WARP_THREADS); }
    };
};

struct LayerNorm {
    using config = config;
    static constexpr int opcode = 1;

    struct layout {
        using globals = config::globals;
        struct input_block { head_vec heads[NUM_CONSUMER_WARPS]; };
        struct output_block { head_vec heads[NUM_CONSUMER_WARPS]; };
    };

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        args.num_iters = BATCH_SIZE / NUM_CONSUMER_WARPS;
        static_assert(BATCH_SIZE % NUM_CONSUMER_WARPS == 0, "Assumes BATCH_SIZE % NUM_CONSUMER_WARPS == 0");
    }

    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == (args.iter % 4)){

                tma::expect_bytes(args.inputs_arrived, sizeof(head_vec) * NUM_CONSUMER_WARPS);
                for(int i = 0; i < NUM_CONSUMER_WARPS; i++) {
                    tma::load_async(args.input.heads[i], args.globals.H, {args.iter * NUM_CONSUMER_WARPS + i, 0}, args.inputs_arrived);
                }
                
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
                __syncwarp();

            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == (args.iter % 4)){

                for(int i = 0; i < NUM_CONSUMER_WARPS; i++) {
                    tma::store_async(args.globals.H, args.output.heads[i], {args.iter * NUM_CONSUMER_WARPS + i, 0});
                }
                tma::store_async_read_wait();
                if(laneid() == 0) arrive(args.outputs_finished, 4);
                __syncwarp();

            }

        }
    };

    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {}
        __device__ static inline void compute(consumer_compute_args<layout> args) {

            rv_bf<DIM> head;
            kittens::load(head, args.input.heads[kittens::warpid()]);
            if(laneid() == 0) arrive(args.inputs_finished);
            __syncwarp();

            // Calculate mean of input vector
            bf16 mean = bf16(kittens::sum(head)) / bf16(DIM);

            // Compute variance
            rv_bf<DIM> temp;
            kittens::sub(temp, head, mean);
            kittens::mul(temp, temp, temp);
            bf16 rstd = rsqrt(bf16(kittens::sum(temp)) / bf16(DIM));

            kittens::sub(temp, head, mean);
            kittens::mul(head, temp, rstd);

            kittens::store(args.output.heads[kittens::warpid()], head);

            __syncwarp();
            if(laneid() == 0) arrive(args.outputs_arrived);

        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };

};

__global__ void init_inputs(bf16 *input){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    input[index] = __float2bfloat16(float(index % 70));
}

__global__ void print_outputs(bf16 *input, int n){
    for(int i = 0; i < n; i++){
        printf("%f, ", __bfloat162float(input[i]));
    }
    printf("\n");
}

int main() {

    constexpr int NUM_INSTRUCTIONS = 5;
    int instructions[NUM_INSTRUCTIONS] = {1, 0, 0, 0, 0}; // last 2 should not execute.
    std::vector<int> instructions_vec(132*NUM_INSTRUCTIONS);
    for(int i = 0; i < 132*NUM_INSTRUCTIONS; i++) {
        instructions_vec[i] = instructions[i % NUM_INSTRUCTIONS];
    }
    int *instructions_d;
    cudaMalloc(&instructions_d, sizeof(int) * NUM_INSTRUCTIONS*132);
    cudaMemcpy(instructions_d, instructions_vec.data(), sizeof(int) * NUM_INSTRUCTIONS*132, cudaMemcpyHostToDevice);
    kittens::gl<int, 1, -1, -1, 4> instructions_gl{instructions_d, nullptr, 132, NUM_INSTRUCTIONS, nullptr};

    bf16 *hidden_d;
    cudaMalloc(&hidden_d, sizeof(bf16) * BATCH_SIZE * DIM);
    h_global hidden_gl{hidden_d, nullptr, nullptr, nullptr, nullptr};

    init_inputs<<<BATCH_SIZE, DIM>>>(hidden_d);
    print_outputs<<<1, 1>>>(hidden_d + 62 * DIM, DIM);

    config::globals G{instructions_gl, hidden_gl};
    kittens::prototype::interpreter::run<config, LayerNorm>(G);

    print_outputs<<<1, 1>>>(hidden_d + 62 * DIM, DIM);

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
