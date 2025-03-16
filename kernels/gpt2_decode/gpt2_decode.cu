#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

// Layer norm
static constexpr int BATCH_SIZE = 256;
static constexpr int DIM = 256;
static constexpr float EPS = 1e-5;
using head_vec   = sv_bf<DIM>; 
// using h_global = gl<bf16, 1, 1, BATCH_SIZE, DIM, tma::descriptor<head_vec>>; // B * DIM

// GEMM
using  base_tile      = st_bf<64, 64>;
using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile, head_vec>;

struct config {
    struct globals {
        
        using instructions_global = kittens::gl<int, 1, -1, -1, 4>;
        instructions_global instructions;

        global_layout layer_input;
        global_layout after_first_norm;
        
        global_layout qkv_weights;
        global_layout qkv;

        int dynamic_shared_memory() { return 226000; }
        dim3 grid()  { return dim3(132); }
        dim3 block() { return dim3((NUM_CONSUMER_WARPS + NUM_PRODUCER_WARPS) * WARP_THREADS); }
    };
};

struct layernorm_template {
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
                    tma::load_async(args.input.heads[i], args.globals.layer_input, {args.iter * NUM_CONSUMER_WARPS + i, 0}, args.inputs_arrived);
                }
                
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
                __syncwarp();

            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == (args.iter % 4)){

                for(int i = 0; i < NUM_CONSUMER_WARPS; i++) {
                    tma::store_async(args.globals.after_first_norm, args.output.heads[i], {args.iter * NUM_CONSUMER_WARPS + i, 0});
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

            rv_fl<DIM> head;
            kittens::load(head, args.input.heads[kittens::warpid()]);
            __syncwarp();
            if(laneid() == 0) arrive(args.inputs_finished);

            // Calculate mean of input vector
            float mean = kittens::sum(head) / float(DIM);

            // Compute variance
            rv_fl<DIM> temp;
            kittens::sub(temp, head, mean);
            kittens::mul(temp, temp, temp);
            float rstd = rsqrt((kittens::sum(temp) / float(DIM)) + EPS);

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

struct qkv_matmul_template {

    using config = config;
    static constexpr int opcode = 2;

    static constexpr int M_BLOCK = 2, N_BLOCK = 4;
    
    struct layout {
        using globals = config::globals;
        struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
        struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
        struct common_state   { int2 coord; };
        struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
    };

    using wide_tile = st_bf<64, 64*N_BLOCK>;

    __device__ static inline void common_setup(common_setup_args<layout> args) {

        args.num_iters = args.globals.after_first_norm.cols() / 64;

        if(kittens::laneid() == 0) printf("%d %d In Common Setup %d\n", blockIdx.x, threadIdx.x, args.num_iters);

        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS / 4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.instruction[1] * M_BLOCK + id, args.instruction[2] * N_BLOCK };

    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::laneid() == 0) printf("[%d %d] In producer load\n", blockIdx.x, threadIdx.x);
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.after_first_norm,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.qkv_weights,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
            }
            if(warpgroup::laneid() == 0) printf("[%d %d] Finished producer load\n", blockIdx.x, threadIdx.x);
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            if(warpgroup::laneid() == 0) printf("[%d %d] In consumer compute\n", blockIdx.x, threadIdx.x);
            warpgroup::mma_AB(
                args.state.accum, // dest registers
                args.input.a[warpgroup::groupid()], // A matrix
                reinterpret_cast<wide_tile&>(args.input.b) // B matrix
            );
            if(warpgroup::laneid() == 0) printf("[%d %d] Into consumer compute\n", blockIdx.x, threadIdx.x);
            warpgroup::mma_async_wait();
            if(warpgroup::laneid() == 0) printf("[%d %d] Intor consumer compute\n", blockIdx.x, threadIdx.x);
            // printf("Compute %f\n", (args.state.accum.tiles[0][0].data[0].x));
            if(laneid() == 0) arrive(args.inputs_finished);
            if(warpgroup::laneid() == 0) printf("[%d %d] Intorr consumer compute\n", blockIdx.x, threadIdx.x);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            if(warpgroup::laneid() == 0) printf("[%d %d] Starting finish\n", blockIdx.x, threadIdx.x);
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            if(warpgroup::laneid() == 0) printf("[%d %d] Startingg finish\n", blockIdx.x, threadIdx.x);
            // printf("Writing %f\n", (args.state.accum.tiles[0][0].data[0].x));

            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::laneid() == 0) printf("[%d %d] Startinggg finish\n", blockIdx.x, threadIdx.x);
            if(warpgroup::warpid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                tma::store_async(args.globals.qkv, args.finish.c[warpgroup::groupid()][i],
                                             {args.common.coord.x, args.common.coord.y+i});
                tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
            }
            if(warpgroup::laneid() == 0) printf("[%d %d] Startingggg finish\n", blockIdx.x, threadIdx.x);
            zero(args.state.accum);
            if(laneid() == 0) arrive(args.finish_finished);
            if(warpgroup::laneid() == 0) printf("[%d %d] Startinggggg finish\n", blockIdx.x, threadIdx.x);
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

PYBIND11_MODULE(gpt2_decode, m) {
    m.doc() = "gpt2_decode python module";
    kittens::py::bind_kernel<interpreter::kernel<config, layernorm_template, qkv_matmul_template>>(m, "gpt2_decode",
        &config::globals::instructions,
        &config::globals::layer_input,
        &config::globals::after_first_norm,
        &config::globals::qkv_weights,
        &config::globals::qkv
    );
}

// int main() {

//     constexpr int NUM_INSTRUCTIONS = 5;
//     int instructions[NUM_INSTRUCTIONS] = {1, 0, 0, 0, 0}; // last 2 should not execute.
//     std::vector<int> instructions_vec(132*NUM_INSTRUCTIONS);
//     for(int i = 0; i < 132*NUM_INSTRUCTIONS; i++) {
//         instructions_vec[i] = instructions[i % NUM_INSTRUCTIONS];
//     }
//     int *instructions_d;
//     cudaMalloc(&instructions_d, sizeof(int) * NUM_INSTRUCTIONS*132);
//     cudaMemcpy(instructions_d, instructions_vec.data(), sizeof(int) * NUM_INSTRUCTIONS*132, cudaMemcpyHostToDevice);
//     kittens::gl<int, 1, -1, -1, 4> instructions_gl{instructions_d, nullptr, 132, NUM_INSTRUCTIONS, nullptr};

//     bf16 *hidden_d;
//     cudaMalloc(&hidden_d, sizeof(bf16) * BATCH_SIZE * DIM);
//     h_global hidden_gl{hidden_d, nullptr, nullptr, nullptr, nullptr};

//     init_inputs<<<BATCH_SIZE, DIM>>>(hidden_d);
//     print_outputs<<<1, 1>>>(hidden_d + 62 * DIM, DIM);

//     config::globals G{instructions_gl, hidden_gl};
//     kittens::prototype::interpreter::run<config, LayerNorm>(G);

//     print_outputs<<<1, 1>>>(hidden_d + 62 * DIM, DIM);

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
