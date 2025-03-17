#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

static constexpr int BATCH_SIZE = 256;
static constexpr int DIM = 768;
static constexpr float EPS = 1e-5;
using head_vec   = sv_bf<DIM>; 

// GEMM
using  base_tile      = st_bf<64, 64>;
using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile, head_vec>;

enum OpCode{
    STOP,
    INPUT_NORM,
    QKV,
    ATTENTION,
    PROJECTION
};

struct config {
    struct globals {
        
        using instructions_global = kittens::gl<int, 1, -1, -1, 4>;
        instructions_global instructions;

        global_layout layer_input;
        global_layout after_first_norm;
        
        global_layout qkv_weights;
        global_layout qkv;

        global_layout attn_output;

        global_layout proj_weights;
        global_layout proj_output;

        int dynamic_shared_memory() { return 226000; }
        dim3 grid()  { return dim3(132); }
        dim3 block() { return dim3((NUM_CONSUMER_WARPS + NUM_PRODUCER_WARPS) * WARP_THREADS); }
    };
};

template <OpCode _opcode, global_layout config::globals::* AMember, global_layout config::globals::* CMember>
struct layernorm_template {
    using config = config;
    static constexpr int opcode = _opcode;

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
                    tma::load_async(args.input.heads[i], args.globals.*AMember, {args.iter * NUM_CONSUMER_WARPS + i, 0}, args.inputs_arrived);
                }
                
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
                __syncwarp();

            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == (args.iter % 4)){

                for(int i = 0; i < NUM_CONSUMER_WARPS; i++) {
                    tma::store_async(args.globals.*CMember, args.output.heads[i], {args.iter * NUM_CONSUMER_WARPS + i, 0});
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

template <OpCode _opcode, global_layout config::globals::* AMember, global_layout config::globals::* BMember, global_layout config::globals::* CMember>
struct matmul_template {

    using config = config;
    static constexpr int opcode = _opcode;

    static constexpr int M_BLOCK = 2, N_BLOCK = 4;
    
    struct layout {
        using globals = config::globals;
        struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
        struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
        struct common_state   { int2 coord; };
        struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
    };

    using wide_tile = st_bf<64, 64 * N_BLOCK>;

    __device__ static inline void common_setup(common_setup_args<layout> args) {

        args.num_iters = (args.globals.*AMember).cols() / 64;

        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS / 4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.instruction[1] * M_BLOCK + id, args.instruction[2] * N_BLOCK };

    }

    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == (args.iter % 4)) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.*AMember,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.*BMember,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(
                args.state.accum, // dest registers
                args.input.a[warpgroup::groupid()], // A matrix
                reinterpret_cast<wide_tile&>(args.input.b) // B matrix
            );
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);

            warpgroup::sync(warpgroup::groupid() + 4);
            if(warpgroup::warpid() == 0){ 
                for(int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(args.globals.*CMember, args.finish.c[warpgroup::groupid()][i],
                                                {args.common.coord.x, args.common.coord.y+i});
                    tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
                }
            }
            zero(args.state.accum);
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template <OpCode _opcode, global_layout config::globals::* AMember, global_layout config::globals::* CMember>
struct attention_template {
    using config = config;
    static constexpr int opcode = _opcode;

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
                    tma::load_async(args.input.heads[i], args.globals.*AMember, {args.iter * NUM_CONSUMER_WARPS + i, 0}, args.inputs_arrived);
                }
                
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
                __syncwarp();

            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == (args.iter % 4)){

                for(int i = 0; i < NUM_CONSUMER_WARPS; i++) {
                    tma::store_async(args.globals.*CMember, args.output.heads[i], {args.iter * NUM_CONSUMER_WARPS + i, 0});
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

            kittens::store(args.output.heads[kittens::warpid()], head);

            __syncwarp();
            if(laneid() == 0) arrive(args.outputs_arrived);

        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };

};

PYBIND11_MODULE(gpt2_decode, m) {
    m.doc() = "gpt2_decode python module";

    pybind11::enum_<OpCode>(m, "OpCode")
        .value("STOP", STOP)
        .value("INPUT_NORM", INPUT_NORM)
        .value("QKV", QKV)
        .value("ATTENTION", ATTENTION)
        .value("PROJECTION", PROJECTION)
        .export_values();

    kittens::py::bind_kernel<
        interpreter::kernel<config, 
            layernorm_template<OpCode::INPUT_NORM, &config::globals::layer_input, &config::globals::after_first_norm>, 
            matmul_template<OpCode::QKV, &config::globals::after_first_norm, &config::globals::qkv_weights, &config::globals::qkv>,
            attention_template<OpCode::ATTENTION, &config::globals::qkv, &config::globals::attn_output>,
            matmul_template<OpCode::PROJECTION, &config::globals::attn_output, &config::globals::proj_weights, &config::globals::proj_output>
        >>(m, "gpt2_decode",
            &config::globals::instructions,
            &config::globals::layer_input,
            &config::globals::after_first_norm,
            &config::globals::qkv_weights,
            &config::globals::qkv,
            &config::globals::attn_output,
            &config::globals::proj_weights,
            &config::globals::proj_output
    );
}
