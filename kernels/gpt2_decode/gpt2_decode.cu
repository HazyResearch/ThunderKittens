#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::interpreter;

static constexpr int SEQ_LEN = 128;
static constexpr int DIM = 768;
static constexpr float EPS = 1e-5;
using head_vec   = sv_bf<DIM>; 

// GEMM
using  base_tile      = st_bf<64, 64>;
using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile, head_vec>;

enum OpCode{
    STOP,
    FIRST_NORM,
    QKV,
    ATTENTION,
    PROJECTION,
    SECOND_NORM,
    FF_EXPAND,
    FF_CONTRACT
};

struct config {
    struct globals {
        
        using instructions_global = kittens::gl<int, 1, -1, -1, 4>;
        instructions_global instructions;

        // Layer input given as hidden + residual
        global_layout input_hidden;
        global_layout input_residual;

        // Weights and output of the first layer norm (sum of hidden + residual and normalized output)
        global_layout gamma_first_norm;
        global_layout beta_first_norm;
        global_layout mid_residual;
        global_layout mid_first_norm;
        
        // Weights and output of the QKV matmul
        global_layout weight_qkv;
        global_layout bias_qkv;
        global_layout mid_qkv;

        // Output of the attention
        global_layout mid_attn;

        // Weights and output of the projection matmul
        global_layout weight_proj;
        global_layout bias_proj;
        global_layout mid_proj;

        // Weights and output (hidden + residual) of the second layer norm
        global_layout gamma_second_norm;
        global_layout beta_second_norm;
        global_layout output_residual;
        global_layout mid_second_norm;

        // Weight and output of the first feed-forward matmul
        global_layout weight_ff_expand;
        global_layout bias_ff_expand;
        global_layout mid_ff_expand;

        // Weight and output of the second feed-forward matmul
        global_layout weight_ff_contract;
        global_layout bias_ff_contract;
        global_layout output_hidden;

        int dynamic_shared_memory() { return 226000; }
        dim3 grid()  { return dim3(132); }
        dim3 block() { return dim3((NUM_CONSUMER_WARPS + NUM_PRODUCER_WARPS) * WARP_THREADS); }
    };
};

/**
    Given tensors A, B, gamma, beta, computes C = A + B and D = norm(C) * gamma + beta
 */
template <
    OpCode _opcode, 
    global_layout config::globals::* AMember, 
    global_layout config::globals::* BMember, 
    global_layout config::globals::* GammaMember, 
    global_layout config::globals::* BetaMember, 
    global_layout config::globals::* CMember, 
    global_layout config::globals::* DMember
>
struct layernorm_template {
    using config = config;
    static constexpr int opcode = _opcode;

    struct layout {
        using globals = config::globals;
        struct input_block { head_vec heads[2][NUM_CONSUMER_WARPS]; };
        struct output_block { head_vec heads[2][NUM_CONSUMER_WARPS]; };
        struct consumer_state { rv_fl<DIM> gamma, beta; };
    };

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        args.num_iters = SEQ_LEN / NUM_CONSUMER_WARPS;
        static_assert(SEQ_LEN % NUM_CONSUMER_WARPS == 0, "Assumes SEQ_LEN % NUM_CONSUMER_WARPS == 0");
    }

    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == (args.iter % 4)){

                tma::expect_bytes(args.inputs_arrived, sizeof(head_vec) * NUM_CONSUMER_WARPS * 2);
                for(int i = 0; i < NUM_CONSUMER_WARPS; i++) {
                    tma::load_async(args.input.heads[0][i], args.globals.*AMember, {args.iter * NUM_CONSUMER_WARPS + i, 0}, args.inputs_arrived);
                    tma::load_async(args.input.heads[1][i], args.globals.*BMember, {args.iter * NUM_CONSUMER_WARPS + i, 0}, args.inputs_arrived);
                }
                
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
                __syncwarp();

            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == (args.iter % 4)){

                for(int i = 0; i < NUM_CONSUMER_WARPS; i++) {
                    tma::store_async(args.globals.*CMember, args.output.heads[0][i], {args.iter * NUM_CONSUMER_WARPS + i, 0});
                    tma::store_async(args.globals.*DMember, args.output.heads[1][i], {args.iter * NUM_CONSUMER_WARPS + i, 0});
                }
                tma::store_async_read_wait();
                if(laneid() == 0) arrive(args.outputs_finished, 4);
                __syncwarp();

            }

        }
    };

    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            kittens::load(args.state.gamma, args.globals.*GammaMember, {0});
            kittens::load(args.state.beta, args.globals.*BetaMember, {0});
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {

            rv_fl<DIM> head, res;
            kittens::load(head, args.input.heads[0][kittens::warpid()]);
            kittens::load(res, args.input.heads[1][kittens::warpid()]);
            __syncwarp();
            if(laneid() == 0) arrive(args.inputs_finished);

            // Add the residual to the hidden
            kittens::add(head, head, res);

            // Store as C
            kittens::store(args.output.heads[0][kittens::warpid()], head);
            __syncwarp();

            // Calculate mean of input vector
            float mean = kittens::sum(head) / float(DIM);

            // Compute variance
            rv_fl<DIM> temp;
            kittens::sub(temp, head, mean);
            kittens::mul(temp, temp, temp);
            float rstd = rsqrt((kittens::sum(temp) / float(DIM)) + EPS);

            kittens::sub(temp, head, mean);
            kittens::mul(head, temp, rstd);

            // Gamma-beta scaling
            kittens::mul(head, head, args.state.gamma);
            kittens::add(head, head, args.state.beta);

            // Store as D
            kittens::store(args.output.heads[1][kittens::warpid()], head);

            __syncwarp();
            if(laneid() == 0) arrive(args.outputs_arrived);

        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };

};

/**
    Given tensors A and B, computes C = A @ B. 
    If BIAS is true, computes C = A @ B + bias.
    If GELU is true, applies GELU activation to C.
 */
template <
    OpCode _opcode, 
    global_layout config::globals::* AMember, 
    global_layout config::globals::* BMember, 
    global_layout config::globals::* CMember, 
    bool GELU = false, bool BIAS = false,
    global_layout config::globals::* BiasMember = CMember // default value just to avoid error, need to specify is using bias
>
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

            if(BIAS){

                rt_fl<16, N_BLOCK*base_tile::cols>::row_vec bias;
                kittens::load(bias, args.globals.*BiasMember, {args.common.coord.y / N_BLOCK});
                add_col(args.state.accum, args.state.accum, bias);

            }

            if(GELU){

                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
                
                rt_fl<16, N_BLOCK*base_tile::cols> temp;

                // Compute x^3
                mul(temp, args.state.accum, args.state.accum);
                mul(temp, temp, args.state.accum);

                // Compute x + 0.044715 * x^3
                mul(temp, temp, 0.044715f);
                add(temp, args.state.accum, temp);

                // Compute sqrt(2 / pi) * (x + 0.044715 * x^3)
                mul(temp, temp, 0.7978845608028654f); // sqrt(2 / pi)

                // Compute tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))
                tanh(temp, temp);

                // Compute 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
                add(temp, temp, 1.0f);
                mul(temp, temp, args.state.accum);
                mul(args.state.accum, temp, 0.5f);

            }

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
        args.num_iters = SEQ_LEN / NUM_CONSUMER_WARPS;
        static_assert(SEQ_LEN % NUM_CONSUMER_WARPS == 0, "Assumes SEQ_LEN % NUM_CONSUMER_WARPS == 0");
    }

    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {}
        __device__ static inline void load(producer_load_args<layout> args) {
            if(laneid() == 0) arrive(args.inputs_arrived);
            __syncwarp();
        }
        __device__ static void store(producer_store_args<layout> args) {
            if(laneid() == 0) arrive(args.outputs_finished);
            __syncwarp();
        }
    };

    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {}
        __device__ static inline void compute(consumer_compute_args<layout> args) {

            int my_idx = args.iter * NUM_CONSUMER_WARPS + kittens::warpid();

            for(int head = 0; head < 12; head++){

                rv_fl<64> query;
                kittens::load(query, args.globals.*AMember, {my_idx, head});

                rv_fl<64> output;
                zero(output);

                float curr_max = -1e9f;
                float curr_softmax_sum = 0;

                for(int i = 0; i <= my_idx; i++) {

                    // Compute query-key product
                    rv_fl<64> key;
                    kittens::load(key, args.globals.*AMember, {i, DIM / 64 + head});
                    rv_fl<64> temp;
                    kittens::mul(temp, query, key);
                    float logit = kittens::sum(temp) / sqrtf(float(64));

                    // Update maximum
                    float new_max = max(curr_max, logit);
                    logit = expf(logit - new_max);

                    kittens::mul(output, output, expf(curr_max - new_max));
                    curr_softmax_sum *= expf(curr_max - new_max);
                    
                    // Compute logit-value product
                    rv_fl<64> value;
                    kittens::load(value, args.globals.*AMember, {i, (DIM * 2) / 64 + head});
                    kittens::mul(value, value, logit);
                    kittens::add(output, output, value);

                    // Update softmax sum
                    curr_softmax_sum += logit;

                    curr_max = new_max;
                    
                }

                kittens::mul(output, output, 1.0f / curr_softmax_sum);
                kittens::store(args.globals.*CMember, output, {my_idx, head});

            }
        
            __syncwarp();
            if(laneid() == 0) arrive(args.inputs_finished);

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
        .value("FIRST_NORM", FIRST_NORM)
        .value("QKV", QKV)
        .value("ATTENTION", ATTENTION)
        .value("PROJECTION", PROJECTION)
        .value("SECOND_NORM", SECOND_NORM)
        .value("FF_EXPAND", FF_EXPAND)
        .value("FF_CONTRACT", FF_CONTRACT)
        .export_values();

    kittens::py::bind_kernel<
        interpreter::kernel<config, 
            layernorm_template<OpCode::FIRST_NORM, &config::globals::input_hidden, &config::globals::input_residual, &config::globals::gamma_first_norm, &config::globals::beta_first_norm, &config::globals::mid_residual, &config::globals::mid_first_norm>, 
            matmul_template<OpCode::QKV, &config::globals::mid_first_norm, &config::globals::weight_qkv, &config::globals::mid_qkv, false, true, &config::globals::bias_qkv>,
            attention_template<OpCode::ATTENTION, &config::globals::mid_qkv, &config::globals::mid_attn>,
            matmul_template<OpCode::PROJECTION, &config::globals::mid_attn, &config::globals::weight_proj, &config::globals::mid_proj, false, true, &config::globals::bias_proj>,
            layernorm_template<OpCode::SECOND_NORM, &config::globals::mid_proj, &config::globals::mid_residual, &config::globals::gamma_second_norm, &config::globals::beta_second_norm, &config::globals::output_residual, &config::globals::mid_second_norm>,
            matmul_template<OpCode::FF_EXPAND, &config::globals::mid_second_norm, &config::globals::weight_ff_expand, &config::globals::mid_ff_expand, true, true, &config::globals::bias_ff_expand>,
            matmul_template<OpCode::FF_CONTRACT, &config::globals::mid_ff_expand, &config::globals::weight_ff_contract, &config::globals::output_hidden, false, true, &config::globals::bias_ff_contract>
        >>(m, "gpt2_decode",
            &config::globals::instructions,
            &config::globals::input_hidden,
            &config::globals::input_residual,
            &config::globals::gamma_first_norm,
            &config::globals::beta_first_norm,
            &config::globals::mid_residual,
            &config::globals::mid_first_norm,
            &config::globals::weight_qkv,
            &config::globals::bias_qkv,
            &config::globals::mid_qkv,
            &config::globals::mid_attn,
            &config::globals::weight_proj,
            &config::globals::bias_proj,
            &config::globals::mid_proj,
            &config::globals::gamma_second_norm,
            &config::globals::beta_second_norm,
            &config::globals::output_residual,
            &config::globals::mid_second_norm,
            &config::globals::weight_ff_expand,
            &config::globals::bias_ff_expand,
            &config::globals::mid_ff_expand,
            &config::globals::weight_ff_contract,
            &config::globals::bias_ff_contract,
            &config::globals::output_hidden
    );
}
