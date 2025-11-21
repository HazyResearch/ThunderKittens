/*
    This kernel leverages one of ThunderKittens' pipeline templates (load-compute-finish) to concisely implement a 
    two-stage pipeline with minimal boilerplate. TK pipeline templates are also used in our matrix multiply, rotary, 
    mamba, and fftconv kernels -- they're pretty flexible. Note that this is NOT the kernel benchmarked in the paper.
*/
#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int D, int NUM_WORKERS> struct attn_fwd_layout {
    using qo_tile   = st_bf<64, D>;
    using kv_tile   = st_bf<D==64?192:128, D>;
    using qo_global = kittens::gl<bf16, -1, -1, -1, D, qo_tile>;
    using kv_global = kittens::gl<bf16, -1, -1, -1, D, kv_tile>;
    struct globals { qo_global O, Q; kv_global K, V; };
    struct input_block    { kv_tile k, v; };
    struct scratch_block  { qo_tile q[NUM_WORKERS]; };
    struct common_state   { int batch, head, seq; };
    struct consumer_state {
        rt_fl<16, qo_tile::cols> o_reg;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec, norm_vec;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec_last_scaled, max_vec_scaled;
        rt_fl<16, kv_tile::rows> att_block;
        rt_bf<16, kv_tile::rows> att_block_mma;
    };
};
template<int D> struct attn_fwd_template {
    static constexpr int NUM_CONSUMER_WARPS = 12, NUM_WORKERS = NUM_CONSUMER_WARPS/4, INPUT_PIPE_STAGES = 2;
    using layout = attn_fwd_layout<D, NUM_WORKERS>;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int task_id = gridDim.x*args.task_iter + blockIdx.x;
        int seq_q = (args.globals.Q.rows() + NUM_WORKERS*layout::qo_tile::rows - 1)/(NUM_WORKERS*layout::qo_tile::rows);
        args.common.batch = task_id / (seq_q*args.globals.K.depth()); task_id -= args.common.batch * seq_q * args.globals.K.depth();
        args.common.head  = task_id / seq_q;                        task_id -= args.common.head  * seq_q;
        args.common.seq   = task_id;
        args.num_iters = args.common.batch < args.globals.Q.batch() ? (args.globals.K.rows() + layout::kv_tile::rows - 1)/(layout::kv_tile::rows) : -1;
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                warp::tma::expect(args.inputs_arrived, args.input);
                warp::tma::load_async(args.input.k, args.globals.K, {args.common.batch, args.common.head, args.iter, 0}, args.inputs_arrived);
                tma::load_async(args.input.v, args.globals.V, {args.common.batch, args.common.head, args.iter, 0}, args.inputs_arrived);
            }
            else if(laneid() == 0) arrive(args.inputs_arrived);
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_WORKERS>();
            if((args.common.seq*NUM_WORKERS + warpgroup::groupid())*layout::qo_tile::rows < args.globals.Q.rows()) // out of bounds?
                warpgroup::load(args.scratch.q[warpgroup::groupid()], args.globals.Q,
                                {args.common.batch, args.common.head, args.common.seq*NUM_WORKERS+warpgroup::groupid(), 0});
            args.state.o_reg = 0.f;
            args.state.norm_vec = 0.f;
            args.state.max_vec = base_types::constants<float>::neg_infty();
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;
            // A = Q @ K.T
            warpgroup::mm<transpose::N, transpose::T>(args.state.att_block, args.scratch.q[warpgroup::groupid()], args.input.k);
            args.state.max_vec_last_scaled = args.state.max_vec * TEMPERATURE_SCALE;
            warpgroup::mma_async_wait();
            // softmax
            warp::right_fill(args.state.att_block, args.state.att_block, args.globals.K.rows() - args.iter*layout::kv_tile::rows, base_types::constants<float>::neg_infty());
            args.state.max_vec = warp::max<axis::COL>(args.state.att_block, args.state.max_vec); // accumulate onto the max_vec
            args.state.max_vec_scaled = args.state.max_vec * TEMPERATURE_SCALE;
            args.state.att_block = warp::exp2((args.state.att_block*TEMPERATURE_SCALE) - args.state.max_vec_scaled);
            args.state.max_vec_last_scaled = warp::exp2(args.state.max_vec_last_scaled - args.state.max_vec_scaled);
            args.state.norm_vec *= args.state.max_vec_last_scaled;
            args.state.norm_vec = warp::sum<axis::COL>(args.state.att_block, args.state.norm_vec); // accumulate onto the norm_vec
            args.state.o_reg *= args.state.max_vec_last_scaled; // normalize o_reg before mma
            args.state.att_block_mma = args.state.att_block; // convert to bf16 for mma
            // O += A @ V
            warpgroup::mma<transpose::N, transpose::N>(args.state.o_reg, args.state.att_block_mma, args.input.v);
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished); // done!
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if((args.common.seq*NUM_WORKERS+warpgroup::groupid())*64 < args.globals.Q.rows()) { // out of bounds?
                args.state.o_reg /= args.state.norm_vec;
                auto &o_smem = reinterpret_cast<typename layout::qo_tile&>(args.scratch.q[warpgroup::groupid()]);
                warpgroup::store(o_smem, args.state.o_reg);
                warpgroup::sync(warpgroup::groupid());
                if(warpgroup::warpid() == 0)
                    warp::tma::store_async(args.globals.O, o_smem, {args.common.batch, args.common.head, args.common.seq*NUM_WORKERS+warpgroup::groupid(), 0});
                warp::tma::store_async_read_wait();
            }
            __syncwarp();
            if(laneid() == 0) arrive(args.finish_finished); // done!
        }
    };
};

#include <iostream>
#include <string>
#include <fstream>

constexpr int ATTN_B = 256;
constexpr int ATTN_H = 1;
constexpr int ATTN_N = 924; // 768*2; // 4096;
constexpr int ATTN_D = 128; // hardcoded into this kernel
constexpr int ITER   = 10;

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line ) {
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}

// Compute FLOPs for forward attention
constexpr uint64_t ATTN_FLOPS = 
    2llu * ATTN_B * ATTN_H * ATTN_N * ATTN_N * ATTN_D + // Q * K^T: 2BHNND (multiply-add)
    4llu * ATTN_B * ATTN_H * ATTN_N * ATTN_N +          // Softmax: 2BHNN (exp and divide, plus flash-attn bookkeeping)
    2llu * ATTN_B * ATTN_H * ATTN_N * ATTN_N * ATTN_D;      // (Q * K^T) * V: 2BHNND (multiply-add)

int main(int argc, char **argv) {
    // TODO: consider doing sequential kernel launches to force batches dimension element to execute sequentially,
    // which may increase the probability of L2 cache hits on KV
    using ker_template = attn_fwd_template<ATTN_D>;

    std::cout << "Entered main!" << std::endl;

    // create dummy variables that are the right size
    constexpr int TOTAL_ELEMENTS = ATTN_B*ATTN_H*ATTN_N*ATTN_D;
    constexpr int TOTAL_UNIQUE_ELEMENTS = ATTN_N*ATTN_D*ATTN_H;

    float *q = new float[TOTAL_ELEMENTS];
    float *k = new float[TOTAL_ELEMENTS];
    float *v = new float[TOTAL_ELEMENTS];
    float *o_ref = new float[TOTAL_ELEMENTS];

    bf16 *q_bf = new bf16[TOTAL_ELEMENTS];
    bf16 *k_bf = new bf16[TOTAL_ELEMENTS];
    bf16 *v_bf = new bf16[TOTAL_ELEMENTS];
    bf16 *o_bf = new bf16[TOTAL_ELEMENTS];
    float *o = new float[TOTAL_ELEMENTS];

    std::ifstream infile(argv[1]);

    std::cout << "Starting to enter!" << std::endl;

    for(int i = 0; i < TOTAL_ELEMENTS/ATTN_B; i++) infile >> q[i];
    std::cout << "Finished loading Q" << std::endl;
    for(int i = 0; i < TOTAL_ELEMENTS/ATTN_B; i++) infile >> k[i];
    std::cout << "Finished loading K" << std::endl;
    for(int i = 0; i < TOTAL_ELEMENTS/ATTN_B; i++) infile >> v[i];
    std::cout << "Finished loading V" << std::endl;
    for(int i = 0; i < TOTAL_ELEMENTS/ATTN_B; i++) infile >> o_ref[i];
    std::cout << "Finished loading O_REF" << std::endl;

    std::cout << "Finished loading file from " << argv[1] << "!" << std::endl;

    // replicate into batch elements
    for(int i = 0; i < TOTAL_ELEMENTS; i++) {
        q_bf[i] = __float2bfloat16(q[i % (TOTAL_ELEMENTS/ATTN_B)]);
        k_bf[i] = __float2bfloat16(k[i % (TOTAL_ELEMENTS/ATTN_B)]);
        v_bf[i] = __float2bfloat16(v[i % (TOTAL_ELEMENTS/ATTN_B)]);
    }

    bf16 *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, TOTAL_ELEMENTS * sizeof(bf16));
    cudaMalloc(&d_k, TOTAL_ELEMENTS * sizeof(bf16));
    cudaMalloc(&d_v, TOTAL_ELEMENTS * sizeof(bf16));
    cudaMalloc(&d_o, TOTAL_ELEMENTS * sizeof(bf16));

    cudaMemcpy(d_q, q_bf, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k_bf, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v_bf, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);

    ker_template::layout::qo_global Qg(d_q, ATTN_B, ATTN_H, ATTN_N, nullptr);
    ker_template::layout::kv_global Kg(d_k, ATTN_B, ATTN_H, ATTN_N, nullptr);
    ker_template::layout::kv_global Vg(d_v, ATTN_B, ATTN_H, ATTN_N, nullptr);
    ker_template::layout::qo_global Og(d_o, ATTN_B, ATTN_H, ATTN_N, nullptr);
    ker_template::layout::globals globals = {Og, Qg, Kg, Vg};
    
    unsigned long mem_size = kittens::MAX_SHARED_MEMORY - 2000; // have the flag tell us
    
    cudaFuncSetAttribute(
        prototype::lcf::kernel<ker_template>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    cudaDeviceSynchronize();
    std::cout << "Starting kernel\n";
    constexpr int NUM_WORKERS = prototype::detail::NUM_CONSUMER_WARPGROUPS_v<ker_template>;
    constexpr int BLOCK_SIZE = prototype::detail::NUM_THREADS_v<ker_template>;
    dim3 grid(132, 1, 1);
    // dim3 bad_grid(grid.z, grid.y, grid.x);
    std::cout << "Grid size: " << grid.x << " x " << grid.y << " x " << grid.z << std::endl;
    // warmup
    for(int j = 0; j < ITER; j++)
        prototype::lcf::kernel<ker_template><<<grid, BLOCK_SIZE, mem_size>>>(globals);
    cudaDeviceSynchronize();
    
    const auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITER; i++) {
        prototype::lcf::kernel<ker_template><<<grid, BLOCK_SIZE, mem_size>>>(globals);
    }
    cudaDeviceSynchronize();
    const auto finish = std::chrono::high_resolution_clock::now();
    CudaCheckError();
    std::cout << "Finished kernel\n";
    
    // check correctness
    cudaMemcpy(o_bf, d_o, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyDeviceToHost);
    for(int i = 0; i < TOTAL_ELEMENTS; i++) {
        o[i] = __bfloat162float(o_bf[i]);
    }

    bool good = true;
    std::ofstream o_ref_file("printouts/o_ref.txt");
    std::ofstream o_file("printouts/o.txt");
    std::ofstream diff_file("printouts/diff.txt");

    float total_diff = 0;
    float max_error = 0; 

    for(int i = 0; i < TOTAL_ELEMENTS; i++) {
        float diff = o[i] - o_ref[i % (TOTAL_ELEMENTS/ATTN_B)];

        if (i < TOTAL_UNIQUE_ELEMENTS) {
            o_ref_file << o_ref[i % (TOTAL_ELEMENTS/ATTN_B)] << ' ';
            o_file << o[i] << ' ';
            diff_file << diff << ' ';
        }
        if (i % ATTN_D == ATTN_D-1) {
            o_ref_file << '\n';
            o_file << '\n';
            diff_file << '\n';
        }
        if(abs(diff) > 0.01 || isnan(diff)) {
            good = false;
        }

        total_diff += abs(diff);
        if (abs(diff) > max_error) {
            max_error = abs(diff);
        }
    }

    // print average difference
    std::cout << "Average o difference: " << total_diff / TOTAL_ELEMENTS << std::endl;
    std::cout << "Max     o difference: " << max_error << std::endl;
    if (abs(total_diff / TOTAL_ELEMENTS) < 1e-3) {
        good = true;
    }

    std::cout << "Average fwd execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / ITER << " us" << std::endl;
    if(good) std::cout << "FWD Correct :)\n";
    else std::cout << "FWD Incorrect :(\n";
    // Compute and print average TFLOPs achieved
    double avg_time_s = (double)(std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()) / (ITER * 1e6);
    double avg_tflops = (ATTN_FLOPS / avg_time_s) / 1e12;
    std::cout << "Efficiency: " << avg_tflops << " TFLOPS\n\n\n" << std::endl;

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);

    delete[] q, k, v, o, o_ref;
    delete[] q_bf, k_bf, v_bf, o_bf;

    return 0;
}
