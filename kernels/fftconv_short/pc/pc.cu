#ifdef TORCH_COMPILE
#define TORCH_COMPILE_FFTCONV
#endif

#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
template<int _wg> struct fftconv_256_layout {
    static constexpr int wg = _wg;
    using seq_tile      = st_bf<64, 64>;
    using seq_layout    =     gl<bf16, -1, -1, 16, 16>;
    using filter_layout = cgl<gl<bf16,  1, -1, 64, 64>>;
    using fft_layout    = cgl<gl<bf16,  1,  1, 64, 64>>;
    struct globals {
        seq_layout o, x;
        filter_layout kf;
        fft_layout f, finv, tw, twinv_t;
    };
    struct input_block    { seq_tile x[wg]; };
    struct output_block   { seq_tile o[wg]; };
    struct scratch_block  {
        cst_bf<64, 64> kf, f, finv, tw, twinv_t, tmp[2];
    };
    struct consumer_state { int current_head; };
};
struct fft_256_template {
    static constexpr int NUM_CONSUMER_WARPS=8, NUM_CONSUMER_WARPGROUPS=NUM_CONSUMER_WARPS/4, NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=3, INPUT_PIPE_STAGES=3;
    using layout = fftconv_256_layout<NUM_CONSUMER_WARPGROUPS>;
    // mine
    __device__ static inline void load_head_data(typename layout::scratch_block &scratch, const layout::globals &g, int head) {
        using consumers = group<NUM_CONSUMER_WARPS>;
        consumers::sync(1);
        consumers::load(scratch.kf, g.kf, {0, head, 0, 0}); // next chunk
        consumers::sync(1);
    }
    // tk
    __device__ static inline int iters(const typename layout::globals &g) {
        int heads_handled = (g.x.depth+131-blockIdx.x) / 132; // I am guaranteeing batch is handled by just one block.
        int iters_per_head = (g.x.batch + (NUM_CONSUMER_WARPGROUPS*16)-1) / (NUM_CONSUMER_WARPGROUPS*16);
        return heads_handled * iters_per_head;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            int iters_per_head = (args.globals.x.batch + (NUM_CONSUMER_WARPGROUPS*16)-1) / (NUM_CONSUMER_WARPGROUPS*16);
            int head  = (args.iter / iters_per_head)*132 + blockIdx.x;
            int batch = (args.iter % iters_per_head) * (NUM_CONSUMER_WARPGROUPS*16); // 16 batches per warpgroup
            if(warpgroup::warpid() == args.iter%4) {
                for(int b = batch; b < batch+(NUM_CONSUMER_WARPGROUPS*16) && b < args.globals.x.batch; b++) {
                    int diff = b-batch;
                    auto st = subtile_inplace<16,16>(args.input.x[diff/16], (diff%16)/2, diff%2);
                    load_async(st, args.globals.x, { b, head, 0, 0 });
                }
                load_async_wait();
                arrive(args.inputs_arrived, 16); // extra arrivals needed
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            int iters_per_head = (args.globals.x.batch + (NUM_CONSUMER_WARPGROUPS*16)-1) / (NUM_CONSUMER_WARPGROUPS*16);
            int head  = (args.iter / iters_per_head)*132 + blockIdx.x;
            int batch = (args.iter % iters_per_head) * (NUM_CONSUMER_WARPGROUPS*16); // 16 batches per warpgroup
            if(warpgroup::warpid() == args.iter%4) {
                for(int b = batch; b < batch+(NUM_CONSUMER_WARPGROUPS*16) && b < args.globals.x.batch; b++) {
                    int diff = b-batch;
                    auto st = subtile_inplace<16,16>(args.output.o[diff/16], (diff%16)/2, diff%2);
                    kittens::store(args.globals.o, st, { b, head, 0, 0 });
                }
                __syncwarp(); // memory must arrive before arrival
                arrive(args.outputs_finished, 16);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            int iters_per_head = (args.globals.x.batch + (NUM_CONSUMER_WARPGROUPS*16)-1) / (NUM_CONSUMER_WARPGROUPS*16);
            args.state.current_head = (0 / iters_per_head)*132 + blockIdx.x; // start for iter 0
            group<NUM_CONSUMER_WARPS>::load(args.scratch.f,       args.globals.f,       {0, 0, 0, 0});
            group<NUM_CONSUMER_WARPS>::load(args.scratch.finv,    args.globals.finv,    {0, 0, 0, 0});
            group<NUM_CONSUMER_WARPS>::load(args.scratch.tw,      args.globals.tw,      {0, 0, 0, 0});
            group<NUM_CONSUMER_WARPS>::load(args.scratch.twinv_t, args.globals.twinv_t, {0, 0, 0, 0});
            load_head_data(args.scratch, args.globals, args.state.current_head);
        }
        __device__ static void work(consumer_work_args<layout> args) {
            // warpgroup::copy(args.output.o[warpgroup::groupid()], args.input.x[warpgroup::groupid()]);
            // X = F^T X
            crt_fl<16, 64> mma_reg; // 64 registers
            crt_bf<16, 64> accum, tmp; // 32 registers each
            warpgroup::mm_AB(mma_reg.real, args.scratch.f.real, args.input.x[warpgroup::groupid()]);
            warpgroup::mm_AB(mma_reg.imag, args.scratch.f.imag, args.input.x[warpgroup::groupid()]);
            warpgroup::mma_async_wait();
            copy(accum, mma_reg);

            warpgroup::load(tmp, args.scratch.tw); // for twiddle first
            mul(accum, accum, tmp);

            group<NUM_CONSUMER_WARPS>::sync();
            warpgroup::mm_AB(mma_reg, accum, args.scratch.f);
            warpgroup::mma_async_wait();
            copy(accum, mma_reg);

            warpgroup::load(tmp, args.scratch.kf); // for filter second
            mul(accum, accum, tmp);

            warpgroup::mm_AB(mma_reg, accum, args.scratch.finv);
            warpgroup::mma_async_wait();
            copy(accum, mma_reg);

            warpgroup::load(tmp, args.scratch.twinv_t); // twiddle inverse is pre-transposed
            mul(accum, accum, tmp);

            warpgroup::store(args.scratch.tmp[warpgroup::groupid()], accum); // must store for AtB
            warpgroup::sync();

            warpgroup::mm_AB(mma_reg, args.scratch.finv, args.scratch.tmp[warpgroup::groupid()]); // TODO: optimize
            warpgroup::mma_async_wait();
            
            warpgroup::store(args.output.o[warpgroup::groupid()], mma_reg.real); // COMMENT ME OUT LATER
            warpgroup::sync();

            arrive(args.inputs_finished);
            arrive(args.outputs_arrived);
            int iters_per_head = (args.globals.x.batch + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
            int next_head = ((args.iter+1) / iters_per_head)*132 + blockIdx.x;
            if(next_head != args.state.current_head) {
                load_head_data(args.scratch, args.globals, next_head);
                args.state.current_head = next_head;
            }
        }
    };
};
template<int _wg> struct fftconv_1024_layout { // 4096
    static constexpr int wg = _wg;
    using seq_tile      = st_bf<64, 64>;
    using seq_layout    =     gl<bf16, -1, -1, 32, 32>;
    using filter_layout = cgl<gl<bf16,  1, -1, 64, 64>>;
    using fft_layout    = cgl<gl<bf16,  1,  1, 64, 64>>;
    struct globals {
        seq_layout o, x;
        filter_layout kf;
        fft_layout f, finv, tw, twinv_t;
    };
    struct input_block    { seq_tile x[wg]; };
    struct output_block   { seq_tile o[wg]; };
    struct scratch_block  {
        cst_bf<64, 64> kf, f, finv, tw, twinv_t, tmp[2];
    };
    struct consumer_state { int current_head; };
};
struct fft_1024_template {
    static constexpr int NUM_CONSUMER_WARPS=8, NUM_CONSUMER_WARPGROUPS=NUM_CONSUMER_WARPS/4, NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=3, INPUT_PIPE_STAGES=3;
    using layout = fftconv_1024_layout<NUM_CONSUMER_WARPGROUPS>;
    // mine
    __device__ static inline void load_head_data(typename layout::scratch_block &scratch, const layout::globals &g, int head) {
        using consumers = group<NUM_CONSUMER_WARPS>;
        consumers::sync(1);
        consumers::load(scratch.kf, g.kf, {0, head, 0, 0}); // next chunk
        consumers::sync(1);
    }
    // tk
    __device__ static inline int iters(const typename layout::globals &g) {
        int heads_handled = (g.x.depth+131-blockIdx.x) / 132; // I am guaranteeing batch is handled by just one block.
        int iters_per_head = (g.x.batch + (NUM_CONSUMER_WARPGROUPS*4)-1) / (NUM_CONSUMER_WARPGROUPS*4);
        return heads_handled * iters_per_head;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            int iters_per_head = (args.globals.x.batch + (NUM_CONSUMER_WARPGROUPS*4)-1) / (NUM_CONSUMER_WARPGROUPS*4);
            int head  = (args.iter / iters_per_head)*132 + blockIdx.x;
            int batch = (args.iter % iters_per_head) * (NUM_CONSUMER_WARPGROUPS*4); // 4 batch per warpgroup
            if(warpgroup::warpid() == args.iter%4) {
                for(int b = batch; b < batch+(NUM_CONSUMER_WARPGROUPS*4) && b < args.globals.x.batch; b++) {
                    int diff = b-batch;
                    auto st = subtile_inplace<32,32>(args.input.x[diff/4], (diff%4)/2, diff%2);
                    load_async(st, args.globals.x, { b, head, 0, 0 });
                }
                load_async_wait();
                arrive(args.inputs_arrived, 4); // extra arrivals needed
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            int iters_per_head = (args.globals.x.batch + (NUM_CONSUMER_WARPGROUPS*4)-1) / (NUM_CONSUMER_WARPGROUPS*4);
            int head  = (args.iter / iters_per_head)*132 + blockIdx.x;
            int batch = (args.iter % iters_per_head) * (NUM_CONSUMER_WARPGROUPS*4); // 4 batch per warpgroup
            if(warpgroup::warpid() == args.iter%4) {
                for(int b = batch; b < batch+(NUM_CONSUMER_WARPGROUPS*4) && b < args.globals.x.batch; b++) {
                    int diff = b-batch;
                    auto st = subtile_inplace<32,32>(args.output.o[diff/4], (diff%4)/2, diff%2);
                    kittens::store(args.globals.o, st, { b, head, 0, 0 });
                }
                __syncwarp(); // memory must arrive before arrival
                arrive(args.outputs_finished, 4);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            int iters_per_head = (args.globals.x.batch + (NUM_CONSUMER_WARPGROUPS*4)-1) / (NUM_CONSUMER_WARPGROUPS*4);
            args.state.current_head = (0 / iters_per_head)*132 + blockIdx.x; // start for iter 0
            group<NUM_CONSUMER_WARPS>::load(args.scratch.f,       args.globals.f,       {0, 0, 0, 0});
            group<NUM_CONSUMER_WARPS>::load(args.scratch.finv,    args.globals.finv,    {0, 0, 0, 0});
            group<NUM_CONSUMER_WARPS>::load(args.scratch.tw,      args.globals.tw,      {0, 0, 0, 0});
            group<NUM_CONSUMER_WARPS>::load(args.scratch.twinv_t, args.globals.twinv_t, {0, 0, 0, 0});
            load_head_data(args.scratch, args.globals, args.state.current_head);
        }
        __device__ static void work(consumer_work_args<layout> args) {
            // warpgroup::copy(args.output.o[warpgroup::groupid()], args.input.x[warpgroup::groupid()]);
            // X = F^T X
            crt_fl<16, 64> mma_reg; // 64 registers
            crt_bf<16, 64> accum, tmp; // 32 registers each
            warpgroup::mm_AB(mma_reg.real, args.scratch.f.real, args.input.x[warpgroup::groupid()]);
            warpgroup::mm_AB(mma_reg.imag, args.scratch.f.imag, args.input.x[warpgroup::groupid()]);
            warpgroup::mma_async_wait();
            copy(accum, mma_reg);

            warpgroup::load(tmp, args.scratch.tw); // for twiddle first
            mul(accum, accum, tmp);

            group<NUM_CONSUMER_WARPS>::sync();
            warpgroup::mm_AB(mma_reg, accum, args.scratch.f);
            warpgroup::mma_async_wait();
            copy(accum, mma_reg);

            warpgroup::load(tmp, args.scratch.kf); // for filter second
            mul(accum, accum, tmp);

            warpgroup::mm_AB(mma_reg, accum, args.scratch.finv);
            warpgroup::mma_async_wait();
            copy(accum, mma_reg);

            warpgroup::load(tmp, args.scratch.twinv_t); // twiddle inverse is pre-transposed
            mul(accum, accum, tmp);

            warpgroup::store(args.scratch.tmp[warpgroup::groupid()], accum); // must store for AtB
            warpgroup::sync();

            warpgroup::mm_AB(mma_reg, args.scratch.finv, args.scratch.tmp[warpgroup::groupid()]); // TODO: optimize
            warpgroup::mma_async_wait();
            
            warpgroup::store(args.output.o[warpgroup::groupid()], mma_reg.real); // COMMENT ME OUT LATER
            warpgroup::sync();

            arrive(args.inputs_finished);
            arrive(args.outputs_arrived);
            int iters_per_head = (args.globals.x.batch + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
            int next_head = ((args.iter+1) / iters_per_head)*132 + blockIdx.x;
            if(next_head != args.state.current_head) {
                load_head_data(args.scratch, args.globals, next_head);
                args.state.current_head = next_head;
            }
        }
    };
};
template<int _wg> struct fftconv_4096_layout { // 4096
    static constexpr int wg = _wg;
    using seq_tile      = st_bf<64, 64>;
    using seq_layout    =     gl<bf16, -1, -1, 64, 64, seq_tile>;
    using filter_layout = cgl<gl<bf16,  1, -1, 64, 64, seq_tile>>;
    using fft_layout    = cgl<gl<bf16,  1,  1, 64, 64>>;
    struct globals {
        seq_layout o, x;
        filter_layout kf;
        fft_layout f, finv, tw, twinv_t;
    };
    struct input_block    { seq_tile x[wg]; };
    struct output_block   { seq_tile o[wg]; };
    struct scratch_block  {
        cst_bf<64, 64> kf, f, finv, tw, twinv_t, tmp[2];
    };
    struct consumer_state { int current_head; };
};
struct fft_4096_template {
    static constexpr int NUM_CONSUMER_WARPS=8, NUM_CONSUMER_WARPGROUPS=NUM_CONSUMER_WARPS/4, NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=2, INPUT_PIPE_STAGES=4;
    using layout = fftconv_4096_layout<NUM_CONSUMER_WARPGROUPS>;
    // mine
    __device__ static inline void load_head_data(typename layout::scratch_block &scratch, const layout::globals &g, int head) {
        using consumers = group<NUM_CONSUMER_WARPS>;
        consumers::sync(1);
        consumers::load(scratch.kf, g.kf, {0, head, 0, 0}); // next chunk
        consumers::sync(1);
    }
    // tk
    __device__ static inline int iters(const typename layout::globals &g) {
        int heads_handled = (g.x.depth+131-blockIdx.x) / 132; // I am guaranteeing batch is handled by just one block.
        int iters_per_head = (g.x.batch + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
        return heads_handled * iters_per_head;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static void load(producer_load_args<layout> args) {
            int iters_per_head = (args.globals.x.batch + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
            int head  = (args.iter / iters_per_head)*132 + blockIdx.x;
            int batch = (args.iter % iters_per_head) * NUM_CONSUMER_WARPGROUPS;
            if(warpgroup::warpid() == args.iter%4) {
                tma::expect_bytes(args.inputs_arrived, sizeof(args.input.x[0]) * min((int)NUM_CONSUMER_WARPGROUPS, (int)(args.globals.x.batch - batch)));
                for(int b = batch; b < batch+NUM_CONSUMER_WARPGROUPS && b < args.globals.x.batch; b++) {
                    tma::load_async(args.input.x[b-batch], args.globals.x, { b, head, 0, 0 }, args.inputs_arrived);
                }
                arrive(args.inputs_arrived, 3); // extra arrivals needed
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            int iters_per_head = (args.globals.x.batch + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
            int head  = (args.iter / iters_per_head)*132 + blockIdx.x;
            int batch = (args.iter % iters_per_head) * NUM_CONSUMER_WARPGROUPS;
            if(warpgroup::warpid() == args.iter%4) {
                for(int b = batch; b < batch+NUM_CONSUMER_WARPGROUPS && b < args.globals.x.batch; b++) {
                    tma::store_async(args.globals.o, args.output.o[b-batch], { b, head, 0, 0 });
                }
                tma::store_async_read_wait();
                arrive(args.outputs_finished, 4);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/4>();
            int iters_per_head = (args.globals.x.batch + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
            args.state.current_head = (0 / iters_per_head)*132 + blockIdx.x; // start for iter 0
            group<NUM_CONSUMER_WARPS>::load(args.scratch.f,       args.globals.f,       {0, 0, 0, 0});
            group<NUM_CONSUMER_WARPS>::load(args.scratch.finv,    args.globals.finv,    {0, 0, 0, 0});
            group<NUM_CONSUMER_WARPS>::load(args.scratch.tw,      args.globals.tw,      {0, 0, 0, 0});
            group<NUM_CONSUMER_WARPS>::load(args.scratch.twinv_t, args.globals.twinv_t, {0, 0, 0, 0});
            load_head_data(args.scratch, args.globals, args.state.current_head);
        }
        __device__ static void work(consumer_work_args<layout> args) {
            // X = F^T X
            crt_fl<16, 64> mma_reg; // 64 registers
            crt_bf<16, 64> accum, tmp; // 32 registers each
            warpgroup::mm_AB(mma_reg.real, args.scratch.f.real, args.input.x[warpgroup::groupid()]);
            warpgroup::mm_AB(mma_reg.imag, args.scratch.f.imag, args.input.x[warpgroup::groupid()]);
            warpgroup::mma_async_wait();
            copy(accum, mma_reg);

            warpgroup::load(tmp, args.scratch.tw); // for twiddle first
            mul(accum, accum, tmp);

            group<NUM_CONSUMER_WARPS>::sync();
            warpgroup::mm_AB(mma_reg, accum, args.scratch.f);
            warpgroup::mma_async_wait();
            copy(accum, mma_reg);

            warpgroup::load(tmp, args.scratch.kf); // for filter second
            mul(accum, accum, tmp);

            warpgroup::mm_AB(mma_reg, accum, args.scratch.finv);
            warpgroup::mma_async_wait();
            copy(accum, mma_reg);

            warpgroup::load(tmp, args.scratch.twinv_t); // twiddle inverse is pre-transposed
            mul(accum, accum, tmp);

            warpgroup::store(args.scratch.tmp[warpgroup::groupid()], accum); // must store for AtB
            warpgroup::sync();

            warpgroup::mm_AB(mma_reg, args.scratch.finv, args.scratch.tmp[warpgroup::groupid()]); // TODO: optimize
            warpgroup::mma_async_wait();
            
            warpgroup::store(args.output.o[warpgroup::groupid()], mma_reg.real); // COMMENT ME OUT LATER
            warpgroup::sync();

            arrive(args.inputs_finished);
            arrive(args.outputs_arrived);
            int iters_per_head = (args.globals.x.batch + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
            int next_head = ((args.iter+1) / iters_per_head)*132 + blockIdx.x;
            if(next_head != args.state.current_head) {
                load_head_data(args.scratch, args.globals, next_head);
                args.state.current_head = next_head;
            }
        }
    };
};

#ifdef TORCH_COMPILE_FFTCONV
#include "common/pyutils/torch_helpers.cuh"
#include <iostream>
void dispatch_fft_conv( 
    bf16 *u, 
    bf16 *kf, bf16 *kf_imag, 
    bf16 *f, bf16 *f_imag, 
    bf16 *finv, bf16 *finv_imag, 
    bf16 *tw, bf16 *tw_imag, 
    bf16 *twinv, bf16 *twinv_imag, 
    bf16 *o, uint B, uint H, uint N, uint N1
){
    // if (N == 1024) {
        // using fftst = fft_1024_template;
    // } else {
    // using fftst = fft_4096_template;
    using fftst = fft_256_template;
    // }
    using globals       = typename fftst::layout::globals;
    using fft_layout    = typename fftst::layout::fft_layout;
    using filter_layout = typename fftst::layout::filter_layout;
    using seq_layout    = typename fftst::layout::seq_layout;

    // input and output
    seq_layout u_gl{u, B, H, nullptr, nullptr};
    seq_layout o_gl{o, B, H, nullptr, nullptr};
    // filters
    filter_layout kf_gl{
        typename filter_layout::GL{kf, nullptr, H, nullptr, nullptr}, 
        typename filter_layout::GL{kf_imag, nullptr, H, nullptr, nullptr}
    };
    // factors
    fft_layout f_gl{
        typename fft_layout::GL{f, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::GL{f_imag, nullptr, nullptr, nullptr, nullptr}
    };
    fft_layout tw_gl{
        typename fft_layout::GL{tw, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::GL{tw_imag, nullptr, nullptr, nullptr, nullptr}
    };
    fft_layout finv_gl{
        typename fft_layout::GL{finv, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::GL{finv_imag, nullptr, nullptr, nullptr, nullptr}
    };
    fft_layout twinv_t_gl{
        typename fft_layout::GL{twinv, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::GL{twinv_imag, nullptr, nullptr, nullptr, nullptr}
    };
    globals G{u_gl, o_gl, kf_gl, f_gl, finv_gl, tw_gl, twinv_t_gl};

    // launch setup
    unsigned long mem_size = (MAX_SHARED_MEMORY-1024);
    cudaFuncSetAttribute(
        pc<fftst>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(132);
    dim3 block(num_threads<fftst>);
    pc<fftst><<<grid, block, mem_size>>>(G);
}


torch::Tensor fftconv(
    const torch::Tensor u_real,
    const torch::Tensor kf_real,
    const torch::Tensor kf_imag,
    const torch::Tensor f_real,
    const torch::Tensor f_imag,
    const torch::Tensor finv_real,
    const torch::Tensor finv_imag,
    const torch::Tensor tw_real,
    const torch::Tensor tw_imag,
    const torch::Tensor twinv_real,
    const torch::Tensor twinv_imag,
    int B,
    int H,
    int N,
    int N1,
    int B_TILE,
    int H_TILE
) {
    CHECK_INPUT(u_real);
    CHECK_INPUT(kf_real);
    CHECK_INPUT(kf_imag);
    CHECK_INPUT(f_real);
    CHECK_INPUT(f_imag);
    CHECK_INPUT(finv_real);
    CHECK_INPUT(finv_imag);
    CHECK_INPUT(tw_real);
    CHECK_INPUT(tw_imag);
    CHECK_INPUT(twinv_real);
    CHECK_INPUT(twinv_imag);

    // printf("B: %d, H: %d, N: %d, N1: %d, B_TILE: %d, H_TILE: %d\n", B, H, N, N1, B_TILE, H_TILE);
    // printf("u_real: %d, %d, %d\n", u_real.size(0), u_real.size(1), u_real.size(2));
    
    // checks
    TORCH_CHECK(u_real.size(0) == B, "u_real has incompatible batch shape");
    TORCH_CHECK(u_real.size(1) == H, "u_real has incompatible head shape");
    TORCH_CHECK(u_real.size(2) == N1, "u_real has incompatible sequence shape");

    TORCH_CHECK(f_real.size(0) == N1, "f_real has incompatible dim");
    TORCH_CHECK(f_real.size(1) == N1, "f_real has incompatible dim");

    TORCH_CHECK(f_imag.size(0) == N1, "f_imag has incompatible dim");
    TORCH_CHECK(f_imag.size(1) == N1, "f_imag has incompatible dim");

    TORCH_CHECK(finv_real.size(0) == N1, "finv_real has incompatible dim");
    TORCH_CHECK(finv_real.size(1) == N1, "finv_real has incompatible dim");

    TORCH_CHECK(finv_imag.size(0) == N1, "finv_imag has incompatible dim");
    TORCH_CHECK(finv_imag.size(1) == N1, "finv_imag has incompatible dim");

    TORCH_CHECK(tw_real.size(0) == N1, "tw_real has incompatible dim");
    TORCH_CHECK(tw_real.size(1) == N1, "tw_real has incompatible dim");

    TORCH_CHECK(tw_imag.size(0) == N1, "tw_imag has incompatible dim");
    TORCH_CHECK(tw_imag.size(1) == N1, "tw_imag has incompatible dim");

    torch::Tensor out = torch::empty({B, H, N1, N1}, u_real.options());

    // convert to bf16
    c10::BFloat16 *u_real_bf16 = u_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kf_real_bf16 = kf_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kf_imag_bf16 = kf_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *f_real_bf16 = f_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *f_imag_bf16 = f_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *finv_real_bf16 = finv_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *finv_imag_bf16 = finv_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *tw_real_bf16 = tw_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *tw_imag_bf16 = tw_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *twinv_real_bf16 = twinv_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *twinv_imag_bf16 = twinv_imag.data_ptr<c10::BFloat16>();

    bf16 *d_u_real = reinterpret_cast<bf16*>(u_real_bf16);
    bf16 *d_kf_real = reinterpret_cast<bf16*>(kf_real_bf16);
    bf16 *d_kf_imag = reinterpret_cast<bf16*>(kf_imag_bf16);
    bf16 *d_f_real = reinterpret_cast<bf16*>(f_real_bf16);
    bf16 *d_f_imag = reinterpret_cast<bf16*>(f_imag_bf16);
    bf16 *d_finv_real = reinterpret_cast<bf16*>(finv_real_bf16);
    bf16 *d_finv_imag = reinterpret_cast<bf16*>(finv_imag_bf16);
    bf16 *d_tw_real = reinterpret_cast<bf16*>(tw_real_bf16);
    bf16 *d_tw_imag = reinterpret_cast<bf16*>(tw_imag_bf16);
    bf16 *d_twinv_real = reinterpret_cast<bf16*>(twinv_real_bf16);
    bf16 *d_twinv_imag = reinterpret_cast<bf16*>(twinv_imag_bf16);
    bf16 *d_out = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());

    dispatch_fft_conv(
        d_u_real, 
        d_kf_real, d_kf_imag, 
        d_f_real, d_f_imag, d_finv_real, d_finv_imag, 
        d_tw_real, d_tw_imag, d_twinv_real, d_twinv_imag, 
        d_out, B, H, N, N1
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    return out;
}
#else
#include "harness_async.impl"
#endif
