#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
template<int _wg> struct fftconv_short_layout { // 4096
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
struct fft_short_template {
    static constexpr int NUM_CONSUMER_WARPS=8, NUM_CONSUMER_WARPGROUPS=NUM_CONSUMER_WARPS/4, NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=2, INPUT_PIPE_STAGES=4;
    using layout = fftconv_short_layout<NUM_CONSUMER_WARPGROUPS>;
    // mine
    __device__ static inline void load_head_data(typename layout::scratch_block &scratch, const layout::globals &g, int head) {
        using consumers = group<NUM_CONSUMER_WARPS>;
        consumers::sync(1);
        consumers::load(scratch.kf, g.kf, {0, head, 0, 0}); // next chunk
        consumers::sync(1);
    }
    // tk
    __device__ static inline int iters(const typename layout::globals &g) {
        int heads_handled = (g.x.depth+132-blockIdx.x) / 132; // I am guaranteeing batch is handled by just one block.
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


#include "harness_async.impl"