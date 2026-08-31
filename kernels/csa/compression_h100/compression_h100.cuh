/*
    CSA block compression: averaging groups of tokens into fewer, weighted rows.

    Dimensions used in the shapes below:
      - B : batch size
      - N : sequence length
      - C : channel width
      - M : block size

    Input (global memory) {gm}:
      - value_a, value_b : [B, N, C] fp8e4m3
      - score_a, score_b : [B, N, C] fp8e4m3
      - bias_a,  bias_b  : [M, C]    bf16

    Output (global memory) {gm}:
      - compressed : [B, N/M, C] bf16

    Algorithm:
    We have a sequence of N token rows, each C numbers wide, and we want to shrink it down
    to N/M rows by combining every M tokens into one. To build output row i:

      - Take M tokens: the "a" group, tokens [i*M, i*M+M).
      - Also take the M tokens right before it: the "b" group, tokens [(i-1)*M, i*M).

            blocks of M tokens, left to right along the sequence:

                0 ... [ block i-2 ] [ block i-1 ] [ block i ] [ block i+1 ] ... N
                                        |             |
                                       "b"           "a"
                                  group for      group for
                                  output i       output i
                                        \_____________/
                                              |
                                              v
                                        output row i

      - Each of these 2M tokens carries two things: a "value" and a "score" .
        A learned per-position bonus (the "bias") is added to each score first.
        For position p = 0 .. M-1 within a block:

            za[p] {rm} = score_a[p] {sm} + bias_a[p] {sm}     | weight input, p-th token of the "a" group
            zb[p] {rm} = score_b[p] {sm} + bias_b[p] {sm}     | weight input, p-th token of the "b" group

      - Turn all 2M scores into weights that add up to 1 (bigger score means bigger
        weight), done separately for each of the C numbers in a row. For p = 0 .. M-1:

            Sa[p] {rm} = exp(za[p] {rm}) / ( sum_q exp(za[q] {rm}) + sum_q exp(zb[q] {rm}) )   | shared denominator
            Sb[p] {rm} = exp(zb[p] {rm}) / ( sum_q exp(za[q] {rm}) + sum_q exp(zb[q] {rm}) )   | over all 2M tokens

        Sa and Sb sharing one denominator is what makes this a single softmax over all 2M
        tokens together, not two separate M-token softmaxes!

      - Multiply every token's value by its own weight and add them all up. That sum is
        output row i, for p = 0 .. M-1 (out_c is this worker's own accumulator -- see
        consumer_state::out_c in the code below):

            out_c {rm} = sum_p Sa[p] {rm} * value_a[p] {sm}  +  sum_p Sb[p] {rm} * value_b[p] {sm}

    The very first output row (i=0) has no "b" group before the start of the sequence, so
    that half is simply left out for that one row: Sa is then normalized using only za,
    and the "b" term above drops out entirely.

            out_c {rm} = sum_p Sa[p] {rm} * value_a[p] {sm}     | i=0 case: "b" term dropped

    [.]

    NUM_WORKERS output rows are computed side by side per task. C_CHUNK: a single kernel
    launch only computes a C_CHUNK-wide slice of the full C channels (TMA's unswizzled-tile
    width cap is smaller than C); the host launches the kernel C/C_CHUNK times
    (globals.chunk_idx) to cover the whole width.

*/
#pragma once
#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

template<int C, int C_CHUNK, int M, int _NUM_WORKERS=4>
struct compression_layout {
    static_assert(C_CHUNK % 32 == 0, "C_CHUNK must be a multiple of 32 so it divides evenly across a warp's 32 lanes");
    static_assert(C % C_CHUNK == 0, "C must be a whole number of C_CHUNK-wide slices");
    static constexpr int NUM_WORKERS = _NUM_WORKERS;
    static constexpr int NUM_CHUNKS  = C / C_CHUNK;

    // M rows x C_CHUNK cols, unswizzled: M=4 is below TK's normal 16-row tile floor, and
    // disabling swizzling is the one tile mechanism that allows a smaller row count.

    using block_tile = st<fp8e4m3, M, C_CHUNK, false>;
    using bias_tile  = st<bf16, M, C_CHUNK, false>;

    using block_global = kittens::gl<fp8e4m3, -1, 1, -1, C, block_tile>; // value_a/b, score_a/b: [B,1,N,C]
    using bias_global  = kittens::gl<bf16, 1, 1, M, C, bias_tile>;       // bias_a/b: [1,1,M,C]
    using out_global   = kittens::gl<bf16, -1, 1, -1, C>;                // compressed: [B,1,N/M,C]

    struct globals {
        block_global value_a, value_b, score_a, score_b;
        bias_global  bias_a,  bias_b;
        out_global   compressed;
        int chunk_idx; // which C_CHUNK-wide column-tile this launch handles, 0..NUM_CHUNKS-1
    };
    struct input_block {
        block_tile value_a[NUM_WORKERS], value_b[NUM_WORKERS];
        block_tile score_a[NUM_WORKERS], score_b[NUM_WORKERS];
    };
    struct scratch_block {
        bias_tile bias_a, bias_b; // loaded once (task_iter == 0), reused by every task thereafter
    };
    struct common_state {
        int batch;
        int block_base; // first of up to NUM_WORKERS output-block indices this task handles
        int num_blocks; // N/M for this batch element
    };
    struct consumer_state {
        // Plain per-lane register array, not a TK tile type: TK's reductions need
        // swizzled tiles, which M=4 rules out (see block_tile above).
        static constexpr int LANE_CHANNELS = C_CHUNK / 32;
        float out_c[LANE_CHANNELS];
    };
};

template<int C, int C_CHUNK, int M, int NUM_WORKERS=4, int _INPUT_PIPE_STAGES=2>
struct compression_template {
    using layout = compression_layout<C, C_CHUNK, M, NUM_WORKERS>;
    static constexpr int NUM_CONSUMER_WARPS = NUM_WORKERS; // one warp per worker, one worker per output row
    static constexpr int INPUT_PIPE_STAGES  = _INPUT_PIPE_STAGES;

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int num_blocks       = args.globals.compressed.rows();
        int tasks_per_batch  = (num_blocks + NUM_WORKERS - 1) / NUM_WORKERS;
        int task_id          = gridDim.x * args.task_iter + blockIdx.x;

        args.common.batch      = task_id / tasks_per_batch;
        int task_in_batch      = task_id - args.common.batch * tasks_per_batch;
        args.common.block_base = task_in_batch * NUM_WORKERS;
        args.common.num_blocks = num_blocks;

        args.num_iters = (args.common.batch < args.globals.compressed.batch()) ? 1 : -1;
    }

    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            // Shrinks producer registers so consumer::setup can claim the freed space.
            warpgroup::producer_registers();

            // bias_a/b never change across the whole launch
            if (args.task_iter == 0) {
                warpgroup::load(args.scratch.bias_a, args.globals.bias_a, {0, 0, 0, args.globals.chunk_idx});
                warpgroup::load(args.scratch.bias_b, args.globals.bias_b, {0, 0, 0, args.globals.chunk_idx});
            }
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            // Every load below must actually be issued, never skipped
            if (warpgroup::warpid() == 0) warp::tma::expect(args.inputs_arrived, args.input);

            // inputs_arrived needs one arrive() per producer warp (PRODUCER_BARRIER_
            // ARRIVALS), independent of the byte-count gate above. expect() supplies warp
            // 0's token; plain load_async doesn't touch this gate, so the other warps need
            // an explicit arrive() even though they issue real loads below.
            if (warpgroup::warpid() != 0 && laneid() == 0) arrive(args.inputs_arrived);

            static_assert(layout::NUM_WORKERS % 4 == 0, "NUM_WORKERS must split evenly across the 4 producer warps");
            constexpr int WORKERS_PER_WARP = layout::NUM_WORKERS / 4;
            int pid = warpgroup::warpid(); // 0..3, which of the 4 producer warps this is
            #pragma unroll
            for (int i = 0; i < WORKERS_PER_WARP; i++) {
                int w     = pid * WORKERS_PER_WARP + i;
                int block = args.common.block_base + w;
                int safe_block   = min(block, args.common.num_blocks - 1);
                int safe_b_block = max(safe_block - 1, 0);
                warp::tma::load_async(args.input.value_a[w], args.globals.value_a, {args.common.batch, 0, safe_block, args.globals.chunk_idx}, args.inputs_arrived);
                warp::tma::load_async(args.input.score_a[w], args.globals.score_a, {args.common.batch, 0, safe_block, args.globals.chunk_idx}, args.inputs_arrived);
                warp::tma::load_async(args.input.value_b[w], args.globals.value_b, {args.common.batch, 0, safe_b_block, args.globals.chunk_idx}, args.inputs_arrived);
                warp::tma::load_async(args.input.score_b[w], args.globals.score_b, {args.common.batch, 0, safe_b_block, args.globals.chunk_idx}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            static_assert(NUM_CONSUMER_WARPS % 4 == 0, "consumer warp count must be a whole number of warpgroups");
            // consumer_registers<NCWG>() fails to compile at NCWG=7 (480/7 truncates to a
            // value that isn't a multiple of 8, required by setmaxnreg)
            if constexpr (NUM_CONSUMER_WARPS/4 == 7) warpgroup::increase_registers<56>();
            else                                     warpgroup::consumer_registers<NUM_CONSUMER_WARPS/4>();
        }
        // Softmax computed per-lane, per-channel 
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            // groupid() picks the consumer warpgroup, warpid() the warp within it
            int w     = warpgroup::groupid() * 4 + warpgroup::warpid();
            int block = args.common.block_base + w;
            if (block >= args.common.num_blocks) {
                if (laneid() == 0) arrive(args.inputs_finished);
                return;
            }
            bool has_b = block > 0;

            constexpr int LANE_CHANNELS = layout::consumer_state::LANE_CHANNELS;
            int lane = laneid();
            int c0   = lane * LANE_CHANNELS; // first channel this lane owns, local to this chunk's tile

            #pragma unroll
            for (int c = 0; c < LANE_CHANNELS; c++) {
                int ch = c0 + c;

                // pass 1: za[p] = score_a[p] + bias_a[p] (and zb, if this isn't block 0).
                float za[M], zb[M];
                float mx = -INFINITY;
                #pragma unroll
                for (int p = 0; p < M; p++) {
                    za[p] = float(args.input.score_a[w][{p, ch}]) + __bfloat162float(args.scratch.bias_a[{p, ch}]);
                    mx = max(mx, za[p]);
                }
                if (has_b) {
                    #pragma unroll
                    for (int p = 0; p < M; p++) {
                        zb[p] = float(args.input.score_b[w][{p, ch}]) + __bfloat162float(args.scratch.bias_b[{p, ch}]);
                        mx = max(mx, zb[p]);
                    }
                }

                // pass 2: shared softmax denominator; exp_za/exp_zb cached for pass 3.
                float exp_za[M], exp_zb[M];
                float denom = 0.f;
                #pragma unroll
                for (int p = 0; p < M; p++) {
                    exp_za[p] = expf(za[p] - mx);
                    denom += exp_za[p];
                }
                if (has_b) {
                    #pragma unroll
                    for (int p = 0; p < M; p++) {
                        exp_zb[p] = expf(zb[p] - mx);
                        denom += exp_zb[p];
                    }
                }

                // pass 3: out_c = sum_p Sa[p]*value_a[p] (+ sum_p Sb[p]*value_b[p]).
                float out = 0.f;
                #pragma unroll
                for (int p = 0; p < M; p++) {
                    float Sa = exp_za[p] / denom;
                    out += Sa * float(args.input.value_a[w][{p, ch}]);
                }
                if (has_b) {
                    #pragma unroll
                    for (int p = 0; p < M; p++) {
                        float Sb = exp_zb[p] / denom;
                        out += Sb * float(args.input.value_b[w][{p, ch}]);
                    }
                }

                args.state.out_c[c] = out;
            }

            if (laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            int w     = warpgroup::groupid() * 4 + warpgroup::warpid();
            int block = args.common.block_base + w;
            if (block < args.common.num_blocks) {
                constexpr int LANE_CHANNELS = layout::consumer_state::LANE_CHANNELS;
                int lane = laneid();
                int c0_local     = lane * LANE_CHANNELS;
                int chunk_offset = args.globals.chunk_idx * C_CHUNK; // this chunk's base in the true C-wide output
                #pragma unroll
                for (int c = 0; c < LANE_CHANNELS; c++)
                    args.globals.compressed[{args.common.batch, 0, block, chunk_offset + c0_local + c}] = __float2bfloat16(args.state.out_c[c]);
            }
            if (laneid() == 0) arrive(args.finish_finished);
        }
    };
};

// C_CHUNK=128, INPUT_PIPE_STAGES=1 is the best measured config on real H100 
using csa_compression_kv      = compression_template</*C=*/512, /*C_CHUNK=*/128, /*M=*/4, /*NUM_WORKERS=*/28, /*INPUT_PIPE_STAGES=*/1>;
using csa_compression_indexer = compression_template</*C=*/512, /*C_CHUNK=*/128, /*M=*/4, /*NUM_WORKERS=*/28, /*INPUT_PIPE_STAGES=*/1>;