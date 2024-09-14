#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
struct attn_fwd_template {
    using q_tile = st_bf<4,8>;
    using k_tile = st_bf<8,8>;
    using v_tile = st_bf<8,8>;
    using o_tile = st_bf<4,8>;
    using q_global = kittens::gl<bf16, -1, -1, -1, 128, q_tile>;
    using k_global = kittens::gl<bf16, -1, -1, -1, 128, k_tile>;
    using v_global = kittens::gl<bf16, -1, -1, -1, 128, v_tile>;
    using o_global = kittens::gl<bf16, -1, -1, -1, 128, o_tile>;
    static constexpr int NUM_CONSUMER_WARPS = 12, NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS / 4;
    static constexpr int INPUT_PIPE_STAGES = 2, OUTPUT_PIPE_STAGES = 0; // irrelevant for this kernel
    struct globals {
        q_global Qg;
        k_global Kg;
        v_global Vg;
        o_global Og;
    };
    struct input_block { // the chunk of data that the producer and consumer are working on
        k_tile k;
        v_tile v;
    };
    struct output_block {}; // nothing here, we store at the end in the consumer finish
    struct scratch_block {
        q_tile q[NUM_CONSUMER_WARPGROUPS];
    };
    struct finish_block {
        o_tile o[NUM_CONSUMER_WARPGROUPS];
    };
    struct producer {
        struct state { int n_blocks; }; // persistent registers
        __device__ static void setup(state &s, globals &g) { // setup and load the first iteration
            warpgroup::decrease_registers<24>(); // decrease registers for the producer warpgroup
            s.n_blocks = g.Kg.rows / k_tile::rows;
        }
        __device__ static bool load(state &s, input_block &b, globals &g, barrier &inputs_arrived, int iter) { // barrier for the producer to load into
            if(warpgroup::warpid() == 0) {
                tma::expect(inputs_arrived, b.k, b.v);
                tma::load_async(b.k, g.Kg, {blockIdx.z, blockIdx.y, iter, 0}, inputs_arrived);
                tma::load_async(b.v, g.Vg, {blockIdx.z, blockIdx.y, iter, 0}, inputs_arrived);
            }
            else arrive(inputs_arrived);
            return iter < s.n_blocks-1; // return true if there are more blocks to process
        }
    };
    struct consumer {
        struct state {
            rt_fl<1, v_tile::width> o_reg;
            col_vec<rt_fl<1, k_tile::height>> max_vec_last, max_vec;
            col_vec<rt_fl<1, k_tile::height>> norm_vec_last, norm_vec;
            int id, n_blocks;
        };
        __device__ static void setup(state &s, scratch_block &scratch, globals &g) { // setup locals for before the first iteration
            warpgroup::increase_registers<160>();
            s.id = warpgroup::groupid();
            s.n_blocks = g.Kg.rows / k_tile::rows;
            if((blockIdx.x*NUM_CONSUMER_WARPGROUPS + s.id)*q_tile::rows < g.Qg.rows)
                warpgroup::load(scratch.q[s.id], g.Qg, {blockIdx.z, blockIdx.y, blockIdx.x*NUM_CONSUMER_WARPGROUPS + s.id, 0});
            zero(s.o_reg);
            neg_infty(s.max_vec);
            zero(s.norm_vec);
            s.n_blocks = g.Kg.rows / k_tile::rows;
            warpgroup::sync();
            warpgroup::mul(scratch.q[s.id], scratch.q[s.id], __float2bfloat16(0.08838834764));
            warpgroup::sync();
        }
        __device__ static bool work(state &s, input_block &b, scratch_block &scratch, output_block &o, barrier &inputs_finished, barrier &outputs_arrived, int iter) {
            rt_fl<1, k_tile::height> att_block;
            warpgroup::mm_ABt(att_block, scratch.q[s.id], b.k);
            copy(s.norm_vec_last, s.norm_vec);
            copy(s.max_vec_last, s.max_vec);
            warpgroup::mma_async_wait();

            row_max(s.max_vec, att_block, s.max_vec); // accumulate onto the max_vec
            sub_row(att_block, att_block, s.max_vec);
            exp(att_block, att_block);
            
            sub(s.max_vec_last, s.max_vec_last, s.max_vec);
            exp(s.max_vec_last, s.max_vec_last);
            mul(s.norm_vec, s.norm_vec, s.max_vec_last);

            row_sum(s.norm_vec, att_block, s.norm_vec); // accumulate onto the norm_vec
            div_row(att_block, att_block, s.norm_vec);
            
            mul(s.norm_vec_last, s.norm_vec_last, s.max_vec_last);
            div(s.norm_vec_last, s.norm_vec_last, s.norm_vec);
            
            rt_bf<1, k_tile::height> att_block_mma;
            copy(att_block_mma, att_block); // convert to bf16 for mma
            mul_row(s.o_reg, s.o_reg, s.norm_vec_last); // normalize o_prev in advance of mma'ing onto it

            warpgroup::mma_AB(s.o_reg, att_block_mma, b.v);
            arrive(outputs_arrived); // we have no outputs, so we can do this early. (they're always ready.)
            warpgroup::mma_async_wait();

            arrive(inputs_finished);
            return iter < s.n_blocks-1;
        }
        __device__ static void finish(state &s, finish_block &f, globals &g, int _) {
            if((blockIdx.x*NUM_CONSUMER_WARPGROUPS + s.id)*q_tile::rows < g.Qg.rows) {
                warpgroup::store(f.o[s.id], s.o_reg);
                warpgroup::sync();
                if(warpgroup::warpid() == 0) {
                    tma::store_async(g.Og, f.o[s.id], {blockIdx.z, blockIdx.y, blockIdx.x*NUM_CONSUMER_WARPGROUPS + s.id, 0});
                }
                tma::store_async_read_wait();
            }
        }
    };
};

#include "harness.impl"