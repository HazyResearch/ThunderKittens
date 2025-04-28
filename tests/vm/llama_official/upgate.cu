#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;
// using namespace kittens::prototype::vm;

namespace kittens::prototype::vm
{

    using globals = llama_1b_globals;
    using config = default_config;

    using block_rt = rt_fl<16, 128>;
    using block_st = st_bf<16, 128>;
    using block_rv = rv_fl<16>;

    template <typename Config, typename Globals>
    struct rms_upgate_silu
    {
        static constexpr int opcode = OPCODE_RMS_DoubleMatVecSiLU; // Op index within the layer -- controls which barrier to listen to.
        static constexpr int prev_opcode = OPCODE_O_ProjResidual;
        static constexpr int EXPECTED_ARRIVAL_COUNT = Globals::hidden_dim / Globals::matvec_block_size;
        static constexpr int NUM_STAGES = 3;

        struct parsed_instruction
        {
            int layer;
            int output_block_idx;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                layer = instruction[1];
                output_block_idx = instruction[2];
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        static constexpr int NUM_UP_PAGES = 4;
        static constexpr int NUM_GATE_PAGES = 4;
        static constexpr int PAGE_RMS_SCALE = 0;
        static constexpr int PAGE_UP_START = PAGE_RMS_SCALE + 1;
        static constexpr int PAGE_GATE_START = PAGE_UP_START + NUM_UP_PAGES;
        static constexpr int PAGE_INPUT = PAGE_GATE_START + NUM_GATE_PAGES;
        static constexpr int PAGE_COUNT = NUM_UP_PAGES + NUM_GATE_PAGES + 2;
        static constexpr int SEM_COUNT = NUM_UP_PAGES + NUM_GATE_PAGES + 3;

        //  semaphores
        __device__ static inline semaphore &up_arrived(state<Config> &s, int i) { return s.semaphores()[i]; }
        __device__ static inline semaphore &gate_arrived(state<Config> &s, int i) { return s.semaphores()[NUM_UP_PAGES + i]; }
        __device__ static inline semaphore &in_arrived(state<Config> &s) { return s.semaphores()[NUM_UP_PAGES + NUM_GATE_PAGES]; }
        __device__ static inline semaphore &rms_scale_arrived(state<Config> &s) { return s.semaphores()[NUM_UP_PAGES + NUM_GATE_PAGES + 1]; }
        __device__ static inline semaphore &out_arrived(state<Config> &s) { return s.semaphores()[NUM_UP_PAGES + NUM_GATE_PAGES + 2]; }

        // getters
        __device__ static inline int get_rms_scale_page(state<Config> &s) { return s.pid(PAGE_RMS_SCALE); }
        __device__ static inline int get_up_page(state<Config> &s, int i) { return s.pid(PAGE_UP_START + i); }
        __device__ static inline int get_gate_page(state<Config> &s, int i) { return s.pid(PAGE_GATE_START + i); }
        __device__ static inline int get_input_page(state<Config> &s) { return s.pid(PAGE_INPUT); }

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
            {
                // int ret_order[] = {
                //     6, 7, 8, 9, 10, 11, 12, 13,
                //     14,
                //     0, 1, 2, 3, 4, 5};

                // TODO the above is too long (we only have 12 pages), get proper order later
                // int ret_order[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

                // first the pages we don't use (we use 10 pages)
                // then input, then rms scale, then up, then gate

                int ret_order[] = {10, 11, 12, PAGE_INPUT, PAGE_RMS_SCALE, PAGE_UP_START, PAGE_UP_START + 1, PAGE_UP_START + 2, PAGE_UP_START + 3, PAGE_GATE_START, PAGE_GATE_START + 1, PAGE_GATE_START + 2, PAGE_GATE_START + 3};

                return ret_order[query];
            }
            static __device__ int init_semaphores(const Globals &g, state<Config> &s)
            {

                // each weight page and the input page needs exactly 1 “ready” signal
                for (int i = 0; i < NUM_UP_PAGES; i++)
                {
                    init_semaphore(up_arrived(s, i), 1);
                }
                for (int i = 0; i < NUM_GATE_PAGES; i++)
                {
                    init_semaphore(gate_arrived(s, i), 1);
                }

                init_semaphore(in_arrived(s), 1);
                // output must wait for all 4 consumer warps
                init_semaphore(out_arrived(s), 16);
                init_semaphore(rms_scale_arrived(s), 1);
                s.record(1);

                return SEM_COUNT;
            }
        };

        struct loader
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {

                parsed_instruction inst{s};
                // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
                ((int *)s.scratch())[laneid()] = 0;
                warp::sync(); // done, now we can proceed to other things.

                if (laneid() == 0)
                {

                    int rms_scale_page = get_rms_scale_page(s);
                    s.wait_page_ready(rms_scale_page);
                    // s.record(16 + laneid());
                    auto &rms_scale = reinterpret_cast<sv_bf<2048> &>(s.pages[rms_scale_page]);
                    tma::expect(rms_scale_arrived(s), rms_scale);
                    tma::load_async(rms_scale, g.mlp_norm_weights, {inst.layer, 0}, rms_scale_arrived(s));

                    for (int i = 0; i < NUM_UP_PAGES; i++)
                    {

                        int pg = get_up_page(s, i);
                        s.wait_page_ready(pg);
                        s.record(16 + i);
                        auto &chunk = reinterpret_cast<st_bf<Globals::matvec_block_size, 512> &>(s.pages[pg]);
                        tma::expect(up_arrived(s, i), chunk);
                        tma::load_async(chunk, g.up_weights,
                                        {0, inst.layer, inst.output_block_idx, i},
                                        up_arrived(s, i));
                    }

                    for (int i = NUM_UP_PAGES; i < NUM_UP_PAGES + NUM_GATE_PAGES; i++)
                    {

                        int idx = i - NUM_UP_PAGES;
                        int pg = get_gate_page(s, idx);
                        s.wait_page_ready(pg);
                        s.record(16 + i);
                        auto &chunk = reinterpret_cast<st_bf<Globals::matvec_block_size, 512> &>(s.pages[pg]);
                        tma::expect(gate_arrived(s, idx), chunk);
                        tma::load_async(chunk, g.gate_weights,
                                        {0, inst.layer, inst.output_block_idx, idx},
                                        gate_arrived(s, idx));
                    }

                    // activations last, since there's a data dependency
                    int pg = get_input_page(s);
                    s.wait_page_ready(pg);
                    // s.record(16 + laneid());
                    // wait on barrier from previous op
                    while (*(volatile int *)&g.Bar[{inst.layer, prev_opcode - 1, 0}] < EXPECTED_ARRIVAL_COUNT)
                        __nanosleep(20);
                    auto &buf = reinterpret_cast<sv_bf<2048> &>(s.pages[pg]);
                    tma::expect(in_arrived(s), buf);
                    tma::load_async(buf, g.hidden_states, {}, in_arrived(s)); // TODO: SA check
                }

                // 5) UNUSED pages: release them immediately so consumer warps can retire
                // else if (laneid() >= PAGE_RMS_SCALE + 1 && laneid() < SEM_COUNT)
                else if (laneid() >= PAGE_COUNT && laneid() < config::NUM_PAGES)
                {
                    int pg = s.pid(laneid());
                    s.wait_page_ready(pg);
                    arrive(s.page_finished[pg], Config::NUM_CONSUMER_WARPS);
                }
            }
        };

        struct launcher
        {
            // launcher does nothing here, since this doesn't use tensor cores.
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                s.wait_tensor_ready();
                if (laneid() == 0)
                    arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
            }
        };

        struct consumer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {

                //--------------------------------------------------
                // LOAD INPUT ACTIVATIONS
                //--------------------------------------------------
                int group_id = warpgroup::groupid();
                int warp_id = warpgroup::warpid();

                // Setup register memory for silu mlp
                block_rt weights, gate_weights, broadcast_activations, gate_broadcast_activations;
                typename block_rt::row_vec activations_vec;
                typename block_rt::col_vec output_col_format, gate_output_col_format;
                block_rv output, gate_output;

                // setup for rms norm
                typename block_rt::row_vec rms_scale_vec;
                typename rt_fl<16, 128>::row_vec float_activations;
                rv_fl<Config::NUM_CONSUMER_WARPS> rms_partial_sums;
                // reinterpret cast!
                shared_allocator al((int *)s.scratch());
                using smem_rms_partial_sums_t = sv_fl<Config::NUM_CONSUMER_WARPS>;
                smem_rms_partial_sums_t(&smem_rms_partial_sums) = al.template allocate<smem_rms_partial_sums_t>();

                // Next we need to load the activations
                wait(in_arrived(s), 0);
                if (laneid() == 0)
                    s.record(32 + warpid());
                // reinterpret the activations page as sv_bf<128>[16]
                int activation_page = get_input_page(s);

                sv_bf<128>(&activations_smem)[16] = reinterpret_cast<sv_bf<128>(&)[16]>(s.pages[activation_page]);
                warp::load(activations_vec, activations_smem[warpid()]);
                warp::sync();
                warp::arrive(s.page_finished[activation_page]); // just 1 is sufficient

                //---------------------------------------------------
                // RMS NORM
                //---------------------------------------------------
                warp::copy(float_activations, activations_vec);                     // cast to float
                warp::mul(float_activations, float_activations, float_activations); // square
                float partial_sum = warp::sum(float_activations);                   // sum

                // aggregate sums across the 16 consumer warps
                if (laneid() == 0)
                {
                    smem_rms_partial_sums[warpid()] = partial_sum;
                }

                group<16>::sync(0);

                warp::load(rms_partial_sums, smem_rms_partial_sums);
                warp::sync();

                float full_sum = warp::sum(rms_partial_sums);
                float variance = full_sum / 2048.0f;
                float rms_scale = rsqrtf(variance + g.rms_norm_eps);
                warp::copy(float_activations, activations_vec);
                warp::mul(float_activations, float_activations, rms_scale);
                warp::copy(activations_vec, float_activations); // back to bf16

                wait(rms_scale_arrived(s), 0);
                int rms_scale_page = get_rms_scale_page(s);
                sv_bf<128>(&rms_scale_smem)[16] = reinterpret_cast<sv_bf<128>(&)[16]>(s.pages[rms_scale_page]);
                warp::load(rms_scale_vec, rms_scale_smem[warpid()]); // no idea why yet but we must load this here, otherwise deadlock
                warp::sync();

                warp::arrive(s.page_finished[rms_scale_page]);

                // multiply by rms scale
                warp::mul(activations_vec, activations_vec, rms_scale_vec);

                //--------------------------------------------------
                // UP MATVEC
                //--------------------------------------------------
                wait(up_arrived(s, group_id), 0);
                if (laneid() == 0)
                    s.record(64 + warpid());
                int weight_page = get_up_page(s, group_id);
                block_st(&weights_smem)[4] = reinterpret_cast<block_st(&)[4]>(s.pages[weight_page]);
                warp::load(weights, weights_smem[warp_id]);
                warp::sync();
                warp::arrive(s.page_finished[weight_page], Config::NUM_CONSUMER_WARPS / 4); // this is called by each warp in the warpgroup

                // broadcast this into a tile
                warp::broadcast_col(broadcast_activations, activations_vec);
                warp::mul(broadcast_activations, broadcast_activations, weights);
                warp::row_sum(output_col_format, broadcast_activations);
                warp::copy(output, output_col_format);
                warp::sync();

                //--------------------------------------------------
                // GATE MATVEC
                //--------------------------------------------------
                wait(gate_arrived(s, group_id), 0);
                if (laneid() == 0)
                    s.record(80 + warpid());
                int gate_weight_page = get_gate_page(s, group_id);
                block_st(&gate_weights_smem)[4] = reinterpret_cast<block_st(&)[4]>(s.pages[gate_weight_page]);
                warp::load(gate_weights, gate_weights_smem[warp_id]);
                warp::sync();
                warp::arrive(s.page_finished[gate_weight_page], Config::NUM_CONSUMER_WARPS / 4); // called by each warp in the warpgroup

                // broadcast this into a tile
                warp::broadcast_col(gate_broadcast_activations, activations_vec);
                warp::mul(gate_broadcast_activations, gate_broadcast_activations, gate_weights);
                warp::row_sum(gate_output_col_format, gate_broadcast_activations);
                warp::copy(gate_output, gate_output_col_format);
                warp::sync();

                float *scratch_f32 = (float *)s.scratch();
                if (laneid() < 16)
                {
                    float upout = output[0][0];
                    float gateout = gate_output[0][0];
                    atomicAdd(&scratch_f32[laneid()], upout);        // up
                    atomicAdd(&scratch_f32[laneid() + 16], gateout); // gate
                }
                warp::sync();                 // all adds have landed
                warp::arrive(out_arrived(s)); // let the storer know we’re done
            }
        };

        struct storer
        {

            static __device__ void run(const Globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};

                if (laneid() == 0)
                {
                    wait(out_arrived(s), 0);
                    s.record(125);

                    float *scratch_f32 = (float *)s.scratch();
                    bf16 *scratch_bf16 = (bf16 *)scratch_f32; // alias
/* fuse up * SiLU(gate) once, in float, then cast */
#pragma unroll
                    for (int i = 0; i < 16; ++i)
                    {
                        float up = scratch_f32[i];
                        float gate = scratch_f32[i + 16];

                        float silu = gate / (1.f + expf(-gate));
                        scratch_bf16[i] = bf16(up * silu);
                    }

                    sv_bf<16> &vec = *reinterpret_cast<sv_bf<16> *>(scratch_bf16);

                    tma::store_async(g.silu_out, vec, {0, 0, 0, inst.output_block_idx});
                    tma::store_async_wait();
                    s.record(126);
                }

                warp::sync();
                asm volatile("fence.acq_rel.gpu;");
                if (laneid() == 0)
                {

                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, 0}], 1);
                    // if constexpr (opcode == g.Bar.rows() - 1)
                    //     atomicAdd(&g.Bar[{inst.layer + 1, 0, 0}], 1);
                    // else
                    //     atomicAdd(&g.Bar[{inst.layer, opcode + 1, 0}], 1);
                    s.record(127);
                }
            }
        };
    };
}
