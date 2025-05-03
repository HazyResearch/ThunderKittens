#include "llama.cuh"
#include "utils.cuh"
using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_1b_globals;
    using config = default_config;

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
        static constexpr int PAGE_RMS_SCALE_ACTIVATION = 0;
        static constexpr int PAGE_UP_START = PAGE_RMS_SCALE_ACTIVATION + 1;
        static constexpr int PAGE_GATE_START = PAGE_UP_START + NUM_UP_PAGES;
        static constexpr int PAGE_COUNT = NUM_UP_PAGES + NUM_GATE_PAGES + 1;
        static constexpr int SEM_COUNT = NUM_UP_PAGES + NUM_GATE_PAGES + 3;

        static constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        //  semaphores
        __device__ static inline semaphore &up_arrived(state<Config> &s, int i) { return s.semaphores()[i]; }
        __device__ static inline semaphore &gate_arrived(state<Config> &s, int i) { return s.semaphores()[NUM_UP_PAGES + i]; }
        __device__ static inline semaphore &in_arrived(state<Config> &s) { return s.semaphores()[NUM_UP_PAGES + NUM_GATE_PAGES]; }
        __device__ static inline semaphore &rms_scale_arrived(state<Config> &s) { return s.semaphores()[NUM_UP_PAGES + NUM_GATE_PAGES + 1]; }
        __device__ static inline semaphore &out_arrived(state<Config> &s) { return s.semaphores()[NUM_UP_PAGES + NUM_GATE_PAGES + 2]; }

        // getters
        __device__ static inline int get_rms_scale_activation_page(state<Config> &s) { return s.pid(PAGE_RMS_SCALE_ACTIVATION); }
        __device__ static inline int get_up_page(state<Config> &s, int i) { return s.pid(PAGE_UP_START + i); }
        __device__ static inline int get_gate_page(state<Config> &s, int i) { return s.pid(PAGE_GATE_START + i); }

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
            {
                // first the pages we don't use (we use 10 pages)
                // then input, then rms scale, then up, then gate

                int ret_order[] = {9, 10, 11, 12, PAGE_RMS_SCALE_ACTIVATION, PAGE_UP_START, PAGE_UP_START + 1, PAGE_UP_START + 2, PAGE_UP_START + 3, PAGE_GATE_START, PAGE_GATE_START + 1, PAGE_GATE_START + 2, PAGE_GATE_START + 3};

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
                init_semaphore(out_arrived(s), Config::NUM_CONSUMER_WARPS);
                init_semaphore(rms_scale_arrived(s), 1);

                return SEM_COUNT;
            }
        };

        struct loader
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};

                // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
                ((uint64_t *)s.scratch())[laneid()] = 0;
                warp::sync(); // done, now we can proceed to other things.

                if (laneid() == 0)
                {
                    int rms_scale_activation_page = get_rms_scale_activation_page(s);
                    s.wait_page_ready(rms_scale_activation_page);
                    auto &rms_scale = *reinterpret_cast<sv_bf<2048> *>(s.pages[rms_scale_activation_page].ptr());

                    s.record(TEVENT_TRIPLES_START);
                    tma::expect(rms_scale_arrived(s), rms_scale);
                    tma::load_async(rms_scale, g.mlp_norm_weights, {inst.layer, 0}, rms_scale_arrived(s));

                    for (int i = 0; i < NUM_UP_PAGES; i++)
                    {

                        int pg = get_up_page(s, i);
                        s.wait_page_ready(pg);
                        auto &chunk = reinterpret_cast<st_bf<Globals::matvec_block_size, 512> &>(s.pages[pg]);
                        s.record(TEVENT_TRIPLES_START + 1 + i);
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
                        auto &chunk = reinterpret_cast<st_bf<Globals::matvec_block_size, 512> &>(s.pages[pg]);
                        s.record(TEVENT_TRIPLES_START + 5 + i);
                        tma::expect(gate_arrived(s, idx), chunk);
                        tma::load_async(chunk, g.gate_weights,
                                        {0, inst.layer, inst.output_block_idx, idx},
                                        gate_arrived(s, idx));
                    }
                }

                // 5) UNUSED pages: release them immediately so consumer warps can retire
                // else if (laneid() >= PAGE_RMS_SCALE + 1 && laneid() < SEM_COUNT)
                else if (laneid() >= PAGE_COUNT && laneid() < config::NUM_PAGES)
                {
                    int pg = s.pid(laneid());
                    s.wait_page_ready(pg);
                    s.finish_page(pg, Config::NUM_CONSUMER_WARPS);
                }
            }
        };

        struct launcher
        {
            // launcher does nothing here, since this doesn't use tensor cores.
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                if (laneid() == 0)
                {
                    s.wait_tensor_ready();
                    arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);

                    parsed_instruction inst{s};

                    int rms_scale_activation_page = get_rms_scale_activation_page(s);
                    s.wait_page_ready(rms_scale_activation_page);
                    auto &activations = *reinterpret_cast<sv_bf<2048> *>(s.pages[rms_scale_activation_page].ptr(sizeof(sv_bf<2048>)));

                    // activations last, since there's a data dependency
                    // wait on barrier from previous op
                    s.record(TEVENT_AT_GMEM_WAIT);
                    while (*(volatile int *)&g.Bar[{inst.layer, prev_opcode - 1, 0}] < EXPECTED_ARRIVAL_COUNT)
                        __nanosleep(20);
                    s.record(TEVENT_DONE_GMEM_WAIT);
                    s.record(TEVENT_TRIPLES_START + 9);
                    tma::expect(in_arrived(s), activations);
                    tma::load_async(activations, g.hidden_states, {}, in_arrived(s)); // TODO: SA check
                }
            }
        };

        struct consumer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                // Setup
                using float_rt_t = rt_fl<16, REDUCTION_DIM_PER_WARP>;
                using float_rv_t = rv_fl<16>;

                parsed_instruction inst{s};
                rv_fl<REDUCTION_DIM_PER_WARP> activations_vec_naive;
                typename float_rt_t::row_vec activations_vec;
                static_assert(NUM_UP_PAGES == NUM_GATE_PAGES, "NUM_UP_PAGES must be equal to NUM_GATE_PAGES");
                static_assert(Config::NUM_CONSUMER_WARPS % NUM_UP_PAGES == 0, "NUM_CONSUMER_WARPS must be divisible by NUM_UP_PAGES");
                constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / NUM_UP_PAGES;

                int page_index = warpid() / WARPS_PER_PAGE;

                rms_norm(g, s, activations_vec_naive, get_rms_scale_activation_page(s), in_arrived(s), rms_scale_arrived(s), 32);
                warp::copy(activations_vec, activations_vec_naive);

                // up matvec
                matvec<float_rt_t, WARPS_PER_PAGE>(g, s, activations_vec, up_arrived(s, page_index), get_up_page(s, page_index), 0);

                s.warp_finish_page(get_rms_scale_activation_page(s), 1);

                // gate matvec
                matvec<float_rt_t, WARPS_PER_PAGE>(g, s, activations_vec, gate_arrived(s, page_index), get_gate_page(s, page_index), 16);

                // Release pages
                warp::sync();
                for (int i = 0; i < NUM_UP_PAGES; i++)
                {
                    s.warp_finish_page(get_up_page(s, i), 1);
                }
                for (int i = 0; i < NUM_GATE_PAGES; i++)
                {
                    s.warp_finish_page(get_gate_page(s, i), 1);
                }

                using block_rt = rt_fl<16, REDUCTION_DIM_PER_WARP>;
                using block_rv = rv_fl<16>;

                warp::sync();                 // all adds have landed
                warp::arrive(out_arrived(s)); // let the storer know we’re done
            }
        };

        struct storer
        {

            static __device__ void run(const Globals &g, state<Config> &s)
            {
                if (kittens::laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_STORE_START);
                }

                parsed_instruction inst{s};

                if (laneid() == 0)
                {
                    wait(out_arrived(s), 0);
                    s.record(TEVENT_TRIPLES_OUTPUT_READY);

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
                }

                warp::sync();
                asm volatile("fence.acq_rel.gpu;");
                if (laneid() == 0)
                {
                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, 0}], 1);
                }
            }
        };
    };
}
