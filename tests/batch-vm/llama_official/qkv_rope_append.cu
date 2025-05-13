#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_8b_globals;

    template <typename Config, typename Globals>
    struct qkv_rope_append
    {
        static constexpr int opcode = OPCODE_QKV_RopeAppend;
        static constexpr int PIPELINE_STAGES = 3;
        static constexpr int REDUCTION_BLOCK_SIZE = 128;
        static constexpr int K_BLOCK_START = Globals::num_attention_heads;
        static constexpr int V_BLOCK_START = Globals::num_attention_heads + Globals::num_kv_heads;

        static_assert(Globals::batch_size == 128 && Globals::head_dim == 128);

        using weight_tile = st_bf<Globals::head_dim, REDUCTION_BLOCK_SIZE>;
        using activation_tile = st_bf<Globals::batch_size, REDUCTION_BLOCK_SIZE>;
        using output_tile = st_bf<Globals::batch_size, Globals::head_dim>;
        using output_vec = sv_bf<Globals::head_dim>;
        using rope_vec = sv_fl<Globals::head_dim>;

        struct parsed_instruction
        {
            int layer_idx;
            int block_idx;
            int num_iters;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                layer_idx = instruction[1];
                block_idx = instruction[2]; // output block idx, in units of 128 (`Globals::head_dim`)
                num_iters = instruction[3]; // iterations on the reduction dimension, in units of 128 (`REDUCTION_BLOCK_SIZE`)
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        __device__ static inline int get_weight_page(state<Config> &s, int stage) { return 0 + stage * 2; }     // 32 KB pages
        __device__ static inline int get_activation_page(state<Config> &s, int stage) { return 6 + stage * 2; } // 32 KB pages

        __device__ static inline semaphore &inputs_arrived(state<Config> &s, int stage) { return s.semaphores()[PIPELINE_STAGES * 0 + stage]; }
        __device__ static inline semaphore &inputs_finished(state<Config> &s, int stage) { return s.semaphores()[PIPELINE_STAGES * 1 + stage]; }
        __device__ static inline semaphore &outputs_arrived(state<Config> &s, int stage) { return s.semaphores()[PIPELINE_STAGES * 2 + stage]; }
        __device__ static inline semaphore &outputs_shared(state<Config> &s) { return s.semaphores()[PIPELINE_STAGES * 3 + 0]; }
        __device__ static inline semaphore &rope_arrived(state<Config> &s) { return s.semaphores()[PIPELINE_STAGES * 3 + 1]; }

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
            {
                return query;
            }
            static __device__ int init_semaphores(const Globals &g, state<Config> &s)
            {
                for (int i = 0; i < PIPELINE_STAGES; i++)
                {
                    init_semaphore(inputs_arrived(s, i), 0, 2);
                    init_semaphore(inputs_finished(s, i), 0, 1);
                    init_semaphore(outputs_arrived(s, i), 0, 1);
                }
                init_semaphore(outputs_shared(s), 0, Config::NUM_CONSUMER_WARPS);
                init_semaphore(rope_arrived(s), 0, 1);
                return 3 * PIPELINE_STAGES + 2;
            }
        };

        struct loader
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                static_assert(Config::SCRATCH_BYTES >= 2 * sizeof(rope_vec), "Not enough scratch space for rope table");

                s.warp_finish_page(12, Config::NUM_CONSUMER_WARPS); // release the unused page immediately
                parsed_instruction inst{s};
                int laneid = warp::laneid();

                if (laneid == 0)
                { // load weights
                    uint32_t phasebits = 0xFFFF0000;
                    for (int i = 0; i < inst.num_iters; i++)
                    {
                        int stage = i % PIPELINE_STAGES;
                        int weight_page = get_weight_page(s, stage);
                        weight_tile &weight = *reinterpret_cast<weight_tile *>(s.pages[weight_page].data);

                        wait(inputs_finished(s, stage), get_phasebit<1>(phasebits, stage));
                        update_phasebit<1>(phasebits, stage);
                        tma::expect(inputs_arrived(s, stage), weight);

                        if (i < PIPELINE_STAGES)
                        {
                            s.wait_page_ready(weight_page);
                            s.wait_page_ready(weight_page + 1);
                        }
                        tma::load_async(weight, g.qkv_weights, {inst.layer_idx, inst.block_idx, i}, inputs_arrived(s, stage));
                    }
                    for (int i = 0; i < PIPELINE_STAGES; i++)
                    {
                        int stage = (inst.num_iters + i) % PIPELINE_STAGES;
                        wait(outputs_arrived(s, stage), 0);
                        int weight_page = get_weight_page(s, stage);
                        s.warp_finish_page(weight_page, Config::NUM_CONSUMER_WARPS);
                        s.warp_finish_page(weight_page + 1, Config::NUM_CONSUMER_WARPS);
                    }
                }
                else if (laneid == 1)
                { // load activations
                    // TODO: match the barrier with the pre-attention rmsnorm
                    // while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_RMS_NORM - 1, 0}] < 1)
                    //     __nanosleep(20);

                    uint32_t phasebits = 0xFFFF0000;
                    for (int i = 0; i < inst.num_iters; i++)
                    {
                        int stage = i % PIPELINE_STAGES;
                        int activation_page = get_activation_page(s, stage);
                        activation_tile &activation = *reinterpret_cast<activation_tile *>(s.pages[activation_page].data);

                        wait(inputs_finished(s, stage), get_phasebit<1>(phasebits, stage));
                        update_phasebit<1>(phasebits, stage);
                        tma::expect(inputs_arrived(s, stage), activation);

                        if (i < PIPELINE_STAGES)
                        {
                            s.wait_page_ready(activation_page);
                            s.wait_page_ready(activation_page + 1);
                        }
                        tma::load_async(activation, g.rms_rope_intermediates, {0, i}, inputs_arrived(s, stage));
                    }
                    for (int i = 0; i < PIPELINE_STAGES - 1; i++)
                    { // last stage is used as output page
                        int stage = (inst.num_iters + i) % PIPELINE_STAGES;
                        wait(outputs_arrived(s, stage), 0);
                        int activation_page = get_activation_page(s, stage);
                        s.warp_finish_page(activation_page, Config::NUM_CONSUMER_WARPS);
                        s.warp_finish_page(activation_page + 1, Config::NUM_CONSUMER_WARPS);
                    }
                }
                else if (laneid == 2 && inst.block_idx < V_BLOCK_START)
                { // load rope table
                    rope_vec &rope_cos = *reinterpret_cast<rope_vec *>(s.scratch());
                    rope_vec &rope_sin = *reinterpret_cast<rope_vec *>(reinterpret_cast<char *>(s.scratch()) + sizeof(rope_vec));
                    tma::expect(rope_arrived(s), rope_cos, rope_sin);
                    tma::load_async(rope_cos, g.rope_cos, {(int)g.pos_id, 0}, rope_arrived(s));
                    tma::load_async(rope_sin, g.rope_sin, {(int)g.pos_id, 0}, rope_arrived(s));
                }
            }
        };

        struct launcher
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};
                int laneid = warp::laneid();
                uint32_t phasebits = 0xFFFF0000;

                s.wait_tensor_ready();

                if (laneid == 0)
                {
                    for (int i = 0; i < inst.num_iters; i++)
                    {
                        int stage = i % PIPELINE_STAGES;
                        wait(inputs_arrived(s, stage), get_phasebit<0>(phasebits, stage));
                        update_phasebit<0>(phasebits, stage);
                        auto accumulator = s.tensor_alloc.template allocate<tt<float, Globals::batch_size, Globals::head_dim>>(stage * Globals::head_dim);
                        weight_tile &weight = *reinterpret_cast<weight_tile *>(s.pages[get_weight_page(s, stage)].data);
                        activation_tile &activation = *reinterpret_cast<activation_tile *>(s.pages[get_activation_page(s, stage)].data);
                        if (i < PIPELINE_STAGES)
                            mm<transpose::N, transpose::T>(accumulator, activation, weight, inputs_finished(s, stage));
                        else if (i >= inst.num_iters - PIPELINE_STAGES)
                            mma<transpose::N, transpose::T>(accumulator, activation, weight, outputs_arrived(s, stage));
                        else
                            mma<transpose::N, transpose::T>(accumulator, activation, weight, inputs_finished(s, stage));
                    }
                }
            }
        };

        struct consumer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                static_assert(Config::NUM_CONSUMER_WARPS == 8, "NUM_CONSUMER_WARPS must be 8");
                using consumer = group<Config::NUM_CONSUMER_WARPS>;

                parsed_instruction inst{s};
                rt_fl<Globals::batch_size / Config::NUM_CONSUMER_WARPS, Globals::head_dim> output_fl;
                consumer::zero(output_fl);

                for (int i = 0; i < PIPELINE_STAGES; i++)
                {
                    int stage = (inst.num_iters + i) % PIPELINE_STAGES;
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, Globals::batch_size, Globals::head_dim>>(stage * Globals::head_dim);
                    wait(outputs_arrived(s, stage), 0);
                    rt_fl<Globals::batch_size / Config::NUM_CONSUMER_WARPS, Globals::head_dim> acc_fl;
                    consumer::load_async(acc_fl, accumulator);
                    tensor_load_wait();
                    __syncwarp();
                    consumer::add(output_fl, output_fl, acc_fl);
                }
                warp::arrive(s.tensor_finished);

                if (inst.block_idx < V_BLOCK_START)
                {
                    // Load RoPE parameters
                    wait(rope_arrived(s), 0);
                    rope_vec &rope_cos = *reinterpret_cast<rope_vec *>(s.scratch());
                    rope_vec &rope_sin = *reinterpret_cast<rope_vec *>(reinterpret_cast<char *>(s.scratch()) + sizeof(rope_vec));
                    row_vec<rt_fl<Globals::batch_size / Config::NUM_CONSUMER_WARPS, Globals::head_dim>> rope_cos_rv;
                    row_vec<rt_fl<Globals::batch_size / Config::NUM_CONSUMER_WARPS, Globals::head_dim>> rope_sin_rv;
                    warp::load(rope_cos_rv, rope_cos);
                    warp::load(rope_sin_rv, rope_sin);

                    // Apply RoPE
                    rt_fl<Globals::batch_size / Config::NUM_CONSUMER_WARPS, Globals::head_dim> output_rotated;

                    // Rotate
                    static_assert(output_fl.width >= 2 && output_fl.width % 2 == 0);
#pragma unroll
                    for (int i = 0; i < output_fl.height; i++)
                    {
#pragma unroll
                        for (int j = 0; j < output_fl.width / 2; j++)
                        {
#pragma unroll
                            for (int k = 0; k < output_fl.packed_per_tile; k++)
                            { // -x2
                                output_rotated.tiles[i][j].data[k] = output_fl.tiles[i][j + output_fl.width / 2].data[k];
                                output_rotated.tiles[i][j].data[k].x *= -1;
                                output_rotated.tiles[i][j].data[k].y *= -1;
                            }
                        }
#pragma unroll
                        for (int j = 0; j < output_fl.width / 2; j++)
                        {
#pragma unroll
                            for (int k = 0; k < output_fl.packed_per_tile; k++)
                            { // x1
                                output_rotated.tiles[i][j + output_fl.width / 2].data[k] = output_fl.tiles[i][j].data[k];
                            }
                        }
                    }

                    warp::mul_col(output_fl, output_fl, rope_cos_rv);
                    warp::mul_col(output_rotated, output_rotated, rope_sin_rv);
                    consumer::add(output_fl, output_fl, output_rotated);
                }

                rt_bf<Globals::batch_size / Config::NUM_CONSUMER_WARPS, Globals::head_dim> output_bf;
                consumer::copy(output_bf, output_fl);

                int last_stage = (inst.num_iters - 1) % PIPELINE_STAGES;
                int output_page = get_activation_page(s, last_stage);
                output_tile &output = *reinterpret_cast<output_tile *>(s.pages[output_page].data);
                consumer::store(output, output_bf);
                __syncwarp();
                warp::arrive(outputs_shared(s));
            }
        };

        struct storer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};
                int laneid = warp::laneid();

                wait(outputs_shared(s), 0);
                int output_page = get_activation_page(s, (inst.num_iters - 1) % PIPELINE_STAGES);
                output_tile &output = *reinterpret_cast<output_tile *>(s.pages[output_page].data);

                if (laneid == 0)
                {
                    if (inst.block_idx < K_BLOCK_START)
                        tma::store_async(g.q_post_rope, output, {0, inst.block_idx});
                    else if (inst.block_idx < V_BLOCK_START)
                        tma::store_async<dim::BATCH, cache_policy::NORMAL>(g.k_cache, output, {inst.layer_idx * (int)Globals::batch_size, (int)g.pos_id, inst.block_idx - K_BLOCK_START, 0});
                    else
                        tma::store_async<dim::BATCH, cache_policy::NORMAL>(g.v_cache, output, {inst.layer_idx * (int)Globals::batch_size, (int)g.pos_id, inst.block_idx - V_BLOCK_START, 0});
                    tma::store_async_wait();
                    s.finish_page(output_page, Config::NUM_CONSUMER_WARPS);
                    s.finish_page(output_page + 1, Config::NUM_CONSUMER_WARPS);
                    atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, inst.block_idx}], 1);
                }
            }
        };
    };

}
