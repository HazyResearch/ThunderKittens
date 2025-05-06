#include "llama.cuh"
#include "utils.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_1b_globals;

    template <typename Config, typename Globals>
    struct qkv_rope_append
    {
        static constexpr int opcode = OPCODE_QKV_MatVecRopeAppend; // Op index within the layer -- controls which barrier to listen to.
        static constexpr int PIPELINE_STAGES = 3;
        static constexpr int NUM_MATMUL_PAGES = 9;

        static constexpr int PAGE_ACTIVATION = 0;
        static constexpr int PAGE_ROPE = PAGE_ACTIVATION + 1;
        static constexpr int PAGE_WEIGHT_START = PAGE_ROPE + 1;
        static constexpr int PAGE_COUNT = PAGE_WEIGHT_START + NUM_WEIGHT_PAGES;

        // TODO: Ensure matvec_block_size is 128
        static constexpr int K_BLK_START = 8192 / Globals::matvec_block_size;
        static constexpr int V_BLK_START = 9216 / Globals::matvec_block_size;

        // 8192 / 16 = 512
        static constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        using rope_sv = sv_fl<128>;
        using activation_tile = st_bf<64, 128>;    // 16KB
        using weight_tile = st_bf<128, 128>;   // 32KB
        using out_tile = st_bf<64, 128>;    // 16KB
        
        struct parsed_instruction
        {
            int output_row;
            int output_col;
            int iters;
            int layer_idx;
            int qkv_block_idx; // can get 
            int page_idx; // What page idx to write to 
            int slot_idx; // which slot in page to use 

            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                output_row = instruction[1];
                output_col = instruction[2];
                layer_idx = instruction[3];
                qkv_block_idx = instruction[4];
                page_idx = instruction[5];
                slot_idx = instruction[6];
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        // Semaphores
        __device__ static inline semaphore &inputs_arrived(state<config> &s, int id) {
            return s.semaphores()[id];
        }
        __device__ static inline semaphore &inputs_finished(state<config> &s, int id) {
            return s.semaphores()[id+PIPELINE_STAGES];
        }
        __device__ static inline semaphore &outputs_arrived(state<config> &s, int id) {
            return s.semaphores()[id+PIPELINE_STAGES*2];
        }
        __device__ static inline semaphore &outputs_shared(state<config> &s, int id) {
            return s.semaphores()[id+PIPELINE_STAGES*2+2];
        }
        __device__ static inline semaphore &rope_cos_arrived(state<Config> &s) { return s.semaphores()[NUM_MATMUL_PAGES]; }
        __device__ static inline semaphore &rope_sin_arrived(state<Config> &s) { return s.semaphores()[NUM_MATMUL_PAGES + 1]; }

        __device__ static inline int get_activation_page(state<config> &s, int stage, int offset) {
            return stage*4 + offset;
        }
        __device__ static inline int get_weight_page(state<config> &s, int stage) {
            return stage*4 + 2;
        }
        __device__ static inline rope_sv &get_rope_sin_smem(state<Config> &s) { 
            int pid = s.pid(PAGE_ROPE);
            char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
            return *reinterpret_cast<rope_sv *>(page_base_ptr);
        }
        __device__ static inline rope_sv &get_rope_cos_smem(state<Config> &s) { 
            int pid = s.pid(PAGE_ROPE);
            char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
            return *reinterpret_cast<rope_sv *>(page_base_ptr + sizeof(rope_sv));
        }

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
            {
                // TODO: Need to correct order
                // unused pages, then activation, then rms scale, then weights, then rope cos, then rope sin
                int ret_order[13] = {7, 8, 9, 10, 11, 12, PAGE_ACTIVATION, PAGE_WEIGHT_START, PAGE_WEIGHT_START + 1, PAGE_WEIGHT_START + 2, PAGE_WEIGHT_START + 3, PAGE_ROPE_COS, PAGE_ROPE_SIN};
                return ret_order[query];
            }
            static __device__ int init_semaphores(const Globals &g, state<Config> &s)
            {
                for(int i = 0; i < PIPELINE_STAGES; i++) {
                    init_semaphore(inputs_arrived(s, i), 1);
                    init_semaphore(inputs_finished(s, i), 2);
                }
                for(int i = 0; i < 2; i++) {
                    init_semaphore(outputs_arrived(s, i), 1);
                    init_semaphore(outputs_shared(s, i), 1);
                }

                init_semaphore(rope_cos_arrived(s), 1);
                init_semaphore(rope_sin_arrived(s), 1);
                return (PIPELINE_STAGES * 2) + (2 * 2) + 2;
            }
        };
        struct loader
        {
            static __device__ void run(const globals &g, state<config> &s) {
                parsed_instruction inst{s};
                uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
                int pipeline_stage = 0;
                for(int i = 0; i < inst.iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                    wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                    // if (laneid() == 0) printf(BLUE_TEXT "Loader Passed stage %d\n" RESET_TEXT, pipeline_stage);
                    warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), sizeof(activation_tile)*2 + sizeof(weight_tile));
                    if(laneid() < 2) {
                        int activation_page = get_activation_page(s, pipeline_stage, laneid());
                        if(i < PIPELINE_STAGES) {
                            s.wait_page_ready(activation_page);
                        }
                        activation_tile &a = *reinterpret_cast<activation_tile *>(s.pages[activation_page].data);
                        tma::load_async(a, g.A, {inst.row + laneid(), i}, inputs_arrived(s, pipeline_stage));
                    } else if (laneid() == 2) {
                        int weight_page = get_weight_page(s, pipeline_stage);
                        if(i < PIPELINE_STAGES) {
                            s.wait_page_ready(weight_page);
                            s.wait_page_ready(weight_page+1);
                        }
                        weight_tile &b = *reinterpret_cast<weight_tile *>(s.pages[weight_page].data);
                        tma::load_async(b, g.B, {inst.col, i}, inputs_arrived(s, pipeline_stage));
                    }
                    update_phasebit<1>(semaphore_bitfield, pipeline_stage);
                }
                warp::sync();
                // if (laneid() == 0) printf(BLUE_TEXT "Loader finished issuing loads\n" RESET_TEXT);
                // Now load rope 
                if (laneid() == 0)
                {
                    // Rope cos
                    auto cos_page_id = get_rope_cos_page(s);
                    s.wait_page_ready(cos_page_id);
                    auto &rope_cos = reinterpret_cast<rope_sv &>(s.pages[cos_page_id]);
                    s.record(TEVENT_TRIPLES_START + 5);
                    tma::expect(rope_cos_arrived(s), rope_cos);
                    tma::load_async(rope_cos, g.rope_cos, {0, 0, static_cast<int>(g.pos_id), inst.qkv_block_idx % 4}, rope_cos_arrived(s));

                    // Rope sin
                    auto sin_page_id = get_rope_sin_page(s);
                    s.wait_page_ready(sin_page_id);
                    auto &rope_sin = reinterpret_cast<rope_sv &>(s.pages[sin_page_id]);
                    s.record(TEVENT_TRIPLES_START + 6);
                    tma::expect(rope_sin_arrived(s), rope_sin);
                    tma::load_async(rope_sin, g.rope_sin, {0, 0, static_cast<int>(g.pos_id), inst.qkv_block_idx % 4}, rope_sin_arrived(s));

                }
                warp::sync(); 

                if(laneid() >= 28) {
                    for(int i = 0; i < PIPELINE_STAGES-1; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                        wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                        int release_pid = pipeline_stage*4 + laneid() - 28;
                        s.finish_page(release_pid, config::NUM_CONSUMER_WARPS);
                    }
                }
            }
        };
        struct launcher 
        { // launches mma's
            static __device__ void run(const globals &g, state<config> &s) {
    
                parsed_instruction inst{s};
                uint32_t semaphore_bitfield = 0xFFFF0000;
                int pipeline_stage = 0;
    
                wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                // if (laneid() == 0) printf(GREEN_TEXT "Launcher Passed stage %d\n" RESET_TEXT, pipeline_stage);
                s.wait_tensor_ready();
                if(laneid() < 2) {
                    // auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(laneid(), 0);
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(0, laneid() * 128);
                    activation_tile &a = *reinterpret_cast<activation_tile *>(s.pages[get_activation_page(s, pipeline_stage, laneid())].data);
                    weight_tile &b = *reinterpret_cast<weight_tile *>(s.pages[get_weight_page(s, pipeline_stage)].data);
                    mm<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
                }
                // if (laneid() == 0) printf(GREEN_TEXT "Finished first mma\n" RESET_TEXT);
                update_phasebit<0>(semaphore_bitfield, pipeline_stage);
                pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage);
                
                for(int i = 1; i < inst.iters-1; i++, update_phasebit<0>(semaphore_bitfield, pipeline_stage), pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                    wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                    // if (laneid() == 0) printf(GREEN_TEXT "Launcher Passed stage %d\n" RESET_TEXT, pipeline_stage);
                    if(laneid() < 2) {
                        // auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(laneid(), 0);
                        auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(0, laneid() * 128);
                        activation_tile &a = *reinterpret_cast<activation_tile *>(s.pages[get_activation_page(s, pipeline_stage, laneid())].data);
                        weight_tile &b = *reinterpret_cast<weight_tile *>(s.pages[get_weight_page(s, pipeline_stage)].data);
                        mma<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
                    }
                }
                
                wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                // if (laneid() == 0) printf(GREEN_TEXT "Launcher Passed stage %d\n" RESET_TEXT, pipeline_stage);
    
                if(laneid() < 2) {
                    // auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(laneid(), 0);
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(0, laneid() * 128);
                    activation_tile &a = *reinterpret_cast<activation_tile *>(s.pages[get_activation_page(s, pipeline_stage, laneid())].data);
                    weight_tile &b = *reinterpret_cast<weight_tile *>(s.pages[get_weight_page(s, pipeline_stage)].data);
                    mma<transpose::N, transpose::T>(accumulator, a, b, outputs_arrived(s, laneid()));
                }
                warp::sync();
                // if (laneid() == 0) printf(RED_TEXT "Finished launcher\n" RESET_TEXT);
            }
        };
        struct consumer 
        {
            static __device__ void run(const globals &g, state<config> &s) {

                parsed_instruction inst{s};
                int groupid = warpgroup::groupid();
                if (groupid < 2) // safeguard for differing num consumers
                {
                    wait(outputs_arrived(s, groupid), 0);
        
                    // auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(groupid, 0);
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 64, 128>>(0, groupid * 128);
                    
                    rt_fl<16, 128> acc_rt;
                    rt_bf<16, 128> acc_bf16;
                    
                    warpgroup::load_async(acc_rt, accumulator);
                    warp::copy(acc_bf16, acc_rt);
                    tensor_load_wait();
                    warp::arrive(s.tensor_finished);
                    
                    int store_page_id = get_store_page(s, inst, groupid);
                    out_tile &store_buffer = *reinterpret_cast<out_tile *>(s.pages[store_page_id].data);
                    warpgroup::store(store_buffer, acc_bf16);
                    warpgroup::sync(groupid);
                    warpgroup::arrive(outputs_shared(s, groupid));
                }

                // RoPE
                if (warpid() == 0)
                { // only a single warp needed from here!

                    // even for V, we need to cast from float to bf16
                    rope_sv &qkv_proj_smem = *reinterpret_cast<rope_sv *>(s.scratch());
                    sv_bf<128> &qkv_proj_smem_bf = *reinterpret_cast<sv_bf<128> *>(s.scratch());
                    warp::load(qkv_proj, qkv_proj_smem);

                    warp::sync();

                    int rope_cos_page = get_rope_cos_page(s);
                    int rope_sin_page = get_rope_sin_page(s);

                    if (inst.qkv_block_idx < V_BLK_START)
                    { // only Q & K need RoPE

                        rope_sv &rope_cos_smem = reinterpret_cast<rope_sv &>(s.pages[rope_cos_page]);
                        wait(rope_cos_arrived(s), 0);
                        if (laneid() == 0)
                        {
                            s.record(TEVENT_TRIPLES_END + 5);
                            s.record(TEVENT_CONSUMER_START + 48);
                        }
                        warp::load(rope_cos, rope_cos_smem);
                        warp::arrive(s.page_finished[rope_cos_page], Config::NUM_CONSUMER_WARPS);

                        rope_sv &rope_sin_smem = reinterpret_cast<rope_sv &>(s.pages[rope_sin_page]);
                        wait(rope_sin_arrived(s), 0);
                        if (laneid() == 0)
                        {
                            s.record(TEVENT_TRIPLES_END + 6);
                            s.record(TEVENT_CONSUMER_START + 49);
                        }
                        warp::load(rope_sin, rope_sin_smem);
                        warp::arrive(s.page_finished[rope_sin_page], Config::NUM_CONSUMER_WARPS);

                        // Fetch the neighbor values
                        int mod = (laneid() & 0b1) ? -1 : 1; // 1 for even, -1 for odd
                        warp::sync();
                        float pair_val = __shfl_sync(MASK_ALL, qkv_proj[0][0], laneid() + mod);

                        // Compute RoPE in-place
                        if (laneid() < 16)
                            // will clean this up later
                            qkv_proj[0][0] = float(qkv_proj[0][0]) * rope_cos[0][0] + float(-1 * mod) * float(pair_val) * rope_sin[0][0];
                    }
                    else
                    {
                        wait(rope_cos_arrived(s), 0);
                        warp::arrive(s.page_finished[rope_cos_page], Config::NUM_CONSUMER_WARPS);

                        wait(rope_sin_arrived(s), 0);
                        warp::arrive(s.page_finished[rope_sin_page], Config::NUM_CONSUMER_WARPS);
                    }

                    // Store back to the scratch
                    warp::store(qkv_proj_smem_bf, qkv_proj);
                    warp::sync();

                    warp::arrive(outputs_arrived(s));
                    if (kittens::group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
                    {
                        s.record(TEVENT_CONSUMER_END);
                    }
                }
            }
        };
        
        struct storer
        {
            // Uses 4 full pages for outputs.
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_TRIPLES_STORE_START);
                }

                parsed_instruction inst{s};

                if (warp::laneid() == 0)
                {
                    sv_bf<16> &qkv_proj_smem = *reinterpret_cast<sv_bf<16> *>(s.scratch());
                    wait(outputs_arrived(s), 0);
                    s.record(TEVENT_TRIPLES_OUTPUT_READY);

                    if (inst.qkv_block_idx < K_BLK_START)
                    { // Q
                        tma::store_async<cache_policy::NORMAL>(g.q_post_rope, qkv_proj_smem, {0, 0, 0, inst.qkv_block_idx});
                    }
                    else if (inst.qkv_block_idx < V_BLK_START)
                    { // K
                        int base_index = (inst.qkv_block_idx - K_BLK_START) * Globals::matvec_block_size;
                        int head_idx = base_index / Globals::head_dim;
                        int dim_idx = (base_index % Globals::head_dim) / Globals::matvec_block_size;
                        tma::store_async<cache_policy::NORMAL>(g.k_cache, qkv_proj_smem, {inst.page_idx, inst.slot_idx, head_idx, dim_idx});
                    }
                    else
                    { // V
                        int base_index = (inst.qkv_block_idx - V_BLK_START) * Globals::matvec_block_size;
                        int head_idx = base_index / Globals::head_dim;
                        int dim_idx = (base_index % Globals::head_dim) / Globals::matvec_block_size;
                        tma::store_async<cache_policy::NORMAL>(g.v_cache, qkv_proj_smem, {inst.page_idx, inst.slot_idx, head_idx, dim_idx});
                    }

                    tma::store_async_wait(); // not just read wait! full wait! must be visible in global!
                    s.record(126);
                }

                warp::sync();
                asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.

                if (warp::laneid() == 0)
                    atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, inst.qkv_block_idx / 4}], 1);

                warp::sync();
                if (laneid() == 0)
                    s.record(TEVENT_STORE_END);
            }
        };
    };
}
