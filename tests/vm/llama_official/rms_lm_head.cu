#include "llama.cuh"
#include "utils.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_1b_globals;

    template <typename Config, typename Globals>
    struct rms_lm_head
    {
        static constexpr int opcode = OPCODE_RMS_LM_Head; // Op index within the layer -- controls which barrier to listen to.
        static constexpr int INPUT_PIPELINE_STAGES = 3;
        static constexpr int STAGE_PAGES = 4;
        static constexpr int OUTPUT_PIPELINE_STAGES = 2;

        static constexpr int SEM_COUNT = (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2 + 1;

        static constexpr int EXPECTED_ARRIVAL_COUNT = 512;

        static constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        struct parsed_instruction
        {
            int start_block_idx, end_block_idx, iters;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                start_block_idx = instruction[1];
                end_block_idx   = instruction[2];
                iters = end_block_idx - start_block_idx;
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        // Semaphores
        __device__ static inline semaphore &weights_arrived(state<Config> &s, int stage) { return s.semaphores()[stage]; }
        __device__ static inline semaphore &weights_finished(state<Config> &s, int stage) { return s.semaphores()[INPUT_PIPELINE_STAGES + stage]; }
        __device__ static inline semaphore &outputs_arrived(state<Config> &s, int stage) { return s.semaphores()[2*INPUT_PIPELINE_STAGES + stage]; }
        __device__ static inline semaphore &outputs_finished(state<Config> &s, int stage) { return s.semaphores()[2*INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES + stage]; }
        __device__ static inline semaphore &activations_rms_scale_arrived(state<Config> &s) { return s.semaphores()[2*(INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES)]; }

        // Pages (very naive for now, no fine-grained usage)
        __device__ static inline int get_rms_scale_activation_page(state<Config> &s) { return s.pid(0); }
        __device__ static inline int get_weight_page(state<Config> &s, int stage, int offset) { return s.pid(1 + stage * STAGE_PAGES + offset); }

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
                parsed_instruction inst{instruction};
                // unused pages, then activation, then weights
                if(inst.iters%3 == 1) {
                    int ret_order[13] = {0,5,6,7,8,9,10,11,12,1,2,3,4};
                    return ret_order[query];
                }
                else if(inst.iters%3 == 2) {
                    int ret_order[13] = {0,9,10,11,12,1,2,3,4,5,6,7,8};
                    return ret_order[query];
                }
                else {
                    int ret_order[13] = {0,1,2,3,4,5,6,7,8,9,10,11,12};
                    return ret_order[query];
                }
            }
            static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
                for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
                    init_semaphore(weights_arrived(s, i), 1);
                    init_semaphore(weights_finished(s, i), Config::NUM_CONSUMER_WARPS);
                }
                for(int i = 0; i < OUTPUT_PIPELINE_STAGES; i++) {
                    init_semaphore(outputs_arrived(s, i), Config::NUM_CONSUMER_WARPS);
                    init_semaphore(outputs_finished(s, i), 1);
                }
                init_semaphore(activations_rms_scale_arrived(s), 1); // get rms scale, too.
                return SEM_COUNT;
            }
        };
        struct loader
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_LOADER_START);
                }
                parsed_instruction inst{s};
                // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
                ((uint64_t *)s.scratch())[laneid()] = 0;
                warp::sync(); // done, now we can proceed to other things.

                if(laneid() == 0) {
                    int input_stage = 0;
                    for(int iter = 0; iter < inst.iters; iter++) {
                        wait(weights_finished(s, input_stage), (iter%(2*INPUT_PIPELINE_STAGES))<INPUT_PIPELINE_STAGES);

                        int block_idx = inst.start_block_idx + iter;

                        warp::tma::expect_bytes(weights_arrived(s, input_stage), sizeof(bf16) * 2048 * 16);
                        #pragma unroll
                        for(int i = 0; i < 4; i++) {
                            int weight_page = get_weight_page(s, input_stage, i);
                            if(iter < INPUT_PIPELINE_STAGES) s.wait_page_ready(weight_page);
                            auto &weight_chunk = reinterpret_cast<st_bf<16, 512> &>(s.pages[weight_page]);
                            tma::load_async(weight_chunk, g.lm_head_weights, {block_idx, i}, weights_arrived(s, input_stage));
                        }

                        input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
                    }
                }
                else if(laneid() >= inst.iters*4 && laneid() < INPUT_PIPELINE_STAGES*4) {
                    int stage = laneid()/4, offset = laneid()%4;
                    auto pid = get_weight_page(s, stage, offset);
                    s.wait_page_ready(pid);
                    s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
                }
            }
        };
        struct launcher
        {
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
                    auto &rms_scale = *reinterpret_cast<sv_bf<2048> *>(s.pages[rms_scale_activation_page].ptr());
                    tma::expect(activations_rms_scale_arrived(s), activations, rms_scale);
                    tma::load_async(rms_scale, g.lm_head_norm_weights, {}, activations_rms_scale_arrived(s));

                    // Activation
                    s.record(TEVENT_AT_GMEM_WAIT);
                    while (*(volatile int *)&g.Bar[{globals::num_layers - 1, OPCODE_DownProjResidual - 1, 0}] < EXPECTED_ARRIVAL_COUNT)
                    {
                        __nanosleep(20);
                    }
                    s.record(TEVENT_DONE_GMEM_WAIT);
                    tma::load_async(activations, g.hidden_states, {}, activations_rms_scale_arrived(s));
                }
            }
        };
        struct consumer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {

                using float_rt_t = rt_fl<16, REDUCTION_DIM_PER_WARP>;
                using float_rv_t = rv_fl<16>;

                // Setup
                parsed_instruction inst{s};

                static_assert(Config::NUM_CONSUMER_WARPS % STAGE_PAGES == 0, "NUM_CONSUMER_WARPS must be divisible by STAGE_PAGES");
                constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / STAGE_PAGES;

                int page_index = warpid() / WARPS_PER_PAGE;

                wait(activations_rms_scale_arrived(s), 0);
                
                auto rms_scale_smem   = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP>*>(s.pages[get_rms_scale_activation_page(s)].ptr());
                auto activations_smem = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP>*>(s.pages[get_rms_scale_activation_page(s)].ptr(sizeof(sv_bf<2048>)));

                auto activations_vec = rms_norm<Config>(rms_scale_smem[warpid()], activations_smem[warpid()], g.rms_norm_eps, (void*)((uint8_t*)s.scratch()+(64 * 12)));
                
                // release the activation page
                warp::sync();
                s.warp_finish_page(get_rms_scale_activation_page(s), 1);

                int input_stage = 0, output_stage = 0;
                for(int i = 0; i < inst.iters; i++) {

                    int weight_page = get_weight_page(s, input_stage, page_index);
                    wait(weights_arrived(s, input_stage), (i%(2*INPUT_PIPELINE_STAGES))>=INPUT_PIPELINE_STAGES);
                    wait(outputs_finished(s, output_stage), (i%(2*OUTPUT_PIPELINE_STAGES))<OUTPUT_PIPELINE_STAGES);
                    st_bf<16, REDUCTION_DIM_PER_WARP> &weights = reinterpret_cast<st_bf<16, REDUCTION_DIM_PER_WARP> *>(s.pages[weight_page].ptr())[warpid() % WARPS_PER_PAGE];
                    sv_fl<16> &out_smem = *reinterpret_cast<sv_fl<16> *>(s.scratch());

                    matvec(out_smem, weights, activations_vec);

                    warp::sync();
                    warp::arrive(outputs_arrived(s, output_stage));

                    if(i >= inst.iters - 3) {
                        // Release pages.
                        #pragma unroll
                        for (int j = 0; j < STAGE_PAGES; j++) {
                            s.warp_finish_page(get_weight_page(s, input_stage, j), 1);
                        }
                    }

                    group<Config::NUM_CONSUMER_WARPS>::sync(0);

                    input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
                    output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
                }

                if (laneid() == 0)
                {
                    s.record(TEVENT_CONSUMER_START + 32 + warpid());
                }
            }
        };
        struct storer
        {
            static __device__ void run(const Globals &g, state<Config> &s)
            {
                parsed_instruction inst{s};

                sv_fl<16> &logits_smem = *reinterpret_cast<sv_fl<16> *>(s.scratch());
                sv_bf<16> &logits_smem_bf = *reinterpret_cast<sv_bf<16> *>(s.scratch());

                int output_stage = 0;
                for(int i = 0; i < inst.iters; i++) {
                    int block_idx = inst.start_block_idx + i; 

                    wait(outputs_arrived(s, output_stage), (i%(2*OUTPUT_PIPELINE_STAGES))>=OUTPUT_PIPELINE_STAGES);

                    rv_fl<16> logits_rv;
                    warp::load(logits_rv, logits_smem);
                    warp::sync();
                    warp::store(logits_smem_bf, logits_rv);
                    warp::sync();

                    if (warp::laneid() == 0)
                    {
                        s.record(TEVENT_OUTPUT_READY);

                        tma::store_async<cache_policy::NORMAL>(g.logits, logits_smem_bf, {0, 0, 0, block_idx});

                        tma::store_async_read_wait(); // not just read wait! full wait! must be visible in global!
                    }

                    warp::arrive(outputs_finished(s, output_stage));
                    output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
                }
            }
        };
    };
}
