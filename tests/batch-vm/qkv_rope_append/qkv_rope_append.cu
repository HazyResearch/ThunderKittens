#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    using globals = llama_70b_globals;

    template <typename Config, typename Globals>
    struct qkv_rope_append
    {
        static constexpr int opcode = OPCODE_QKV_RopeAppend; // Op index within the layer -- controls which barrier to listen to.
        static constexpr int PIPELINE_STAGES = 3;
        static constexpr int NUM_MATMUL_SEMS = 10;

        static constexpr int ROPE_PAGE = 12;

        static constexpr int K_BLK_START = 8192 / Globals::matmul_out_block_size;
        static constexpr int V_BLK_START = 9216 / Globals::matmul_out_block_size;

        using routing_sv = sv<int, 256>;
        using rope_st = st_fl<128, 64>;
        using rope_subtile = st_subtile<rope_st, 16, 64>;
        using activation_tile = st_bf<64, 128>;    // 16KB
        using weight_tile = st_bf<128, 128>;   // 32KB
        using out_tile = st_bf<64, 128>;    // 16KB
        using kv_row_vec = out_tile::row_vec;
        
        struct parsed_instruction
        {
            int layer;
            int row; // NEEDS TO BE IN OFFSETS OF 64!!
            int col;
            int iters;

            __device__ inline parsed_instruction(typename Config::instruction_t &instruction)
            {
                layer = instruction[1];
                row = instruction[2];
                col = instruction[3];
                iters = instruction[4];
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        // Semaphores
        __device__ static inline semaphore &inputs_arrived(state<config> &s, int id) {
            return s.semaphores()[id]; // 0, 1, 2
        }
        __device__ static inline semaphore &inputs_finished(state<config> &s, int id) {
            return s.semaphores()[id+PIPELINE_STAGES]; // 3, 4, 5
        }
        __device__ static inline semaphore &outputs_arrived(state<config> &s, int id) {
            return s.semaphores()[id+PIPELINE_STAGES*2]; // 6, 7
        }
        __device__ static inline semaphore &outputs_shared(state<config> &s, int id) {
            return s.semaphores()[id+PIPELINE_STAGES*2+2]; // 8, 9
        }
        __device__ static inline semaphore &rope_cos_arrived(state<Config> &s) { return s.semaphores()[NUM_MATMUL_SEMS]; } // 10
        __device__ static inline semaphore &rope_sin_arrived(state<Config> &s) { return s.semaphores()[NUM_MATMUL_SEMS + 1]; } // 11    
        __device__ static inline semaphore &routing_page_ready(state<Config> &s) { return s.semaphores()[NUM_MATMUL_SEMS + 2]; } // 12
        /*
        Activation Pages Cycle Through:
        Stage 0:
            - 0, 1
        Stage 1:
            - 4, 5
        Stage 2:
            - 8, 9
        
        Weight Pages Cycle Through:
        Stage 0:
            - 2 AND 3
        Stage 1:
            - 6 AND 7
        Stage 2:
            - 10 AND 11
        

        Store Page is EITHER:
        Stage 0:
            - 0 AND 1
        Stage 1:
            - 4 AND 5
        Stage 2:
            - 8 AND 9

        We need TWO store pages one for each 64 x 128 tensor
        
        Logic of get_store_page is that...
        inst.iters % PIPELINE_STAGES gives the stage we are on when doing last iteration

        THUS => if we move 2 after that we're on the first page that will free up

        For (last stage + 2)%PIPELINE_STAGES:
        - first two pages are used for store
        - third page is used for routing
        - fourth page is unused
        
        */
        __device__ static inline int get_activation_page(state<config> &s, int stage, int offset) {
            return stage*4 + offset;
        }
        __device__ static inline int get_weight_page(state<config> &s, int stage) {
            return stage*4 + 2;
        }
        __device__ static inline int get_store_page(state<config> &s, parsed_instruction &inst, int offset) {
            return ((inst.iters+1)%PIPELINE_STAGES)*4 + offset;
        }
        __device__ static inline int get_routing_page(state<config> &s, parsed_instruction &inst) {
            return ((inst.iters+1)%PIPELINE_STAGES)*4 + 2;
        }
        __device__ static inline int get_cos_page(state<config> &s, parsed_instruction &inst) {
            return ((inst.iters+2)%PIPELINE_STAGES)*4;
        }
        __device__ static inline int get_sin_page(state<config> &s, parsed_instruction &inst) {
            return ((inst.iters+2)%PIPELINE_STAGES)*4 + 2;
        }
        __device__ static inline out_tile &get_output_buffer(state<config> &s, parsed_instruction &inst, int store_page) {
            char *page_base_ptr = reinterpret_cast<char *>(s.pages[store_page].data);
            return *reinterpret_cast<out_tile *>(page_base_ptr);
        }
        __device__ static inline kv_row_vec &get_output_buffer(state<config> &s, parsed_instruction &inst, int store_page, int row) {
            char *page_base_ptr = reinterpret_cast<char *>(s.pages[store_page].data);
            constexpr int bytes_per_row = 128 * 2; // Assuming bfloat16 is 2 bytes
            return *reinterpret_cast<kv_row_vec *>(page_base_ptr + row * bytes_per_row);
        }

        // __device__ static inline rope_st &get_rope_cos_smem(state<Config> &s) { 
        //     int pid = s.pid(ROPE_PAGE);
        //     char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        //     return *reinterpret_cast<rope_st *>(page_base_ptr);
        // }
        // __device__ static inline rope_st &get_rope_sin_smem(state<Config> &s) { 
        //     int pid = s.pid(ROPE_PAGE);
        //     char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        //     return *reinterpret_cast<rope_st *>(page_base_ptr + sizeof(rope_st));
        // }

        struct controller
        {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query)
            {
                // TODO: Need to correct order
                // unused pages, then activation, then rms scale, then weights, then rope cos, then rope sin
                // int ret_order[13] = {7, 8, 9, 10, 11, 12, PAGE_ACTIVATION, PAGE_WEIGHT_START, PAGE_WEIGHT_START + 1, PAGE_WEIGHT_START + 2, PAGE_WEIGHT_START + 3, ROPE_PAGE_COS, ROPE_PAGE_SIN};
                // return ret_order[query];
                return query; 
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
                init_semaphore(routing_page_ready(s), 1);
                return (PIPELINE_STAGES * 2) + (2 * 2) + 3;
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
                        tma::load_async(a, g.rms_rope_intermediates, {inst.row + laneid(), i}, inputs_arrived(s, pipeline_stage));
                    } else if (laneid() == 2) {
                        int weight_page = get_weight_page(s, pipeline_stage);
                        if(i < PIPELINE_STAGES) {
                            s.wait_page_ready(weight_page);
                            s.wait_page_ready(weight_page+1);
                        }
                        weight_tile &b = *reinterpret_cast<weight_tile *>(s.pages[weight_page].data);
                        tma::load_async(b, g.qkv_weights, {inst.col, i}, inputs_arrived(s, pipeline_stage));
                    } else if (laneid() == 3 && i == inst.iters - 1) { // If last iteration, load routing table
                        int next_stage = ring_advance<PIPELINE_STAGES>(pipeline_stage);
                        // wait(inputs_finished(s, next_stage), get_phasebit<1>(semaphore_bitfield, next_stage));
                        wait(outputs_arrived(s, 0), 0);
                        arrive(routing_page_ready(s));
                    } else if (laneid() == 4 && i == inst.iters - 1) {
                        // int next_next_stage = ring_advance<PIPELINE_STAGES>(pipeline_stage, 2);
                        // wait(inputs_finished(s, next_next_stage), get_phasebit<1>(semaphore_bitfield, next_next_stage));

                        // int cos_page = get_cos_page(s, inst);
                        // rope_st &rope_cos_smem = *reinterpret_cast<rope_st *>(s.pages[cos_page].data);
                        // tma::expect(rope_cos_arrived(s), rope_cos_smem);
                        // tma::load_async(rope_cos_smem, g.rope_cos, {0, inst.batch_head_idx, static_cast<int>(g.pos_id), 0}, rope_cos_arrived(s));

                        // int sin_page = get_sin_page(s, inst);
                        // rope_st &rope_sin_smem = *reinterpret_cast<rope_st *>(s.pages[sin_page].data);
                        // tma::expect(rope_sin_arrived(s), rope_sin_smem);
                        // tma::load_async(rope_sin_smem, g.rope_sin, {0, inst.batch_head_idx, static_cast<int>(g.pos_id), 0}, rope_sin_arrived(s));
                    }
                        
                    update_phasebit<1>(semaphore_bitfield, pipeline_stage);
                }
                // if (laneid() == 0) printf(BLUE_TEXT "Loader finished issuing loads\n" RESET_TEXT);
                
                // Now load rope => Need to load a 128 x 128 tile here!! 
                warp::sync();
                // if (laneid() == 0)
                // {
                    // Rope cos
                    // s.wait_page_ready(ROPE_PAGE);
                    // rope_st &rope_cos_smem = get_rope_cos_smem(s);
                    // tma::expect(rope_cos_arrived(s), rope_cos_smem);
                    // tma::load_async(rope_cos_smem, g.rope_cos, {0, 0, inst.batch_head_idx, 0}, rope_cos_arrived(s));
            
                    // // Rope sin
                    // rope_st &rope_sin_smem = get_rope_sin_smem(s);
                    // s.record(TEVENT_TRIPLES_START + 6);
                    // tma::expect(rope_sin_arrived(s), rope_sin_smem);
                    // tma::load_async(rope_sin_smem, g.rope_sin, {0, 0, inst.batch_head_idx, 0}, rope_sin_arrived(s));
                // }
                // warp::sync(); 

                // FREE EVERYTHING BUT THE LAST STAGE 
                // if (laneid() >= 28) {
                //     for(int i = 0; i < PIPELINE_STAGES-1; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                //         wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                //         int release_pid = pipeline_stage*4 + laneid() - 28;
                //         s.finish_page(release_pid, config::NUM_CONSUMER_WARPS);
                //     }
                // }
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
                    rt_bf<16, 128> out_bf16;
                    
                    warpgroup::load_async(acc_rt, accumulator);
                    tensor_load_wait();

                    if (inst.col < V_BLK_START) {
                        // rt_fl<16, 64> x1, x2, temp1, temp2, cos, sin;
                        // Load cos and sin
                        // wait(rope_cos_arrived(s), 0);
                        // int cos_page = get_cos_page(s, inst);
                        // rope_st &rope_cos_smem = *reinterpret_cast<rope_st *>(s.pages[cos_page].data);
                        // rope_subtile cos_smem(rope_cos_smem, {warpgroup::warpid(), 0});
                        // warp::load(cos, cos_smem);

                        // wait(rope_sin_arrived(s), 0);
                        // int sin_page = get_sin_page(s, inst);   
                        // rope_st &rope_sin_smem = *reinterpret_cast<rope_st *>(s.pages[sin_page].data);
                        // rope_subtile sin_smem(rope_sin_smem, {warpgroup::warpid(), 0});
                        // warp::load(sin, sin_smem);

                        // for(int i = 0; i < 128/32; i++) {
                        //     #pragma unroll
                        //     for(int j = 0; j < 4; j++) {
                        //         x1.tiles[0][i].data[j] = acc_rt.tiles[0][i].data[j];
                        //         x2.tiles[0][i].data[j] = acc_rt.tiles[0][i+128/32].data[j];
                        //     }
                        // }

                        // warp::mul(temp1, x1, cos);
                        // warp::mul(temp2, x2, cos);
                        // warp::mul(x2, x2, -1.f);
                        // warp::mul(x1, x1, sin);
                        // warp::mul(x2, x2, sin);
                        // warp::add(temp1, temp1, x2);
                        // warp::add(temp2, temp2, x1);

                        // for(int i = 0; i < 128/32; i++) {
                        //     #pragma unroll
                        //     for(int j = 0; j < 4; j++) {
                        //         acc_rt.tiles[0][i].data[j] = temp1.tiles[0][i].data[j];
                        //         acc_rt.tiles[0][i+128/32].data[j] = temp2.tiles[0][i].data[j];
                        //     }
                        // }
                    }

                    warp::copy(out_bf16, acc_rt);
                    warp::arrive(s.tensor_finished);
                    
                    int store_page_id = get_store_page(s, inst, groupid);
                    out_tile &store_buffer = *reinterpret_cast<out_tile *>(s.pages[store_page_id].data);
                    warpgroup::sync(groupid);
                    warpgroup::store(store_buffer, out_bf16);
                    warpgroup::sync(groupid);
                    warpgroup::arrive(outputs_shared(s, groupid));
                }
                
                if (kittens::group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
                {
                    s.record(TEVENT_CONSUMER_END);
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

                wait(routing_page_ready(s), 0);
                int routing_page = get_routing_page(s, inst);
                routing_sv &routing_vec = *reinterpret_cast<routing_sv *>(s.pages[routing_page].data);
                // warp::load(routing_vec, g.routing_table, {0, 0});

                if (laneid() < 2)
                // if (laneid() > 64)
                // if (laneid() == 0)
                // if (laneid() == 1)
                {
                    wait(outputs_shared(s, laneid()), 0);
                    int store_page = get_store_page(s, inst, laneid());
                    out_tile &output = *reinterpret_cast<out_tile *>(s.pages[store_page].ptr());
                    // uint32_t output_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&output.data[0]));

                    // for (int i = 0; i < 64; ++i) {
                    //     for (int j = 0; j < 128; ++j) {
                    //         // printf("%f ", __bfloat162float(output.data[i * 128 + j]));
                    //         printf("%f ", __bfloat162float(output.idx(output_ptr, {i, j})));
                    //     }
                    //     printf("\n");
                    // }

                    if (inst.col < K_BLK_START) // Q
                    {
                        tma::store_async(g.q_post_rope, output, {(inst.row) + laneid(), inst.col});
                    } 
                    else 
                    {
                        int routing_start = laneid() * 64;
                        for (int i = routing_start; i < routing_start + 64; ++i)
                        {                            
                            // int page_idx, slot_idx;
                            // uint32_t page_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&routing_vec.data[i * 2]));
                            // uint32_t slot_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&routing_vec.data[i * 2 + 1]));
                            // move<int>::lds(page_idx, page_ptr);
                            // move<int>::lds(slot_idx, slot_ptr);
                            int page_idx = (127 - i); 
                            int slot_idx = 0;
                            
                            int kv_row_index = i - routing_start;
                            // kv_row_vec& kv_sv = *reinterpret_cast<kv_row_vec*>(&output.data[kv_row_index * 128]);
                            kv_row_vec* kv_sv = reinterpret_cast<kv_row_vec*>(s.pages[store_page].ptr());
                            
                            // kv_row_vec& kv_sv = get_output_buffer(s, inst, store_page, kv_row_index);
                            // printf("kv_row_index: %d\n", kv_row_index);
                            // printf("routing_start: %d\n", routing_start);
                            // printf("Storing to page_idx: %d\n", page_idx);
                            
                            printf("Calculating row idx: %d\n", kv_row_index);
                            if (inst.col < V_BLK_START) // K
                            {
                                int head_idx = inst.col - K_BLK_START;

                                for (int j = 0; j < 128; j++) {
                                    printf("%f ", __bfloat162float(kv_sv[kv_row_index].data[j]));
                                }
                                printf("\n");

                                
                                tma::store_async(g.k_cache, kv_sv[kv_row_index], {page_idx, slot_idx, head_idx, 0});
                                tma::store_async_wait();
                                tma::store_async_read_wait();
                                // tma::store_async(g.k_cache, kv_sv, {0, 0});
                            }
                            else // V
                            {
                                int head_idx = inst.col - V_BLK_START;
                                // tma::store_async(g.v_cache, kv_sv, {page_idx, slot_idx, head_idx, 0});
                            }
                            
                            // break;
                        }
                    }

                    tma::store_async_read_wait();
                    // s.finish_page(store_page, config::NUM_CONSUMER_WARPS);
                } 

                warp::sync();
                if (laneid() >= 28) {
                    int pipeline_stage = 0;
                    for(int i = 0; i < PIPELINE_STAGES; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                        int release_pid = pipeline_stage*4 + laneid() - 28;
                        s.finish_page(release_pid, config::NUM_CONSUMER_WARPS);
                    }
                }
                // if (laneid() == 0) 
                // {
                //     int routing_page = get_routing_page(s, inst);
                //     s.finish_page(routing_page, config::NUM_CONSUMER_WARPS);
                //     s.finish_page(routing_page + 1, config::NUM_CONSUMER_WARPS); // Need to free unused fourth page as well for stage!!
                // }

                asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.

                // TODO: What should atomic add here be?
                if (warp::laneid() == 0)
                    atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.col}], 1);

                warp::sync();
                if (laneid() == 0)
                    s.record(TEVENT_STORE_END);
            }
        };
    };
}