#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm
{

    template <typename config, typename globals>
    struct attention_partial
    {
        static constexpr int opcode = OPCODE_GQA_AttentionDecode;
        static constexpr int NUM_STAGES = 3;
        static constexpr int GQA_RATIO = LLAMA_70B_NUM_ATTENTION_HEADS / LLAMA_70B_NUM_KV_HEADS;
        static constexpr int QOL_PAGE = 0;
        static constexpr int KV_PAGE = 1;
        static constexpr int KV_INDICES_LEN = 0;
        static constexpr int MAX_KV_INDICES_LEN = 0;

        static_assert(GQA_RATIO == 8, "GQA_RATIO must be 8.");
        static_assert(NUM_STAGES <= 4, "Modify page allocation for KVs.");

        using q_rt = rt_bf<16, LLAMA_70B_HEAD_DIM>; // only 4 rows are used
        using q_st = st_bf<16, LLAMA_70B_HEAD_DIM>; // only 4 rows are used
        using k_rt = rt_bf<LLAMA_70B_KV_BLOCK_SIZE, LLAMA_70B_HEAD_DIM>;
        using v_rt = rt_bf<LLAMA_70B_KV_BLOCK_SIZE, LLAMA_70B_HEAD_DIM, col_l>;
        using kv_st = st_bf<LLAMA_70B_KV_BLOCK_SIZE, LLAMA_70B_HEAD_DIM>;
        using attn_fl_rt = rt_fl<16, LLAMA_70B_KV_BLOCK_SIZE>;      // only 4 values are used
        using attn_bf_rt = rt_bf<16, LLAMA_70B_KV_BLOCK_SIZE>;      // only 4 values are used
        using max_vec_rv = col_vec<rt_fl<16, LLAMA_70B_HEAD_DIM>>;  // only 4 values are used
        using max_vec_sv = sv_fl<16>;                              // only 4 values are used
        using norm_vec_rv = col_vec<rt_fl<16, LLAMA_70B_HEAD_DIM>>; // only 4 values are used
        using norm_vec_sv = sv_fl<16>;                             // only 4 values are used
        using l_rv = col_vec<rt_fl<16, LLAMA_70B_HEAD_DIM>>;        // only 4 values are used
        using l_sv = sv_fl<16>;                                    // only 4 values are used
        using o_rt = rt_fl<16, LLAMA_70B_HEAD_DIM>;                 // only 4 rows are used
        using o_rt_bf = rt_bf<16, LLAMA_70B_HEAD_DIM>;                 // only 4 rows are used
        using o_sv = sv_bf<LLAMA_70B_HEAD_DIM>;

        struct parsed_instruction
        {
            int layer_idx;
            int kv_head_idx;
            int partial_idx;
            int kv_indices_len;
            int kv_indices[KV_INDICES_LEN]; // actual physical indices to read in from paged KV

            __device__ inline parsed_instruction(typename config::instruction_t &instruction)
            {                
                layer_idx = instruction[1];
                kv_head_idx = instruction[2];
                kv_indices_len = instruction[3]; // length of number of indices from paged KV cache to read 
                // partial_idx = instruction[4];

                #pragma unroll
                for (int k = 0; k < MAX_KV_INDICES_LEN; ++k) 
                {
                    if (k < kv_indices_len) 
                    {
                        kv_indices[k] = instruction[4 + k];
                    }
                }
            }
            __device__ inline parsed_instruction(state<config> &s) : parsed_instruction(s.instruction()) {}
        };

        // We have 32 dynamic semaphores total
        __device__ static inline semaphore &Q_arrived(state<config> &s)
        {
            return s.semaphores()[0];
        }
        __device__ static inline semaphore &O_arrived(state<config> &s)
        {
            return s.semaphores()[1];
        }
        __device__ static inline semaphore &K_arrived(state<config> &s, int stage)
        {
            return s.semaphores()[2 + stage * 2];
        }
        __device__ static inline semaphore &V_arrived(state<config> &s, int stage)
        {
            return s.semaphores()[2 + stage * 2 + 1];
        }
        __device__ static inline semaphore &K_finished(state<config> &s, int stage)
        {
            return s.semaphores()[2 + NUM_STAGES * 2 + stage * 2];
        }
        __device__ static inline semaphore &V_finished(state<config> &s, int stage)
        {
            return s.semaphores()[2 + NUM_STAGES * 2 + stage * 2 + 1];
        }

        __device__ static inline void wait_QOL_page(state<config> &s) { s.wait_page_ready(s.pid(QOL_PAGE)); }
        __device__ static inline void wait_KV_page(state<config> &s) { s.wait_page_ready(s.pid(KV_PAGE)); }
        __device__ static inline void finish_QOL_page(state<config> &s)
        {
            if (warp::laneid() == 0)
                arrive(s.page_finished[s.pid(QOL_PAGE)], config::NUM_CONSUMER_WARPS);
        }
        __device__ static inline void finish_KV_page(state<config> &s)
        {
            if (warp::laneid() == 0)
                arrive(s.page_finished[s.pid(KV_PAGE)], config::NUM_CONSUMER_WARPS);
        }
        __device__ static inline q_st &get_Q_smem(state<config> &s)
        {
            int pid = s.pid(QOL_PAGE);
            return *reinterpret_cast<q_st *>(s.pages[pid].data);
        }
        __device__ static inline o_sv (&get_O_smem(state<config> &s))[8]
        {
            int pid = s.pid(QOL_PAGE);
            return *reinterpret_cast<o_sv(*)[8]>(
                reinterpret_cast<char *>(s.pages[pid].data) + sizeof(q_st));
        }
        __device__ static inline kv_st &get_K_smem(state<config> &s, int stage)
        {
            int pid = s.pid(KV_PAGE);
            return *reinterpret_cast<kv_st *>(
                reinterpret_cast<char *>(s.pages[pid].data) + sizeof(kv_st) * (stage * 2));
        }
        __device__ static inline kv_st &get_V_smem(state<config> &s, int stage)
        {
            int pid = s.pid(KV_PAGE);
            return *reinterpret_cast<kv_st *>(
                reinterpret_cast<char *>(s.pages[pid].data) + sizeof(kv_st) * (1 + stage * 2));
        }

        template <ducks::sv::all SV, ducks::rt::all RT>
        __device__ static inline void store_8_rows(SV (&dst)[8], const RT &src)
        {
            static_assert(RT::rows == 16, "src rows must be 16.");
            static_assert(SV::length == src.cols, "dst length must match src cols.");

            using T2 = RT::dtype;
            using T = base_types::packing<T2>::unpacked_type;
            using U = SV::dtype;
            using U2 = base_types::packing<U>::packed_type;
                    
            uint32_t dst_ptr[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                dst_ptr[i] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[i].data[0]));
            }

            int laneid = kittens::laneid();
            int local_row_idx = (laneid % 32) / 4;
            int local_col_idx = laneid % 4;

            for (int j = 0; j < src.width; j++)
            {
                U2 tmp[2];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);
                int col_idx = local_col_idx * 2 + j * 16;
                move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx+8), tmp[1]);
            }
        }
        template <ducks::rt::row_layout RT>
        __device__ static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val = 0)
        {
            if (col_idx >= dst.cols)
                return;
#pragma unroll
            for (int i = 0; i < dst.height; i++)
            {
#pragma unroll
                for (int j = 0; j < dst.width; j++)
                {
#pragma unroll
                    for (int k = 0; k < dst.packed_per_tile; k++)
                    {
                        const int col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((warp::laneid() % 4) * 2);
                        const int col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((warp::laneid() % 4) * 2) + 1;
                        if (col_idx_x >= col_idx)
                        {
                            dst.tiles[i][j].data[k].x = val;
                        }
                        else
                        {
                            dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x;
                        }
                        if (col_idx_y >= col_idx)
                        {
                            dst.tiles[i][j].data[k].y = val;
                        }
                        else
                        {
                            dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y;
                        }
                    }
                }
            }
        }
        // This is super specific to loading Q in a single warp
        // Mainly two things are different:
        //   1. Ignores Q global dimensions
        //   2. Only loads 8 rows of Q, not 16 (assumes GQA_RATIO == 8)
        __device__ static inline void load_Q_async(q_st &dst, const globals::activations_t &src, int q_head_start_idx)
        {
            static_assert(LLAMA_70B_HEAD_DIM == 128 && GQA_RATIO == 8, "Fix this function.");
            using T = typename q_st::dtype;
            constexpr int elem_per_memcpy = sizeof(float4) / sizeof(typename q_st::dtype); // 8
            constexpr int memcpy_per_row = LLAMA_70B_HEAD_DIM / elem_per_memcpy;            // 16

            typename globals::activations_t::dtype *src_ptr = &src.raw_ptr[q_head_start_idx * LLAMA_70B_HEAD_DIM];
            uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));

            int laneid = warp::laneid();
            int col = (laneid % memcpy_per_row) * elem_per_memcpy; // (0...15) * 8
            int base_row_in_group = (laneid < memcpy_per_row) ? 0 : 1;

            #pragma unroll
            for (int i = 0; i < (GQA_RATIO / 2); ++i)
            {
                int row = base_row_in_group + i * 2;
                asm volatile(
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
                    "r"(dst.idx(dst_ptr, {row, col})),
                    "l"(&src_ptr[row * LLAMA_70B_HEAD_DIM + col])
                    : "memory");
            }

            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        struct controller
        {
            static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query)
            {
                int ret_order[13] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1};
                return ret_order[query];
            }
            static __device__ int init_semaphores(const globals &g, state<config> &s)
            {
                init_semaphore(Q_arrived(s), 0, 1);
                init_semaphore(O_arrived(s), 0, 1);
                for (int i = 0; i < NUM_STAGES; i++)
                {
                    init_semaphore(K_arrived(s, i), 0, 1);
                    init_semaphore(V_arrived(s, i), 0, 1);
                    init_semaphore(K_finished(s, i), 0, 1);
                    init_semaphore(V_finished(s, i), 0, 1);
                }
                return 3 + 4 * NUM_STAGES;
            }
        };
        struct loader
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_LOADER_START);
                }

                auto laneid = warp::laneid();
                if (laneid == 0)
                {
                    // Setup
                    parsed_instruction inst{s};
                    int seq_len = g.pos_id + 1;

                    s.record(TEVENT_AT_GMEM_WAIT);

                    // Wait for the previous ops to finish (16 dims each, so 4 ops on the same head)
                    while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, LLAMA_70B_NUM_ATTENTION_HEADS + inst.kv_head_idx}] < 4 ||                       // K
                           *(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, LLAMA_70B_NUM_ATTENTION_HEADS + LLAMA_70B_NUM_KV_HEADS + inst.kv_head_idx}] < 4) // V
                    {
                        __nanosleep(20);
                    }

                    s.record(TEVENT_DONE_GMEM_WAIT);

                    // Wait for the KV page
                    wait_KV_page(s);
                    
                    // TODO: Do we still need this case? 
                    // if (start_blk_idx >= end_blk_idx)
                    //     finish_KV_page(s);

                    // Run the pipeline!
                    // for (int i = 0; i + start_blk_idx < end_blk_idx; ++i)
                    for (int i = 0; i < inst.kv_indices_len; ++i)
                    {
                        int stage = i % NUM_STAGES;
                        kv_st &K_smem = get_K_smem(s, stage);
                        kv_st &V_smem = get_V_smem(s, stage);

                        if (i >= NUM_STAGES)
                        {
                            wait(K_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                            wait(V_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                        }
                        
                        int cur_page_idx = inst.kv_indices[i];
                        // TODO: Need to ensure right indexing based on k_cache and v_cache shape
                        tma::expect(K_arrived(s, stage), K_smem);
                        // TODO: ensure AXIS access is right here? 
                        tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(K_smem, g.k_cache, {cur_page_idx, 0, inst.kv_head_idx, 0}, K_arrived(s, stage));
                        tma::expect(V_arrived(s, stage), V_smem);
                        tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(V_smem, g.v_cache, {cur_page_idx, 0, inst.kv_head_idx, 0}, V_arrived(s, stage));
                    }
                }
                else if (laneid >= 2 && laneid < config::NUM_PAGES)
                {
                    int unused_page = s.pid(laneid);
                    s.wait_page_ready(unused_page);
                    arrive(s.page_finished[unused_page], config::NUM_CONSUMER_WARPS);
                }

                warp::sync();
                if (laneid == 0)
                {
                    s.record(TEVENT_LOADER_END);
                }
            }
        };
        struct launcher
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                if (warp::laneid() == 0)
                {
                    s.wait_tensor_ready();
                    arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
                }
            }
        };
        struct consumer
        {
            static __device__ void run(const globals &g, state<config> &s)
            {

                if (warp::laneid() == 0)
                {
                    s.record(TEVENT_CONSUMER_START + warpid());
                }

                if (warpid() == 0)
                {
                    // Wait for the previous ops to finish
                    parsed_instruction inst{s};
                    int q_head_start_idx = inst.kv_head_idx * GQA_RATIO;
                    while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, q_head_start_idx + 0}] < 4 ||
                           *(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, q_head_start_idx + 1}] < 4 ||
                           *(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, q_head_start_idx + 2}] < 4 ||
                           *(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, q_head_start_idx + 3}] < 4 ||
                           *(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, q_head_start_idx + 4}] < 4 ||
                           *(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, q_head_start_idx + 5}] < 4 ||
                           *(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, q_head_start_idx + 6}] < 4 ||
                           *(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, q_head_start_idx + 7}] < 4)
                    {
                        __nanosleep(20);
                    }
                    warp::sync();

                    // Initiate the load on Q
                    wait_QOL_page(s);

                    q_st &Q_smem = get_Q_smem(s);
                    load_Q_async(Q_smem, g.q_post_rope, q_head_start_idx);

                    // Setup
                    int q_head_local_idx = (q_head_start_idx % q_rt::tile_size_row) / 8;
                    int seq_len = g.pos_id + 1;
                    
                    float softmax_temp = g.attn_scale * 1.44269504089f; // 1 / (sqrt(D_h) * ln(2))
                    q_rt Q_reg;
                    k_rt K_reg;
                    v_rt V_reg;
                    o_rt O_reg;
                    attn_fl_rt attn_fl_reg;
                    attn_bf_rt attn_bf_reg;
                    max_vec_rv max_vec_reg;
                    max_vec_rv scaled_max_vec_reg;
                    max_vec_rv last_scaled_max_vec_reg;
                    max_vec_rv diff_scaled_max_vec_reg;
                    norm_vec_rv norm_vec_reg;
                    warp::neg_infty(max_vec_reg);
                    warp::zero(last_scaled_max_vec_reg); // just not +-inf
                    warp::zero(norm_vec_reg);
                    warp::zero(O_reg);
                    o_sv(&O_smem)[8] = get_O_smem(s);

                    // Wait for Q to arrive
                    warp::load_async_wait();

                    if (laneid() == 0)
                    {
                        s.record(TEVENT_CONSUMER_START + 16);
                    }

                    warp::load(Q_reg, Q_smem);

                    // Run the pipeline!
                    for (int i = 0; i < inst.kv_indices_len; ++i)
                    {
                        int stage = i % NUM_STAGES;
                        int cur_page_idx = inst.kv_indices[i];
                        kv_st &K_smem = get_K_smem(s, stage);
                        kv_st &V_smem = get_V_smem(s, stage);

                        // Perform Q @ K.T
                        warp::zero(attn_fl_reg);
                        warp::wait(K_arrived(s, stage), (i / NUM_STAGES) % 2);
                        if (laneid() == 0 && i < 16)
                            s.record(TEVENT_CONSUMER_START + 32 + i);
                        warp::load(K_reg, K_smem);
                        warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                        warp::sync();
                        warp::arrive(K_finished(s, stage));

                        // Mask out invalid positions at the end
                        if ((i + 1) * LLAMA_70B_KV_BLOCK_SIZE > seq_len)
                            right_fill(attn_fl_reg, attn_fl_reg, seq_len % LLAMA_70B_KV_BLOCK_SIZE, -999999999999.f);

                        // Obtain maximums per row (which is per head)
                        warp::row_max(max_vec_reg, attn_fl_reg, max_vec_reg); // includes previous max

                        // Scale attention block and maximums by sqrt(D_h)
                        warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                        warp::mul(scaled_max_vec_reg, max_vec_reg, softmax_temp);

                        // Calculate softmax numerator
                        warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
                        warp::exp2(attn_fl_reg, attn_fl_reg);

                        // Calculate softmax denominator
                        warp::sub(diff_scaled_max_vec_reg, last_scaled_max_vec_reg, scaled_max_vec_reg);
                        warp::exp2(diff_scaled_max_vec_reg, diff_scaled_max_vec_reg);

                        // Normalize and accumulate numerator (A @ V)
                        warp::mul_row(O_reg, O_reg, diff_scaled_max_vec_reg);
                        warp::wait(V_arrived(s, stage), (i / NUM_STAGES) % 2);
                        if (laneid() == 0 && i < 16)
                            s.record(TEVENT_CONSUMER_START + 48 + i);
                        warp::load(V_reg, V_smem);
                        warp::copy(attn_bf_reg, attn_fl_reg); // Convert to bf16 to do matmul
                        warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
                        warp::sync();
                        warp::arrive(V_finished(s, stage));

                        // Normalize and accumulate demoniator
                        warp::mul(norm_vec_reg, norm_vec_reg, diff_scaled_max_vec_reg);
                        warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);

                        // Save for next iteration
                        warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
                    }

                    // Finish
                    warp::sync();
                    if (laneid() == 0)
                        s.record(TEVENT_CONSUMER_START + 64);
                    if (inst.kv_indices_len > 0)
                    {
                        finish_KV_page(s);
                        warp::div_row(O_reg, O_reg, norm_vec_reg);
                    }

                    // Store the results
                    o_rt_bf O_reg_bf; 
                    warp::copy(O_reg_bf, O_reg);
                    store_8_rows(O_smem, O_reg_bf);
                    
                    warp::sync();
                    if (laneid() == 0)
                    {
                        s.record(TEVENT_CONSUMER_START + 65);
                    }
                    warp::arrive(O_arrived(s));
                    warp::sync();
                }

                if (laneid == 0)
                {
                    s.record(TEVENT_CONSUMER_END + warpid());
                }
            }
        };
        struct storer
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                if (laneid == 0)
                {
                    s.record(TEVENT_STORE_START);
                }

                parsed_instruction inst{s};
                int laneid = warp::laneid();
                int q_head_start_idx = inst.kv_head_idx * GQA_RATIO; // 0, 4, 8, 12, 16, 20, 24, 28
                int q_head_vec_start_idx = q_head_start_idx % 16;

                // Store partial attention output to global memory
                if (laneid == 0)
                {
                    o_sv(&O_smem)[8] = get_O_smem(s);
                    wait(O_arrived(s), 0);
                    s.record(TEVENT_OUTPUT_READY);

                    for (int i = 0; i < 8; ++i)
                    {
                        // TODO: Check indexing here, based on final shape of atttn_out
                        tma::store_async<cache_policy::NORMAL>(g.attn_out, O_smem[i], {0, 0, 0, q_head_start_idx + i});
                    }
                }

                warp::sync(); // ensure all writes are committed
                asm volatile("fence.acq_rel.gpu;");

                // Wait and finish
                if (laneid < GQA_RATIO)
                {
                    tma::store_async_wait();
                    if (laneid == 0)
                        s.record(123 + laneid);
                    finish_QOL_page(s);
                    // Adding only at 0, 4, 8, ... should be sufficient for the reduction op!
                    atomicAdd(&g.Bar[{inst.layer_idx, OPCODE_GQA_AttentionDecode - 1, q_head_start_idx + laneid}], 1);
                }
                warp::sync();
                if (laneid == 0)
                {
                    s.record(TEVENT_STORE_END);
                }
            }
        };
    };
}
