#include "llama.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

    template <typename config, typename globals>
    struct attention_decode {
        static constexpr int opcode = OPCODE_GQA_AttentionDecode;
        static constexpr int GQA_RATIO = globals::num_attention_heads / globals::num_kv_heads;
        static constexpr int NUM_STAGES = 7;
        static constexpr int KV_PER_PAGE = config::PAGE_SIZE / (globals::kv_block_size * globals::head_dim * sizeof(bf16)); // 4
        static constexpr int UNUSED_PAGE_START = 2*((NUM_STAGES + KV_PER_PAGE-1) / KV_PER_PAGE);

        static_assert(GQA_RATIO == 4, "GQA_RATIO must be 4.");
        static_assert(GQA_RATIO*globals::head_dim*sizeof(bf16) <= config::SCRATCH_BYTES, "Q and O don't fit in scratch.");
        static_assert(NUM_STAGES <= 7, "Not enough semaphores.");

        using q_st = st_bf<16, globals::head_dim>; // only 4 rows are used
        using kv_st = st_bf<globals::kv_block_size, globals::head_dim>;
        using o_sv = sv_bf<globals::head_dim>;
        using o_sv_array_t = o_sv[GQA_RATIO];

        struct parsed_instruction {
            int layer_idx;
            int batch_idx;
            int kv_head_idx;
            __device__ inline parsed_instruction(typename config::instruction_t &instruction)
            {
                layer_idx = instruction[1];
                batch_idx = instruction[2];
                kv_head_idx = instruction[3];
            }
            __device__ inline parsed_instruction(state<config> &s) : parsed_instruction(s.instruction()) {}
        };

        __device__ static inline semaphore &O_arrived(state<config> &s)             { return s.semaphores()[0]; }
        __device__ static inline semaphore &K_arrived(state<config> &s, int stage)  { return s.semaphores()[1 + NUM_STAGES*0 + stage]; }
        __device__ static inline semaphore &V_arrived(state<config> &s, int stage)  { return s.semaphores()[1 + NUM_STAGES*1 + stage]; }
        __device__ static inline semaphore &K_finished(state<config> &s, int stage) { return s.semaphores()[1 + NUM_STAGES*2 + stage]; }
        __device__ static inline semaphore &V_finished(state<config> &s, int stage) { return s.semaphores()[1 + NUM_STAGES*3 + stage]; }

        __device__ static inline int get_K_page(state<config> &s, int stage) {return s.pid(stage/KV_PER_PAGE * 2 + 0); }
        __device__ static inline int get_V_page(state<config> &s, int stage) {return s.pid(stage/KV_PER_PAGE * 2 + 1); }

        template <ducks::sv::all SV, ducks::rt::all RT>
        __device__ static inline void store_4_rows(SV (&dst)[4], const RT &src)
        {
            static_assert(RT::rows == 16, "src rows must be 16.");
            static_assert(SV::length == src.cols, "dst length must match src cols.");

            using T2 = RT::dtype;
            using T = base_types::packing<T2>::unpacked_type;
            using U = SV::dtype;
            using U2 = base_types::packing<U>::packed_type;

            uint32_t dst_ptr[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                dst_ptr[i] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[i].data[0]));

            int laneid = kittens::laneid();            
            if (laneid < 16)
            {
                int local_row_idx = laneid / 4;
                int local_col_idx = laneid % 4;
                for (int j = 0; j < src.width; j++)
                {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
                }
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
                            dst.tiles[i][j].data[k].x = val;
                        else
                            dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x;
                        if (col_idx_y >= col_idx)
                            dst.tiles[i][j].data[k].y = val;
                        else
                            dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y;
                    }
                }
            }
        }

        __device__ static inline void load_Q_async(q_st &dst, const globals::activations_t &src, int batch_idx,int q_head_start_idx)
        {
            static_assert(globals::head_dim == 128 && GQA_RATIO == 4, "Fix this function.");
            using T = typename q_st::dtype;
            constexpr int elem_per_memcpy = sizeof(float4) / sizeof(typename q_st::dtype); // 8
            constexpr int memcpy_per_row = globals::head_dim / elem_per_memcpy;            // 16

            typename globals::activations_t::dtype *src_ptr = &src[coord<>{batch_idx, q_head_start_idx * globals::head_dim}];
            uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));

            int laneid = warp::laneid();
            int col = (laneid % memcpy_per_row) * elem_per_memcpy; // (0...15) * 8
            int base_row_in_group = (laneid < memcpy_per_row) ? 0 : 1;

            #pragma unroll
            for (int i = 0; i < (GQA_RATIO / 2); ++i)
            {
                int row = base_row_in_group + i * 2;
                asm volatile(
                    "{cp.async.cg.shared.global.L2::128B [%0], [%1], 16;}" ::
                    "r"(dst.idx(dst_ptr, {row, col})),
                    "l"(&src_ptr[row * globals::head_dim + col])
                    : "memory");
            }
            asm volatile("{cp.async.commit_group;}" ::: "memory");
        }

        struct controller
        {
            static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query)
            {
                return (query + UNUSED_PAGE_START) % config::NUM_PAGES;
            }
            static __device__ int init_semaphores(const globals &g, state<config> &s)
            {
                init_semaphore(O_arrived(s), 0, 1);
                for (int i = 0; i < NUM_STAGES; i++)
                {
                    init_semaphore(K_arrived(s, i), 0, 1);
                    init_semaphore(V_arrived(s, i), 0, 1);
                    init_semaphore(K_finished(s, i), 0, 1);
                    init_semaphore(V_finished(s, i), 0, 1);
                }
                return 1 + 4*NUM_STAGES;
            }
        };

        struct loader
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                if (warp::laneid() == 0) s.record(TEVENT_LOADER_START);
                parsed_instruction inst{s};
                int laneid = warp::laneid();
                int batch_block_idx = inst.batch_idx / globals::matmul_batch_block_size;

                if (laneid == 0) { // Load Ks
                    while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, batch_block_idx, (int)globals::num_attention_heads + inst.kv_head_idx}] < 1)
                        __nanosleep(20);

                    uint32_t phasebits = 0xFFFFFFFF;
                    int total_attn_blocks = (g.pos_id+1 + globals::kv_block_size-1) / globals::kv_block_size;
                    for (int i = 0; i < total_attn_blocks; ++i) {
                        int stage = i % NUM_STAGES;
                        int k_page = get_K_page(s, stage);
                        kv_st &K_smem = *reinterpret_cast<kv_st *>((char*)s.pages[k_page].data + (stage%KV_PER_PAGE)*sizeof(kv_st));
                        
                        wait(K_finished(s, stage), get_phasebit<0>(phasebits, stage));
                        update_phasebit<0>(phasebits, stage);

                        if ((i < NUM_STAGES) && (stage % KV_PER_PAGE == 0))
                            s.wait_page_ready(k_page);

                        tma::expect(K_arrived(s, stage), K_smem);
                        tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(K_smem, g.k_cache, {(int)g.batch_size*inst.layer_idx + inst.batch_idx, i, inst.kv_head_idx, 0}, K_arrived(s, stage));
                    }
                    for (int i = 0; i < NUM_STAGES; i++) { // Finish K pages
                        int stage = i % NUM_STAGES;
                        wait(K_finished(s, stage), get_phasebit<0>(phasebits, stage));
                        if (stage >= total_attn_blocks)
                            s.wait_page_ready(get_K_page(s, stage));
                    }
                    for (int i = 0; i < NUM_STAGES; i++) { // Finish K pages TODO FIX
                        int stage = i % NUM_STAGES;
                        if (stage % KV_PER_PAGE == 0)
                            s.finish_page(get_K_page(s, stage), config::NUM_CONSUMER_WARPS);
                    }
                } else if (laneid == 1) { // Load Vs
                    while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, batch_block_idx, (int)globals::num_attention_heads + (int)globals::num_kv_heads + inst.kv_head_idx}] < 1)
                        __nanosleep(20);

                    uint32_t phasebits = 0xFFFFFFFF;
                    int total_attn_blocks = (g.pos_id+1 + globals::kv_block_size-1) / globals::kv_block_size;
                    for (int i = 0; i < total_attn_blocks; ++i) {
                        int stage = i % NUM_STAGES;
                        int v_page = get_V_page(s, stage);
                        kv_st &V_smem = *reinterpret_cast<kv_st *>((char*)s.pages[v_page].data + (stage%KV_PER_PAGE)*sizeof(kv_st));

                        wait(V_finished(s, stage), get_phasebit<0>(phasebits, stage));
                        update_phasebit<0>(phasebits, stage);

                        if ((i < NUM_STAGES) && (stage % KV_PER_PAGE == 0))
                            s.wait_page_ready(v_page);

                        tma::expect(V_arrived(s, stage), V_smem);
                        tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(V_smem, g.v_cache, {(int)g.batch_size*inst.layer_idx + inst.batch_idx, i, inst.kv_head_idx, 0}, V_arrived(s, stage));
                    }
                    for (int i = 0; i < NUM_STAGES; i++) { // Finish V pages
                        int stage = i % NUM_STAGES;
                        wait(V_finished(s, stage), get_phasebit<0>(phasebits, stage));
                        if (stage >= total_attn_blocks)
                            s.wait_page_ready(get_V_page(s, stage));
                    }
                    for (int i = 0; i < NUM_STAGES; i++) { // Finish V pages TODO FIX
                        int stage = i % NUM_STAGES;
                        if (stage % KV_PER_PAGE == 0)
                            s.finish_page(get_V_page(s, stage), config::NUM_CONSUMER_WARPS);
                    }
                } else if (UNUSED_PAGE_START <= laneid && laneid < config::NUM_PAGES) {
                    s.wait_page_ready(s.pid(laneid));
                    s.finish_page(s.pid(laneid), config::NUM_CONSUMER_WARPS);
                }
            }
        };
        struct launcher
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                if (warp::laneid() == 0) {
                    s.wait_tensor_ready();
                    arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
                }
            }
        };
        struct consumer
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                parsed_instruction inst{s};
                int laneid = warp::laneid();
                int warpid = group<config::NUM_CONSUMER_WARPS>::warpid();

                if (warpid == 0) {
                    int batch_block_idx = inst.batch_idx / globals::matmul_batch_block_size;
                    #pragma unroll
                    for (int i = 0; i < GQA_RATIO; i++)
                        while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_QKV_RopeAppend - 1, batch_block_idx, inst.kv_head_idx * GQA_RATIO + i}] < 1)
                            __nanosleep(20);

                    // Initiate the load on Q
                    q_st &Q_smem = *reinterpret_cast<q_st *>(s.scratch());
                    load_Q_async(Q_smem, g.q_post_rope, inst.batch_idx, inst.kv_head_idx * GQA_RATIO);

                    rt_fl<16, globals::head_dim> O_reg;
                    col_vec<rt_fl<16, globals::head_dim>> max_vec_reg;
                    col_vec<rt_fl<16, globals::head_dim>> last_scaled_max_vec_reg;
                    col_vec<rt_fl<16, globals::head_dim>> norm_vec_reg;

                    warp::zero(O_reg);
                    warp::neg_infty(max_vec_reg);
                    warp::zero(last_scaled_max_vec_reg); // just not +-inf
                    warp::zero(norm_vec_reg);

                    float softmax_temp = g.attn_scale * 1.44269504089f; // 1 / (sqrt(D_h) * ln(2))

                    // Wait for Q to arrive
                    warp::load_async_wait();
                    rt_bf<16, globals::head_dim> Q_reg;
                    warp::load(Q_reg, Q_smem);

                    // Run the pipeline!
                    int seq_len = g.pos_id + 1;
                    int total_attn_blocks = (seq_len + globals::kv_block_size - 1) / globals::kv_block_size;
                    uint32_t phasebits = 0;
                    for (int i = 0; i < total_attn_blocks; ++i)
                    {
                        int stage = i % NUM_STAGES;

                        // Load K
                        kv_st &K_smem = *reinterpret_cast<kv_st *>((char*)s.pages[get_K_page(s, stage)].data + (stage%KV_PER_PAGE)*sizeof(kv_st));
                        wait(K_arrived(s, stage), get_phasebit<0>(phasebits, stage));
                        rt_bf<globals::kv_block_size, globals::head_dim> K_reg;
                        warp::load(K_reg, K_smem);

                        // Perform Q @ K.T
                        rt_fl<16, globals::kv_block_size> attn_fl_reg;
                        warp::zero(attn_fl_reg);
                        warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                        __syncwarp();
                        warp::arrive(K_finished(s, stage));

                        // Mask out invalid positions at the end
                        if ((i + 1) * globals::kv_block_size > seq_len)
                            right_fill(attn_fl_reg, attn_fl_reg, seq_len % globals::kv_block_size, -999999999999.f);

                        // Obtain maximums per row (which is per head)
                        warp::row_max(max_vec_reg, attn_fl_reg, max_vec_reg); // includes previous max

                        // Scale attention block and maximums by sqrt(D_h)
                        warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                        col_vec<rt_fl<16, globals::head_dim>> scaled_max_vec_reg;
                        warp::mul(scaled_max_vec_reg, max_vec_reg, softmax_temp);

                        // Calculate softmax numerator
                        warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
                        warp::exp2(attn_fl_reg, attn_fl_reg);

                        // Calculate softmax denominator
                        col_vec<rt_fl<16, globals::head_dim>> diff_scaled_max_vec_reg;
                        warp::sub(diff_scaled_max_vec_reg, last_scaled_max_vec_reg, scaled_max_vec_reg);
                        warp::exp2(diff_scaled_max_vec_reg, diff_scaled_max_vec_reg);

                        // Normalize previous QK 
                        warp::mul_row(O_reg, O_reg, diff_scaled_max_vec_reg);
                        
                        // Load V
                        kv_st &V_smem = *reinterpret_cast<kv_st *>((char*)s.pages[get_V_page(s, stage)].data + (stage%KV_PER_PAGE)*sizeof(kv_st));
                        wait(V_arrived(s, stage), get_phasebit<0>(phasebits, stage));
                        rt_bf<globals::kv_block_size, globals::head_dim, col_l> V_reg;
                        warp::load(V_reg, V_smem);
                        
                        // accumulate numerator (A @ V)
                        rt_bf<16, globals::kv_block_size> attn_bf_reg;
                        warp::copy(attn_bf_reg, attn_fl_reg); // Convert to bf16 to do matmul
                        warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
                        warp::sync();
                        warp::arrive(V_finished(s, stage));

                        // Normalize and accumulate demoniator
                        warp::mul(norm_vec_reg, norm_vec_reg, diff_scaled_max_vec_reg);
                        warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);

                        // Save for next iteration
                        warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
                        update_phasebit<0>(phasebits, stage);
                    }

                    // Finish
                    __syncwarp();
                    warp::div_row(O_reg, O_reg, norm_vec_reg);

                    // Store the results
                    rt_bf<16, globals::head_dim> O_reg_bf; 
                    warp::copy(O_reg_bf, O_reg);
                    o_sv_array_t &O_smem = *reinterpret_cast<o_sv_array_t *>(s.scratch());
                    store_4_rows(O_smem, O_reg_bf);
                    __syncwarp();
                    warp::arrive(O_arrived(s));
                }

                if (warp::laneid() == 0) s.record(TEVENT_CONSUMER_END + warpid);
            }
        };
        struct storer
        {
            static __device__ void run(const globals &g, state<config> &s)
            {
                parsed_instruction inst{s};
                int laneid = warp::laneid();

                wait(O_arrived(s), 0);
                o_sv_array_t &O_smem = *reinterpret_cast<o_sv_array_t *>(s.scratch());
                if (laneid < GQA_RATIO)
                    tma::store_async(g.attn_out, O_smem[laneid], {0, 0, inst.batch_idx, inst.kv_head_idx*GQA_RATIO + laneid});

                tma::store_async_wait();
                asm volatile("{fence.acq_rel.gpu;}");
                warp::sync(); // ensure all writes are committed

                if (laneid == 0) {
                    int batch_block_idx = inst.batch_idx / globals::matmul_batch_block_size;
                    atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, batch_block_idx, 0}], 1); // this is sufficient
                }
            }
        };
    };
}
