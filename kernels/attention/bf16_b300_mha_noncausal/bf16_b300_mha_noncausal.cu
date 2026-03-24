#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

template <int _Mb, int _Nb, int _Dqk, int _Dvo>
struct config {
    static_assert(_Mb == 128,  "Mb must be 128");
    static_assert(_Nb == 128,  "Nb must be 128");
    static_assert(_Dqk == 192, "Dqk must be 192, other shapes will be supported in the future");
    static_assert(_Dvo == 128, "Dvo must be 128");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_SM = 148;

    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_CORRECTORS = 1;
    static constexpr int NUM_SOFTMAXXERS = 2;
    static constexpr int TOTAL_WGS = NUM_PRODUCERS + NUM_CORRECTORS + NUM_SOFTMAXXERS;
    static constexpr int NUM_WARPS = (TOTAL_WGS) * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_STAGES = 3;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;
};

template <typename C>
struct globals {
    using q_tile = st_bf<C::Mb, C::Dqk>;
    using k_tile = st_bf<C::Nb/2, C::Dqk>;
    using v_tile = st_bf<C::Nb, C::Dvo/2>;
    using o_tile = st_bf<C::Mb, C::Dvo>;

    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<k_tile, dim::DEPTH>>;
    using v_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<v_tile, dim::DEPTH>>;
    using o_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<o_tile, dim::DEPTH>>;
    using lse_gl = gl<float, -1, -1, -1, -1>;

    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;
    lse_gl lse;

    __host__ __inline__ dim3 grid() { return dim3(C::NUM_SM); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return C::DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ globals<C> g) {
    using G = globals<C>;

    const int cta_rank = cluster_ctarank();
    const int total_bids = g.q.batch() * g.q.rows() * (g.q.depth() / (C::Mb * C::NUM_SOFTMAXXERS));
    const int iters_per_task = g.k.depth() / C::Nb;

    auto get_tile_idx = [&](int block_idx) -> int3 {
        int cluster_linear = block_idx / C::CLUSTER_SIZE;
        int tiles_m = g.q.depth() / C::Mb;
        int clusters_m = tiles_m / (C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE);
        int clusters_per_batch = g.q.rows() * clusters_m;
        int b   = cluster_linear / clusters_per_batch;
        int rem = cluster_linear - b * clusters_per_batch;
        int h   = rem / clusters_m;
        int m_cluster = rem - h * clusters_m;
        int m_tile_base_cluster = m_cluster * (C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE);
        return {b, m_tile_base_cluster, h};
    };

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    using d_tt_scores = tt<float, C::Mb, C::Nb>;
    using d_tt_scores_bf = tt<bf16, C::Mb, C::Nb>;
    using d_tt_scores_bf_1q = tt<bf16, C::Mb, C::Nb/4>;
    using v_quarter_tile = st_bf<C::Nb/4, C::Dvo/2>;
    using d_tt_outputs = tt<float, C::Mb, C::Dvo>;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    using max_vec_sv = sv_fl<C::Mb>;
    typename G::q_tile (&q_smem)[C::NUM_SOFTMAXXERS] = al.allocate<G::q_tile, C::NUM_SOFTMAXXERS>();
    typename G::k_tile (&kv_smem)[C::LOAD_STAGES] = al.allocate<G::k_tile, C::LOAD_STAGES>();
    typename G::o_tile (&o_smem)[1] = al.allocate<G::o_tile, 1>();
    max_vec_sv (&max_vec_smem)[C::NUM_SOFTMAXXERS] = al.allocate<max_vec_sv, C::NUM_SOFTMAXXERS>();
    max_vec_sv (&lse_smem)[C::NUM_SOFTMAXXERS] = al.allocate<max_vec_sv, C::NUM_SOFTMAXXERS>();

    __shared__ semaphore q_arrived[C::NUM_SOFTMAXXERS], q_finished[C::NUM_SOFTMAXXERS], kv_arrived[C::LOAD_STAGES], kv_finished[C::LOAD_STAGES];
    __shared__ semaphore scores_arrived[C::NUM_SOFTMAXXERS], norm_scores_arrived[C::NUM_SOFTMAXXERS], norm_scores_quarter_arrived[3][C::NUM_SOFTMAXXERS];
    __shared__ semaphore corr_arrived[C::NUM_SOFTMAXXERS], tile_arrived[C::NUM_SOFTMAXXERS];
    __shared__ semaphore rescale_finished[C::NUM_SOFTMAXXERS];

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<typename G::q_tile, dim::DEPTH>();
        g.k.template prefetch_tma<typename G::k_tile, dim::DEPTH>();
        g.v.template prefetch_tma<typename G::v_tile, dim::DEPTH>();
        g.o.template prefetch_tma<typename G::o_tile, dim::DEPTH>();
        #pragma unroll
        for (int i = 0; i < C::NUM_SOFTMAXXERS; i++) {
            init_semaphore(q_arrived[i], 0, 1);
            init_semaphore(scores_arrived[i], 0, 1);
            init_semaphore(norm_scores_arrived[i], 0, (4 + C::NUM_CORRECTORS) * C::CLUSTER_SIZE);
            #pragma unroll
            for (int q = 0; q < 3; q++) {
                init_semaphore(norm_scores_quarter_arrived[q][i], 0, 4 * C::CLUSTER_SIZE);
            }
            init_semaphore(corr_arrived[i], 0, 4);
            init_semaphore(tile_arrived[i], 0, 1);
            init_semaphore(rescale_finished[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < C::LOAD_STAGES; i++) {
            init_semaphore(kv_arrived[i], 0, 1);
            init_semaphore(kv_finished[i], 0, C::NUM_SOFTMAXXERS);
        }
        for (int i = 0; i < C::NUM_SOFTMAXXERS; i++) init_semaphore(q_finished[i], 0, 1);
    }

    everyone::tma::cluster::sync();

    if (warpgroup::groupid() == C::TOTAL_WGS - 1) {  // producer warpgroup
        warpgroup::decrease_registers<128>();
        if (warpgroup::warpid() == 3 && warp::elect_leader()) {  // tma warp
            int kv_idx = 0;
            int kv_phase = 1;
            int q_phase = 1;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += C::NUM_SM) {
                int3 t_coord = get_tile_idx(cur_bid);
                #pragma unroll
                for (int i = 0; i < C::NUM_SOFTMAXXERS; i++) {
                    tma::cluster::wait(q_finished[i], q_phase);
                    tma::cluster::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[i], g.q, {t_coord.x, (t_coord.y+cta_rank*C::NUM_SOFTMAXXERS)+i, t_coord.z, 0}, q_arrived[i], (uint16_t)(1<<cta_rank), 0);
                }
                for (int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(kv_finished[kv_idx], kv_phase);
                    tma::cluster::load_async<dim::DEPTH, cache_policy::NORMAL>(kv_smem[kv_idx], g.k, {t_coord.x, idx * C::CLUSTER_SIZE + cta_rank, t_coord.z, 0}, kv_arrived[kv_idx], (uint16_t)(1<<cta_rank), 0);
                    kv_idx++; if (kv_idx == C::LOAD_STAGES) { kv_idx = 0; kv_phase ^= 1; }

                    tma::cluster::wait(kv_finished[kv_idx], kv_phase);
                    tma::cluster::load_async<dim::DEPTH, cache_policy::NORMAL>(reinterpret_cast<typename G::v_tile&>(kv_smem[kv_idx]), g.v, {t_coord.x, idx, t_coord.z, cta_rank}, kv_arrived[kv_idx], (uint16_t)(1<<cta_rank), 0);
                    kv_idx++; if (kv_idx == C::LOAD_STAGES) { kv_idx = 0; kv_phase ^= 1; }
                }
                q_phase ^= 1;
            }
        }
        else if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {  // single mma warp for both q tiles
            d_tt_scores tt_scores[C::NUM_SOFTMAXXERS];
            d_tt_outputs tt_outputs[C::NUM_SOFTMAXXERS];
            #pragma unroll
            for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                tt_scores[qid] = tm_alloc.template allocate<d_tt_scores>(qid * (C::Nb + C::Dvo));
                tt_outputs[qid] = tm_alloc.template allocate<d_tt_outputs>(qid * (C::Nb + C::Dvo) + C::Nb);
            }
            int kv_idx = 0;
            int kv_phase = 0;
            for (int cur_bid = blockIdx.x, task_num = 0; cur_bid < total_bids; cur_bid += C::NUM_SM, task_num++) {
                int q_phase = task_num & 1;
                int norm_scores_phase = (task_num * iters_per_task) & 1;
                #pragma unroll
                for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                    tma::cluster::expect_bytes(q_arrived[qid], C::CLUSTER_SIZE * sizeof(G::q_tile));
                    tma::cluster::wait(q_arrived[qid], q_phase);
                }
                // first QK
                int k_slot = kv_idx;
                tma::cluster::expect_bytes(kv_arrived[k_slot], C::CLUSTER_SIZE * sizeof(G::k_tile));
                tma::cluster::wait(kv_arrived[k_slot], kv_phase);
                for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                    mm2_ABt(tt_scores[qid], q_smem[qid], kv_smem[k_slot], kv_finished[k_slot]);
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[qid]);
                }
                if (iters_per_task == 1) {
                    for (int i = 0; i < C::NUM_SOFTMAXXERS; i++)
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[i]);
                }
                kv_idx++; if (kv_idx == C::LOAD_STAGES) { kv_idx = 0; kv_phase ^= 1; }

                // repeat PV then QK
                for (int idx = 0; idx < iters_per_task - 1; idx++) {
                    int v_slot = kv_idx;
                    tma::cluster::expect_bytes(kv_arrived[v_slot], C::CLUSTER_SIZE * sizeof(G::v_tile));
                    tma::cluster::wait(kv_arrived[v_slot], kv_phase);
                    kv_idx++; if (kv_idx == C::LOAD_STAGES) { kv_idx = 0; kv_phase ^= 1; }
                    int k_slot = kv_idx;
                    auto* v_base = reinterpret_cast<const char*>(&kv_smem[v_slot]);
                    const v_quarter_tile* v_q[4];
                    #pragma unroll
                    for (int q = 0; q < 4; q++)
                        v_q[q] = reinterpret_cast<const v_quarter_tile*>(v_base + q * sizeof(v_quarter_tile));
                    #pragma unroll
                    for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                        tma::cluster::wait(norm_scores_arrived[qid], norm_scores_phase);
                        if (idx == 0) {
                            mm2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr), *v_q[0]);
                        }
                        else {
                            mma2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr), *v_q[0]);
                        }
                        #pragma unroll
                        for (int q = 1; q < 4; q++) {
                            tma::cluster::wait(norm_scores_quarter_arrived[q-1][qid], norm_scores_phase);
                            if (q == 3)
                                mma2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr + q * C::Nb/8), *v_q[q], kv_finished[v_slot]);
                            else
                                mma2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr + q * C::Nb/8), *v_q[q]);
                        }
                        if (qid == 0) {
                            tma::cluster::expect_bytes(kv_arrived[k_slot], C::CLUSTER_SIZE * sizeof(G::k_tile));
                            tma::cluster::wait(kv_arrived[k_slot], kv_phase);
                        }
                        mm2_ABt(tt_scores[qid], q_smem[qid], kv_smem[k_slot], kv_finished[k_slot]);
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[qid]);
                    }
                    if (idx == iters_per_task - 2) {
                        for (int i = 0; i < C::NUM_SOFTMAXXERS; i++) {
                            detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[i]);
                        }
                    }
                    kv_idx++; if (kv_idx == C::LOAD_STAGES) { kv_idx = 0; kv_phase ^= 1; }
                    norm_scores_phase ^= 1;
                }

                // last PV
                int v_slot = kv_idx;
                tma::cluster::expect_bytes(kv_arrived[v_slot], C::CLUSTER_SIZE * sizeof(G::v_tile));
                tma::cluster::wait(kv_arrived[v_slot], kv_phase);

                auto* v_base = reinterpret_cast<const char*>(&kv_smem[v_slot]);
                const v_quarter_tile* v_q[4];
                #pragma unroll
                for (int q = 0; q < 4; q++)
                    v_q[q] = reinterpret_cast<const v_quarter_tile*>(v_base + q * sizeof(v_quarter_tile));
                #pragma unroll
                for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                    tma::cluster::wait(norm_scores_arrived[qid], norm_scores_phase);
                    if (iters_per_task == 1) {
                        mm2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr), *v_q[0]);
                    }
                    else {
                        mma2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr), *v_q[0]);
                    }
                    #pragma unroll
                    for (int q = 1; q < 4; q++) {
                        tma::cluster::wait(norm_scores_quarter_arrived[q-1][qid], norm_scores_phase);
                        if (q == 3)
                            mma2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr + q * C::Nb/8), *v_q[q], kv_finished[v_slot]);
                        else
                            mma2_AB(tt_outputs[qid], d_tt_scores_bf_1q(tt_scores[qid].addr + q * C::Nb/8), *v_q[q]);
                    }
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[qid]);
                }

                kv_idx++; if (kv_idx == C::LOAD_STAGES) { kv_idx = 0; kv_phase ^= 1; }
                norm_scores_phase ^= 1;
            }
        }
    }
    else if (warpgroup::groupid() == C::TOTAL_WGS - 2) {  // correction warpgroup
        warpgroup::decrease_registers<48>();
        static constexpr int CORR_TILE = 16;
        using d_tt_chunk = tt<float, C::Mb, CORR_TILE>;
        d_tt_outputs tt_outputs[C::NUM_SOFTMAXXERS];
        for (int i = 0; i < C::NUM_SOFTMAXXERS; i++) {
            tt_outputs[i] = tm_alloc.template allocate<d_tt_outputs>(i * (C::Nb + C::Dvo) + C::Nb);
        }
        int end_phase = 0;
        int corr_phase = 0;
        #pragma unroll
        for (int i = 0; i < C::NUM_SOFTMAXXERS; i++) {
            warpgroup::tma::cluster::arrive(norm_scores_arrived[i], 0);
        }
        for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += C::NUM_SM) {
            wait(corr_arrived[0], corr_phase);
            warpgroup::arrive(rescale_finished[0]);
            wait(corr_arrived[1], corr_phase);
            warpgroup::arrive(rescale_finished[1]);
            corr_phase ^= 1;
            for (int idx = 1; idx < iters_per_task; idx++) {
                #pragma unroll
                for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                    wait(corr_arrived[qid], corr_phase);
                    float correction = max_vec_smem[qid][warpgroup::laneid()];
                    bool needs_rescale = __any_sync(0xFFFFFFFF, correction < 1.0f);
                    if (needs_rescale) {
                        float2 corr_2 = {correction, correction};
                        #pragma unroll
                        for (int col = 0; col < C::Dvo; col += CORR_TILE) {
                            float2 o_reg[CORR_TILE / 2];
                            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                                : "r"(tt_outputs[qid].addr + ((warpgroup::warpid() * 32) << 16) + col));
                            tensor_load_wait();
                            #pragma unroll
                            for (int ii = 0; ii < CORR_TILE / 2; ii++) {
                                o_reg[ii] = __fmul2_rn(o_reg[ii], corr_2);
                            }
                            asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
                                :: "f"(o_reg[0].x), "f"(o_reg[0].y), "f"(o_reg[1].x), "f"(o_reg[1].y),
                                   "f"(o_reg[2].x), "f"(o_reg[2].y), "f"(o_reg[3].x), "f"(o_reg[3].y),
                                   "f"(o_reg[4].x), "f"(o_reg[4].y), "f"(o_reg[5].x), "f"(o_reg[5].y),
                                   "f"(o_reg[6].x), "f"(o_reg[6].y), "f"(o_reg[7].x), "f"(o_reg[7].y),
                                   "r"(tt_outputs[qid].addr + ((warpgroup::warpid() * 32) << 16) + col));
                        }
                        tensor_store_wait();
                    }
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::tma::cluster::arrive(norm_scores_arrived[qid], 0);
                    warpgroup::arrive(rescale_finished[qid]);
                }
                corr_phase ^= 1;
            }
            // final normalization
            for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                wait(corr_arrived[qid], corr_phase);
                float row_sum = max_vec_smem[qid][warpgroup::laneid()];
                float row_max = lse_smem[qid][warpgroup::laneid()];
                warpgroup::arrive(rescale_finished[qid]);
                bool row_invalid = (row_sum == 0.0f) | (row_sum != row_sum);
                float inv_norm_s;
                asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_norm_s) : "f"(row_invalid ? 1.0f : row_sum));
                float2 inv_norm = {inv_norm_s, inv_norm_s};
                wait(tile_arrived[qid], end_phase);
                warpgroup::tma::store_async_read_wait<0>();
                warpgroup::sync(warpgroup::groupid()+1);
                constexpr int SUBTILE_COLS = 64;
                uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
                uint32_t row_offset = warpgroup::laneid() * SUBTILE_COLS * sizeof(bf16);
                for (int col = 0; col < C::Dvo; col += CORR_TILE) {
                    float2 o_reg[CORR_TILE / 2];
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                        : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                          "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                          "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                          "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                        : "r"(tt_outputs[qid].addr + ((warpgroup::warpid() * 32) << 16) + col));
                    uint32_t row_base = base_addr + (col / SUBTILE_COLS) * (C::Mb * SUBTILE_COLS * sizeof(bf16))
                                      + row_offset + (col % SUBTILE_COLS) * sizeof(bf16);
                    #pragma unroll
                    for (int i = 0; i < CORR_TILE / 2; i++) {
                        bf16_2 tmp = __float22bfloat162_rn(__fmul2_rn(o_reg[i], inv_norm));
                        uint32_t addr = row_base + i * 4;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(addr ^ (((addr & 0x380) >> 7) << 4)), "r"(*(uint32_t*)&tmp));
                    }
                }
                warpgroup::sync(warpgroup::groupid()+1);
                int3 t_coord = get_tile_idx(cur_bid);
                warpgroup::tma::store_async<dim::DEPTH, cache_policy::EVICT_FIRST>(g.o, o_smem[0], {t_coord.x, (t_coord.y+cta_rank*C::NUM_SOFTMAXXERS)+qid, t_coord.z, 0});
                warpgroup::tma::cluster::arrive(norm_scores_arrived[qid], 0);

                // compute LSE and store to global
                const float SCALE_LOG2 = 1.44269504089f / sqrtf(float(C::Dqk));  // log2(e) / sqrt(Dqk)
                constexpr float LN2 = 0.693147180559945f;
                float lse_val;
                if (!row_invalid) {
                    float log2_row_sum;
                    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_row_sum) : "f"(row_sum));
                    lse_val = (row_max * SCALE_LOG2 + log2_row_sum) * LN2;
                }
                else {
                    lse_val = base_types::constants<float>::neg_infty();
                }
                int m_tile = t_coord.y + cta_rank * C::NUM_SOFTMAXXERS + qid;
                g.lse[{t_coord.x, t_coord.z, 0, m_tile * C::Mb + warpgroup::laneid()}] = lse_val;
                
            }
            corr_phase ^= 1;
            end_phase ^= 1;
        }
    }
    else if (warpgroup::groupid() < C::NUM_SOFTMAXXERS) {  // softmax warpgroups
        warpgroup::increase_registers<168>();
        d_tt_scores tt_scores[1];
        d_tt_scores_bf tt_scores_bf[1];
        tt_scores[0] = tm_alloc.template allocate<d_tt_scores>(warpgroup::groupid()*(C::Nb + C::Dvo));
        tt_scores_bf[0] = d_tt_scores_bf(tt_scores[0].addr);
        int scores_phase = 0;
        int rescale_phase = 1;
        uint32_t score_tt_base = tt_scores[0].addr + ((warpgroup::warpid() * 32) << 16);
        uint32_t score_bf_base = tt_scores_bf[0].addr + ((warpgroup::warpid() * 32) << 16);
        for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += C::NUM_SM) {
            float row_sum = 0.0f;
            float row_max = base_types::constants<float>::neg_infty();
            wait(rescale_finished[warpgroup::groupid()], rescale_phase);
            rescale_phase ^= 1;
            for (int idx = 0; idx < iters_per_task; idx++) {
                float2 scores_reg[C::Nb / 2];  // each thread holds one row of tmem
                wait(scores_arrived[warpgroup::groupid()], scores_phase);
                #pragma unroll
                for (int ii = 0; ii < C::Nb / 32; ii++) {
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                        : "=f"(scores_reg[ii * 16 + 0].x), "=f"(scores_reg[ii * 16 + 0].y), "=f"(scores_reg[ii * 16 + 1].x), "=f"(scores_reg[ii * 16 + 1].y), "=f"(scores_reg[ii * 16 + 2].x), "=f"(scores_reg[ii * 16 + 2].y), "=f"(scores_reg[ii * 16 + 3].x), "=f"(scores_reg[ii * 16 + 3].y),
                          "=f"(scores_reg[ii * 16 + 4].x), "=f"(scores_reg[ii * 16 + 4].y), "=f"(scores_reg[ii * 16 + 5].x), "=f"(scores_reg[ii * 16 + 5].y), "=f"(scores_reg[ii * 16 + 6].x), "=f"(scores_reg[ii * 16 + 6].y), "=f"(scores_reg[ii * 16 + 7].x), "=f"(scores_reg[ii * 16 + 7].y),
                          "=f"(scores_reg[ii * 16 + 8].x), "=f"(scores_reg[ii * 16 + 8].y), "=f"(scores_reg[ii * 16 + 9].x), "=f"(scores_reg[ii * 16 + 9].y), "=f"(scores_reg[ii * 16 + 10].x), "=f"(scores_reg[ii * 16 + 10].y), "=f"(scores_reg[ii * 16 + 11].x), "=f"(scores_reg[ii * 16 + 11].y),
                          "=f"(scores_reg[ii * 16 + 12].x), "=f"(scores_reg[ii * 16 + 12].y), "=f"(scores_reg[ii * 16 + 13].x), "=f"(scores_reg[ii * 16 + 13].y), "=f"(scores_reg[ii * 16 + 14].x), "=f"(scores_reg[ii * 16 + 14].y), "=f"(scores_reg[ii * 16 + 15].x), "=f"(scores_reg[ii * 16 + 15].y)
                        : "r"(score_tt_base + ii * 32));
                }

                const float SCALE_LOG2 = 1.44269504089f / sqrtf(float(C::Dqk));  // log2(e) / sqrt(Dqk)
                float row_max_old = row_max;
                // calculate row max
                float lm0 = base_types::constants<float>::neg_infty(), lm1 = base_types::constants<float>::neg_infty(), lm2 = base_types::constants<float>::neg_infty(), lm3 = base_types::constants<float>::neg_infty();
                float lm4 = base_types::constants<float>::neg_infty(), lm5 = base_types::constants<float>::neg_infty(), lm6 = base_types::constants<float>::neg_infty(), lm7 = base_types::constants<float>::neg_infty();
                #pragma unroll
                for (int j = 0; j < C::Nb / 2; j += 8) {
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(scores_reg[j + 0].x), "f"(scores_reg[j + 0].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm1) : "f"(lm1), "f"(scores_reg[j + 1].x), "f"(scores_reg[j + 1].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm2) : "f"(lm2), "f"(scores_reg[j + 2].x), "f"(scores_reg[j + 2].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm3) : "f"(lm3), "f"(scores_reg[j + 3].x), "f"(scores_reg[j + 3].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm4) : "f"(lm4), "f"(scores_reg[j + 4].x), "f"(scores_reg[j + 4].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm5) : "f"(lm5), "f"(scores_reg[j + 5].x), "f"(scores_reg[j + 5].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm6) : "f"(lm6), "f"(scores_reg[j + 6].x), "f"(scores_reg[j + 6].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm7) : "f"(lm7), "f"(scores_reg[j + 7].x), "f"(scores_reg[j + 7].y));
                }
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(lm1), "f"(lm2));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm4) : "f"(lm4), "f"(lm5), "f"(lm6));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(lm3), "f"(lm4));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(row_max) : "f"(row_max), "f"(lm0), "f"(lm7));

                float acc_scale = 1.0f;

                // give rescale factor to correction warpgroup
                if (idx > 0) {
                    constexpr float rescale_threshold = 8.f;
                    float acc_scale_ = (row_max_old - row_max) * SCALE_LOG2;
                    if (acc_scale_ >= -rescale_threshold) {
                        row_max = row_max_old;
                        acc_scale = 1.0f;
                    }
                    else {
                        acc_scale = exp2f(acc_scale_);
                    }
                    max_vec_smem[warpgroup::groupid()][warpgroup::laneid()] = acc_scale;
                }
                warp::sync();
                warp::arrive(corr_arrived[warpgroup::groupid()]);

                float neg_max_scaled = row_max * (-SCALE_LOG2);
                float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                constexpr int CONVERT_SIZE = 32;
                // scale, exp2, convert and store P in 4 quarters, signal after each
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    int ii = q;
                    bf16_2 scores_bf_reg[CONVERT_SIZE / 2];
                    #pragma unroll
                    for (int jj = 0; jj < 16; jj++) {
                        int idx = ii * 16 + jj;
                        scores_reg[idx] = __ffma2_rn(scores_reg[idx], scale_2, neg_max_scaled_2);
                        scores_reg[idx].x = exp2f(scores_reg[idx].x);
                        scores_reg[idx].y = exp2f(scores_reg[idx].y);
                        scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[idx]);
                    }
                    asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15};}"
                        :: "r"(*(uint32_t*)&scores_bf_reg[0]), "r"(*(uint32_t*)&scores_bf_reg[1]), "r"(*(uint32_t*)&scores_bf_reg[2]), "r"(*(uint32_t*)&scores_bf_reg[3]),
                           "r"(*(uint32_t*)&scores_bf_reg[4]), "r"(*(uint32_t*)&scores_bf_reg[5]), "r"(*(uint32_t*)&scores_bf_reg[6]), "r"(*(uint32_t*)&scores_bf_reg[7]),
                           "r"(*(uint32_t*)&scores_bf_reg[8]), "r"(*(uint32_t*)&scores_bf_reg[9]), "r"(*(uint32_t*)&scores_bf_reg[10]), "r"(*(uint32_t*)&scores_bf_reg[11]),
                           "r"(*(uint32_t*)&scores_bf_reg[12]), "r"(*(uint32_t*)&scores_bf_reg[13]), "r"(*(uint32_t*)&scores_bf_reg[14]), "r"(*(uint32_t*)&scores_bf_reg[15]),
                           "r"(score_bf_base + ii * 16));
                    tensor_store_wait();
                    if (q == 0)
                        warp::tma::cluster::arrive(norm_scores_arrived[warpgroup::groupid()], 0);
                    else
                        warp::tma::cluster::arrive(norm_scores_quarter_arrived[q-1][warpgroup::groupid()], 0);
                }
                wait(rescale_finished[warpgroup::groupid()], rescale_phase);
                rescale_phase ^= 1;

                // row sum
                float2 ls0 = {0.0f, 0.0f}, ls1 = {0.0f, 0.0f};
                #pragma unroll
                for (int ii = 0; ii < C::Nb / 2; ii += 2) {
                    ls0 = __fadd2_rn(ls0, scores_reg[ii]);
                    ls1 = __fadd2_rn(ls1, scores_reg[ii + 1]);
                }
                ls0 = __fadd2_rn(ls0, ls1);
                row_sum = row_sum * acc_scale + ls0.x + ls0.y;
                scores_phase ^= 1;
            }
            lse_smem[warpgroup::groupid()][warpgroup::laneid()] = row_max;
            max_vec_smem[warpgroup::groupid()][warpgroup::laneid()] = row_sum;
            warp::sync();
            warp::arrive(corr_arrived[warpgroup::groupid()]);
        }
    }
}

// pybind11 dispatch
void dispatch(pybind11::object Q_obj, pybind11::object K_obj, pybind11::object V_obj,
              pybind11::object O_obj, pybind11::object LSE_obj) {
    using C = config<128, 128, 192, 128>;
    using G = globals<C>;

    G g{
        py::from_object<typename G::q_gl>::make(Q_obj),
        py::from_object<typename G::k_gl>::make(K_obj),
        py::from_object<typename G::v_gl>::make(V_obj),
        py::from_object<typename G::o_gl>::make(O_obj),
        py::from_object<typename G::lse_gl>::make(LSE_obj)
    };

    CUDACHECK(cudaFuncSetAttribute(kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), 0, C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel<C>, g));
}

PYBIND11_MODULE(_C, m) {
    m.def("forward", &dispatch, "MHA forward (bf16, B300, non-causal)");
}
