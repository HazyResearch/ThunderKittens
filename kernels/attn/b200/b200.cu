#include "kittens.cuh"
#include "prototype.cuh"
#include <iostream>

constexpr int NUM_CONSUMERS = (2); 
constexpr int WARPGROUPS_PER_CONSUMER = (2);
constexpr int NUM_PRODUCERS = (1);

constexpr int NUM_WARPS = (NUM_CONSUMERS*WARPGROUPS_PER_CONSUMER + NUM_PRODUCERS) * 4;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

using namespace kittens;

struct rescale_add {
    template<typename T> static __device__ inline T op(const T &a, const T &b) {
        if constexpr (std::is_same_v<T, float2>) {
            constexpr float2 scale = {1.44269504089f*0.08838834764f, 1.44269504089f*0.08838834764f};
            float2 c;
            asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(*(uint64_t*)&c) : "l"(*(uint64_t*)&a), "l"(*(uint64_t*)&scale), "l"(*(uint64_t*)&b));
            return c;
        }
        else {
            static_assert(sizeof(T) == 999, "Currently unsupported type");
        }
    }
};
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void rescale_add_row(T &dst, const T &src, const V &row_values) {
    row_map<rescale_add, T, V>(dst, src, row_values);
}

template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (128);
    constexpr static int kv_height  = (128);
    constexpr static int stages     = (2); 
};

template<int D=128> struct fwd_globals {
    using q_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height,   fwd_attend_ker_tile_dims<D>::tile_width>  ;
    using k_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height/2, fwd_attend_ker_tile_dims<D>::tile_width>  ; // since we're using a two-SM dispatch, split on N dim.
    using v_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height,   fwd_attend_ker_tile_dims<D>::tile_width/2>; // since we're using a two-SM dispatch, split on N dim.
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height,   fwd_attend_ker_tile_dims<D>::tile_width>> ;
    using o_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height,   fwd_attend_ker_tile_dims<D>::tile_width>  ;

    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;
};

template<int D=128> struct softmax_registers {
    static constexpr int rows = fwd_attend_ker_tile_dims<D>::qo_height/(WARPGROUPS_PER_CONSUMER*WARPGROUP_WARPS), cols = fwd_attend_ker_tile_dims<D>::kv_height;
    rt_fl<rows, cols> att_block;
    rt_bf<rows, cols> att_block_mma;
    col_vec<rt_fl<rows, cols>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;
    __device__ inline void init() {
        neg_infty(max_vec);
        zero(norm_vec);
    }
};
// __device__ static inline void softmax(softmax_registers &regs) {
//     regs.max_vec_last_scaled = regs.max_vec * 1.44269504089f*0.08838834764f;
//     row_max(regs.max_vec, regs.att_block, regs.max_vec);
//     regs.max_vec_scaled = regs.max_vec * -1.44269504089f*0.08838834764f;
//     rescale_add_row(regs.att_block, regs.att_block, regs.max_vec_scaled);
//     regs.att_block = exp2(regs.att_block);
//     regs.max_vec_last_scaled += regs.max_vec_scaled;
//     regs.max_vec_last_scaled = exp2(regs.max_vec_last_scaled);
//     regs.norm_vec *= regs.max_vec_last_scaled;
//     row_sum(regs.norm_vec, regs.att_block, regs.norm_vec);
//     copy(regs.att_block_mma, regs.att_block);
// }
template<int D> __device__ static inline void softmax(softmax_registers<D> &regs) {
    mul(regs.max_vec_last_scaled, regs.max_vec, 1.44269504089f*0.08838834764f);
    row_max(regs.max_vec, regs.att_block, regs.max_vec);
    mul(regs.max_vec_scaled, regs.max_vec, -1.44269504089f*0.08838834764f);
    rescale_add_row(regs.att_block, regs.att_block, regs.max_vec_scaled);
    exp2(regs.att_block, regs.att_block);
    add(regs.max_vec_last_scaled, regs.max_vec_last_scaled, regs.max_vec_scaled);
    exp2(regs.max_vec_last_scaled, regs.max_vec_last_scaled);
    mul(regs.norm_vec, regs.norm_vec, regs.max_vec_last_scaled);
    row_sum(regs.norm_vec, regs.att_block, regs.norm_vec);
    copy(regs.att_block_mma, regs.att_block);
}

__device__ static inline int get_iters_per_task(const fwd_globals<> &g) {
    return g.k.rows / fwd_globals<>::v_tile::rows;
}
__device__ static inline int3 get_task_idx(const fwd_globals<> &g, int task_iter) {
    int cluster_x = clusterIdx().x, ctarank = cluster_ctarank();
    int task_id = task_iter * (gridDim.x/2) + cluster_x;
    constexpr int q_rows_per_task = 2 * NUM_CONSUMERS*fwd_globals<>::q_tile::rows;
    int seq_q = (g.q.rows + q_rows_per_task - 1)/(q_rows_per_task);
    int3 task_idx;
    task_idx.x = task_id / (seq_q*g.k.depth);
    task_id -= task_idx.x * seq_q * g.k.depth;
    task_idx.y  = task_id / seq_q;
    task_id -= task_idx.y  * seq_q;
    task_idx.z   = task_id;
    if(task_idx.x >= g.q.batch) return { -1, -1, -1 };
    return task_idx;
}

template<int D=128, bool causal=false> __global__ __cluster_dims__(2) __launch_bounds__(NUM_THREADS, 1)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
    static_assert(!causal, "Causal attention not supported yet");
    static_assert(D==128, "Only D=128 is supported");
    using K = fwd_attend_ker_tile_dims<D>;
    using G = fwd_globals<D>;

    using consumer = group<4*WARPGROUPS_PER_CONSUMER>;
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpgroupid = warpgroup::groupid(), consumerid = warpgroupid==NUM_CONSUMERS*WARPGROUPS_PER_CONSUMER ? 0 : consumer::groupid();
    int ctarank = cluster_ctarank(); 
    int iters_per_task = get_iters_per_task(g);

    using q_tile    = G::q_tile;
    using k_tile    = G::k_tile;
    using v_tile    = G::v_tile;
    using l_col_vec = G::l_col_vec;
    using o_tile    = G::o_tile;
    
    q_tile    (&q_smem)[NUM_CONSUMERS] = al.allocate<q_tile, NUM_CONSUMERS>();
    k_tile    (&k_smem)[K::stages]     = al.allocate<k_tile, K::stages>();
    v_tile    (&v_smem)[K::stages]     = al.allocate<v_tile, K::stages>();
    l_col_vec (&l_smem)[NUM_CONSUMERS] = al.allocate<l_col_vec, NUM_CONSUMERS>();
    auto      (*o_smem)                = reinterpret_cast<o_tile(*)>(&q_smem);
    st_bf<128,128> (&att_smem)[NUM_CONSUMERS] = al.allocate<st_bf<128,128>, NUM_CONSUMERS>();

    auto all_tmem = allocate_tmem<1, 2>();
    using att_tm_fl_t = tmem<float, K::qo_height, K::kv_height>;
    using att_tm_bf_t = tmem<bf16,  K::qo_height, K::kv_height>;
    using o_tm_fl_t   = tmem<float, K::qo_height, K::tile_width>;
    att_tm_fl_t att_tm    = all_tmem.subtile<att_tm_fl_t>(0, consumerid*K::kv_height);
    o_tm_fl_t   o_tm      = all_tmem.subtile<o_tm_fl_t>  (0, (NUM_CONSUMERS*K::kv_height) + consumerid*K::tile_width);
    att_tm_bf_t att_bf_tm = reinterpret_cast<att_tm_bf_t&>(att_tm);

    __shared__ kittens::semaphore q_smem_arrived[NUM_CONSUMERS],
                                  k_smem_arrived[K::stages], v_smem_arrived[K::stages],
                                  k_smem_finished[K::stages], v_smem_finished[K::stages],
                                  attn_unloaded[NUM_CONSUMERS],
                                  attn_mma_stored[NUM_CONSUMERS],
                                  qk_matmul_done[NUM_CONSUMERS], av_matmul_done[NUM_CONSUMERS];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 0) {
        for(int i = 0; i < K::stages; i++) {
            init_semaphore(k_smem_arrived[i], 0, 2);
            init_semaphore(v_smem_arrived[i], 0, 2);
            init_semaphore(k_smem_finished[i], 0, NUM_CONSUMERS); 
            init_semaphore(v_smem_finished[i], 0, NUM_CONSUMERS); 
        }
        for(int i = 0; i < NUM_CONSUMERS; i++) {
            init_semaphore(q_smem_arrived[i], 0, 2); 
            init_semaphore(attn_unloaded[i], 0, 2*8);
            init_semaphore(attn_mma_stored[i], 0, 2*8);
            init_semaphore(qk_matmul_done[i], 0, 1);
            init_semaphore(av_matmul_done[i], 0, 1);
        }
    }

    tma::cluster::sync();
    
    if(warpgroupid == NUM_CONSUMERS*WARPGROUPS_PER_CONSUMER) {
        warpgroup::decrease_registers<64>();
        if(ctarank == 0 && warpgroup::warpid() == 0) { // launch the QK MMA's
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int3 batchheadrow = get_task_idx(g, task_iter);
                if(batchheadrow.x == -1) return;
                for(int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(k_smem_arrived[input_ring], prototype::get_phasebit<0>(bitfield, input_ring));
                    #pragma unroll
                    for(int i = 0; i < NUM_CONSUMERS; i++) {
                        if(idx == 0) tma::cluster::wait(q_smem_arrived[i], task_iter%2);    // make sure Q is loaded
                        tma::cluster::wait(attn_unloaded[i], prototype::get_phasebit<1>(bitfield, i)); // make sure ready to launch the next one.
                        auto att_tm_i = att_tm.template subtile<att_tm_fl_t>(0, i*K::kv_height);
                        mm2_ABt(att_tm_i, q_smem[i], k_smem[input_ring], qk_matmul_done[i]);
                        prototype::update_phasebit<1>(bitfield, i);
                    }
                    prototype::update_phasebit<0>(bitfield, input_ring);
                    input_ring=prototype::ring_advance<K::stages>(input_ring);
                }
            }
        }
        else if(ctarank == 0 && warpgroup::warpid() == 1) { // launch the AV MMA's
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int3 batchheadrow = get_task_idx(g, task_iter);
                if(batchheadrow.x == -1) return;
                for(int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(v_smem_arrived[input_ring], prototype::get_phasebit<0>(bitfield, input_ring));
                    #pragma unroll
                    for(int i = 0; i < NUM_CONSUMERS; i++) {
                        tma::cluster::wait(attn_mma_stored[i], prototype::get_phasebit<0>(bitfield, K::stages+i));
                        auto o_tm_i = o_tm.template subtile<o_tm_fl_t>(0, i*K::tile_width);
                        mma2_AB(o_tm_i, att_smem[i], v_smem[input_ring], av_matmul_done[i]);
                        prototype::update_phasebit<0>(bitfield, K::stages+i);
                    }
                    prototype::update_phasebit<0>(bitfield, input_ring);
                    input_ring=prototype::ring_advance<K::stages>(input_ring);
                }
            }
        }
        else if(warpgroup::warpid() == 2) { // K loader
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int3 batchheadrow = get_task_idx(g, task_iter);
                if(batchheadrow.x == -1) return;
                for (int idx = 0; idx < iters_per_task; idx++) {
                    kittens::wait(k_smem_finished[input_ring], prototype::get_phasebit<1>(bitfield, input_ring));
                    tma::cluster::expect(k_smem_arrived[input_ring], 0, k_smem[input_ring]);
                    tma::cluster::load_async(k_smem[input_ring], g.k, {batchheadrow.x, batchheadrow.y, 2*idx + ctarank, 0},
                                            k_smem_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    prototype::update_phasebit<1>(bitfield, input_ring);
                    input_ring=prototype::ring_advance<K::stages>(input_ring);
                    __syncwarp();
                }
            }
        }
        else if(warpgroup::warpid() == 3) { // V loader
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int3 batchheadrow = get_task_idx(g, task_iter);
                if(batchheadrow.x == -1)  return;
                for (int idx = 0; idx < iters_per_task; idx++) {
                    kittens::wait(v_smem_finished[input_ring], prototype::get_phasebit<1>(bitfield, input_ring));
                    tma::cluster::expect(v_smem_arrived[input_ring], 0, v_smem[input_ring]);
                    tma::cluster::load_async(v_smem[input_ring], g.v, {batchheadrow.x, batchheadrow.y, idx, ctarank},
                                             v_smem_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    prototype::update_phasebit<1>(bitfield, input_ring);
                    input_ring=prototype::ring_advance<K::stages>(input_ring);
                    __syncwarp();
                }
            }
        }
    }
    else {
        warpgroup::increase_registers<104>();
        using all_consumers = group<NUM_CONSUMERS*WARPGROUPS_PER_CONSUMER*WARPGROUP_WARPS>;
        using all_barrier = kittens::barrier<all_consumers::GROUP_WARPS>;
        all_barrier bar(15);
        
        softmax_registers<D> sr;
        rt_fl<K::qo_height/(WARPGROUPS_PER_CONSUMER*WARPGROUP_WARPS), K::tile_width> o_reg;
        int k_input_ring = 0, v_input_ring = 0;

        for(int task_iter = 0; true; task_iter++) {
            int3 batchheadrow = get_task_idx(g, task_iter);
            if(batchheadrow.x == -1) return;
            // Load Q matrices
            if(consumer::warpid() == 0) {
                tma::cluster::expect(q_smem_arrived[consumerid], 0, q_smem[consumerid]);
                tma::cluster::load_async(q_smem[consumerid], g.q,
                    {batchheadrow.x, batchheadrow.y, 2*NUM_CONSUMERS*batchheadrow.z + NUM_CONSUMERS*ctarank + consumerid, 0}, q_smem_arrived[consumerid], (uint16_t)(1<<ctarank), 0);
            }

            // Initialize register state
            sr.init();
            zero(o_reg);

            for(int idx = 0; idx < iters_per_task; idx++) {

                // wait for QK matmul
                tma::cluster::wait(qk_matmul_done[consumerid], prototype::get_phasebit<0>(bitfield, K::stages+0));
                prototype::update_phasebit<0>(bitfield, K::stages+0);
                if(consumer::laneid() == 0) arrive(k_smem_finished[k_input_ring]);
                k_input_ring=prototype::ring_advance<K::stages>(k_input_ring); // Advance the ring to the next input block

                consumer::load_async(sr.att_block, att_tm);
                tm_load_wait();
                __syncwarp();
                if(laneid() == 0) tma::cluster::arrive(attn_unloaded[consumerid], 0); // signal that we're ready to launch the next QK matmul for this consumer.
                
                // Do softmax and o register rescaling.
                softmax(sr);

                // Do the O rescaling, store the attention matrix, and signal to launch the next AV matmul.
                if(idx>0) { // Don't wait or signal on the first iteration.
                    tma::cluster::wait(av_matmul_done[consumerid], prototype::get_phasebit<0>(bitfield, K::stages+1)); // O must be ready for us to use.
                    if(consumer::laneid() == 0) arrive(v_smem_finished[v_input_ring]); // since that matmul finished, we can next the load the next block.
                    prototype::update_phasebit<0>(bitfield, K::stages+1);
                    v_input_ring=prototype::ring_advance<K::stages>(v_input_ring); // Advance the ring to the next input block
                }
                consumer::load_async(o_reg, o_tm);
                consumer::store(att_smem[consumerid], sr.att_block_mma);
                mul_row(o_reg, o_reg, sr.max_vec_last_scaled);
                consumer::store_async(o_tm, o_reg);
                tm_store_wait();
                __syncwarp();
                if(laneid() == 0) tma::cluster::arrive(attn_mma_stored[consumerid], 0);
                consumer::sync(consumerid);
            }

            // Wait for the last AV matmul to finish and store the output.
            tma::cluster::wait(av_matmul_done[consumerid], prototype::get_phasebit<0>(bitfield, K::stages+1)); // O must be ready for us to use.
            if(consumer::laneid() == 0) arrive(v_smem_finished[v_input_ring]); // since that matmul finished, we can next the load the next block.
            prototype::update_phasebit<0>(bitfield, K::stages+1);
            v_input_ring=prototype::ring_advance<K::stages>(v_input_ring); // Advance the ring to the next input block

            consumer::load_async(o_reg, o_tm);
            div_row(o_reg, o_reg, sr.norm_vec);
            consumer::store(o_smem[consumerid], o_reg);
            consumer::sync(consumerid);
            if(consumer::warpid() == 0) {
                tma::store_async(g.o, o_smem[consumerid], {batchheadrow.x, batchheadrow.y, 2*NUM_CONSUMERS*batchheadrow.z + NUM_CONSUMERS*ctarank + consumerid, 0});
                tma::store_async_read_wait();
            }
            consumer::sync(consumerid);
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// ----------------------------------- Backward preparation kernel -----------------------------------
// ---------------------------------------------------------------------------------------------------

template<int D>
struct bwd_prep_globals {
    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;

    og_gl og;
    o_gl  o;
    d_gl  d;
};

template<int D>
__global__  __launch_bounds__(4*kittens::WARP_THREADS, (D == 64) ? 2 : 1)
void bwd_attend_prep_ker(const __grid_constant__ bwd_prep_globals<D> g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();

    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    og_tile (&og_smem)[4] = al.allocate<og_tile, 4>();
    o_tile  (&o_smem) [4] = al.allocate<o_tile , 4>();
    d_tile  (&d_smem) [4] = al.allocate<d_tile , 4>();
    
    rt_fl<4*16, D> og_reg, o_reg; 
    col_vec<rt_fl<4*16, D>> d_reg;

    __shared__ kittens::semaphore smem_semaphore;

    if (threadIdx.x == 0) {
        init_semaphore(smem_semaphore, 0, 1);
        tma::expect_bytes(smem_semaphore, sizeof(og_smem[0]) * 4 * 2);
    }
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < 4; w++) {
            coord<o_tile> tile_idx = {blockIdx.z, blockIdx.y, (blockIdx.x * 4) + w, 0};
            tma::load_async(o_smem[w],  g.o,  tile_idx, smem_semaphore);
            tma::load_async(og_smem[w], g.og, tile_idx, smem_semaphore);
        }
    }

    kittens::wait(smem_semaphore, 0);
    load(o_reg, o_smem[warpid]);
    load(og_reg, og_smem[warpid]);
    mul(og_reg, og_reg, o_reg);
    row_sum(d_reg, og_reg);
    store(d_smem[warpid], d_reg);
    __syncthreads(); 

    if (warpid == 0) {
        for (int w = 0; w < 4; w++) {
            coord<d_tile> tile_idx = {blockIdx.z, blockIdx.y, 0, (blockIdx.x * 4) + w};
            tma::store_async(g.d, d_smem[w], tile_idx);
        }
    }
    tma::store_async_wait();
}

template<int D> struct bwd_attend_ker_tile_dims {};
template<> struct bwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int tile_h     = (4*16);
    constexpr static int tile_h_qo  = (4*16);
    constexpr static int blocks_sm = 1;
};
template<> struct bwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int tile_h     = (4*16);
    constexpr static int tile_h_qo  = (4*16);
    constexpr static int blocks_sm = 1; 
};

constexpr int BWD_CONSUMER_WARPGROUPS = (2); 
constexpr int BWD_PRODUCER_WARPGROUPS = (1); 
constexpr int BWD_NUM_WARPGROUPS      = (BWD_CONSUMER_WARPGROUPS+BWD_PRODUCER_WARPGROUPS); 
constexpr int BWD_NUM_WORKERS         = (BWD_NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 

template<int D>
struct bwd_globals {
    using G = bwd_attend_ker_tile_dims<D>;

    using q_tile  =         st_bf<G::tile_h_qo, G::tile_width>;
    using k_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using v_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using og_tile =         st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile =         st_fl<G::tile_h_qo, G::tile_width>;
    using kg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using vg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using l_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

    using q_gl  = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl  = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl  = gl<bf16,  -1, -1, -1, -1, v_tile>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;

    using qg_gl = gl<float, -1, -1, -1, -1, qg_tile>;
    using kg_gl = gl<float, -1, -1, -1, -1, kg_tile>;
    using vg_gl = gl<float, -1, -1, -1, -1, vg_tile>;

    using l_gl  = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>; 

    q_gl  q;
    k_gl  k;
    v_gl  v;
    og_gl og;
    qg_gl qg;
    kg_gl kg;
    vg_gl vg;
    l_gl  l;
    d_gl  d;

    const int N;
    const int hr;
};

__device__ static inline void
stream_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(kittens::laneid()%4);
        reg_tile.tiles[0][i].data[0] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[1] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[2] = *(float2*)&smem_vec[tic][base_col + 8];
        reg_tile.tiles[0][i].data[3] = *(float2*)&smem_vec[tic][base_col + 8];
    }
}

__device__ static inline void
stream_sub_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(laneid()%4);
        reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[0], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[1], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[2], *(float2*)&smem_vec[tic][base_col + 8]);
        reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[3], *(float2*)&smem_vec[tic][base_col + 8]);
    }
}

template<int tile_h_qo, int tile_h>
__device__ static inline void 
causal_mask(auto &reg_tile, int qo_idx) {
    int q_blk = (qo_idx) * (tile_h_qo/kittens::TILE_ROW_DIM<bf16>);
    int k_blk = (blockIdx.x * BWD_CONSUMER_WARPGROUPS * (tile_h/kittens::TILE_ROW_DIM<bf16>)) 
                + ((kittens::warpid()/kittens::WARPGROUP_WARPS) * (tile_h/kittens::TILE_ROW_DIM<bf16>)) 
                + (kittens::warpid() % kittens::WARPGROUP_WARPS);

    for (int j = 0; j < (tile_h_qo/kittens::TILE_ROW_DIM<bf16>); j++) {
        int q_idx = q_blk + j;
        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(reg_tile.tiles[0][j]);
        if      (q_idx  < k_blk) { neg_infty(attn_subtile); }
        else if (q_idx == k_blk) { make_causal_t(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
    }
}

template<bool is_causal, int tile_h_qo, int tile_h, int tile_width, int D>
__device__ static inline void
compute_bwd_loop(
        kittens::semaphore *vec_b, kittens::semaphore *q_b, kittens::semaphore *o_b, 
        rt_fl<16, 64> &s_block_t, rt_fl<16, 64> &dp_block_t, 
        rt_fl<16, 64> &p_block_t, rt_fl<16, 64> &ds_block_t,  
        rt_bf<16, 64> &p_block_t_mma, rt_bf<16, 64> &ds_block_t_mma, 
        tmem<float, 64, tile_width> &qg_tm, tmem<float, 64, tile_width> &kg_tm, tmem<float, 64, tile_width> &vg_tm, 
        tmem<float, 64, 64> &s_tm, tmem<float, 64, 64> &p_tm, tmem<bf16, 64, 64> &p_tm_bf, 
        rt_fl<16, tile_width> &kg_reg, rt_fl<16, tile_width> &vg_reg,
        auto &q_smem, auto &k_smem, auto &v_smem, 
        auto &og_smem, auto &ds_smem, auto &l_smem, auto &d_smem,
        auto &kv_mma_semaphore,
        int qo_idx, int q_start, int tic, int toc) 
{
    kittens::wait(vec_b[tic], ((qo_idx - q_start)/2)%2);
    stream_tile(s_block_t, l_smem, tic);
    kittens::wait(q_b[tic], ((qo_idx - q_start)/2)%2);

    // warpgroup::mma_ABt(s_block_t, k_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], q_smem[tic]);
    warpgroup::store_async(s_tm, s_block_t);
    tm_store_wait();
    if (kittens::warpid() % 4 == 0) {
        mma_ABt(s_tm, k_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], q_smem[tic], kv_mma_semaphore); 
    }
    
    kittens::wait(o_b[tic], ((qo_idx - q_start)/2)%2);
    // warpgroup::mm_ABt(dp_block_t, v_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], og_smem[tic]);
    // warpgroup::mma_async_wait();
    if (kittens::warpid() % 4 == 0) {
        mm_ABt(p_tm, v_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], og_smem[tic], kv_mma_semaphore); 
    }
    kittens::wait(kv_mma_semaphore, 0); 
    warpgroup::load_async(dp_block_t, p_tm);
    warpgroup::load_async(s_block_t, s_tm);

    if constexpr (D == 64) { mul(s_block_t, s_block_t, 1.44269504089f*0.125f); }
    else                   { mul(s_block_t, s_block_t, 1.44269504089f*0.08838834764f); }

    if constexpr (is_causal) { causal_mask<tile_h_qo, tile_h>(s_block_t, qo_idx); }

    exp2(s_block_t, s_block_t);
    copy(p_block_t, s_block_t);
    copy(p_block_t_mma, s_block_t);
    stream_sub_tile(dp_block_t, d_smem, tic);
    mul(ds_block_t, p_block_t, dp_block_t);

    if constexpr (D == 64) { mul(ds_block_t, ds_block_t, 0.125f); }
    else                   { mul(ds_block_t, ds_block_t, 0.08838834764f); }

    // warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[tic]);
    warpgroup::store_async(vg_tm, vg_reg);
    warpgroup::store_async(p_tm_bf, p_block_t_mma);
    tm_store_wait();
    if (kittens::warpid() % 4 == 0) {
        mma_AB(vg_tm, p_tm_bf, og_smem[tic], kv_mma_semaphore); 
    }

    copy(ds_block_t_mma, ds_block_t);

    // warpgroup::store(ds_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], ds_block_t);
    // warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[tic]);
    // warpgroup::mma_async_wait();
    warpgroup::store(ds_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], ds_block_t);
    group<4>::sync(warpgroup::groupid()+4);
    warpgroup::store_async(kg_tm, kg_reg);
    tm_store_wait();
    if (kittens::warpid() % 4 == 0) {
        mma_AB(kg_tm, ds_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], q_smem[tic], kv_mma_semaphore); 
    }
    kittens::wait(kv_mma_semaphore, 1);
    warpgroup::load_async(kg_reg, kg_tm);
    warpgroup::load_async(vg_reg, vg_tm);

    group<8>::sync(10); 
}

template<typename kg_tile, typename vg_tile>
__device__ static inline void 
kv_store(auto &kg_smem, auto &kg_reg, 
         auto &vg_smem, auto &vg_reg,
         auto &dst, auto &bar, int kv_head_idx, int toc) 
{
    group<8>::sync(10); 
    warpgroup::store(kg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], kg_reg);

    group<4>::sync(warpgroup::groupid()+4);
    if (kittens::warpid() % 4 == 0) {
        coord<kg_tile> tile_idx = {blockIdx.z, kv_head_idx, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + (kittens::warpid()/kittens::WARPGROUP_WARPS), 0};
        tma::store_add_async(dst.kg, kg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], tile_idx);
        tma::store_commit_group();
    }

    kittens::wait(bar, toc);
    warpgroup::store(vg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], vg_reg);
    group<4>::sync(warpgroup::groupid()+4);

    if (kittens::warpid() % 4 == 0) {
        coord<vg_tile> tile_idx = {blockIdx.z, kv_head_idx, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + (kittens::warpid()/kittens::WARPGROUP_WARPS), 0};
        tma::store_add_async(dst.vg, vg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], tile_idx);
        tma::store_commit_group();
    }
    tma::store_async_wait(); 
}

template<int D, bool is_causal>
__global__ __launch_bounds__(BWD_NUM_WORKERS*kittens::WARP_THREADS, bwd_attend_ker_tile_dims<D>::blocks_sm)
void bwd_attend_ker(const __grid_constant__ bwd_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int N = g.N, hr = g.hr;
    using G = bwd_attend_ker_tile_dims<D>;
    
    using kg_tile   = st_fl<G::tile_h, G::tile_width>;
    using vg_tile   = st_fl<G::tile_h, G::tile_width>;
    using k_tile    = st_bf<G::tile_h, G::tile_width>;
    using v_tile    = st_bf<G::tile_h, G::tile_width>;
    using q_tile    = st_bf<G::tile_h_qo, G::tile_width>;
    using og_tile   = st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile   = st_fl<G::tile_h_qo, G::tile_width>;
    using l_tile    = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile    = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using attn_tile = st_bf<G::tile_h_qo, G::tile_h>; 

    k_tile  (&k_smem) [BWD_CONSUMER_WARPGROUPS] = al.allocate<k_tile, BWD_CONSUMER_WARPGROUPS>();
    v_tile  (&v_smem) [BWD_CONSUMER_WARPGROUPS] = al.allocate<v_tile, BWD_CONSUMER_WARPGROUPS>();

    q_tile  (&q_smem) [2] = al.allocate<q_tile,  2>(); 
    og_tile (&og_smem)[2] = al.allocate<og_tile, 2>(); 
    qg_tile (&qg_smem)    = al.allocate<qg_tile>();

    l_tile   (&l_smem)[2] = al.allocate<l_tile, 2>();
    d_tile   (&d_smem)[2] = al.allocate<d_tile, 2>();
    kg_tile (*kg_smem)    = reinterpret_cast<kg_tile*>(&k_smem[0].data[0]); 
    vg_tile (*vg_smem)    = reinterpret_cast<vg_tile*>(&q_smem[0].data[0]); 

    attn_tile (&ds_smem)[BWD_CONSUMER_WARPGROUPS] = al.allocate<attn_tile, BWD_CONSUMER_WARPGROUPS>();

    auto all_tmem = allocate_tmem<1, 2>();

    using qg_tm_tile = tmem<float, 64, G::tile_width>; 
    using kg_tm_tile = tmem<float, 64, G::tile_width>; 
    using vg_tm_tile = tmem<float, 64, G::tile_width>; 

    using s_tm_tile    = tmem<float, 64, 64>; 
    using p_tm_tile    = tmem<float, 64, 64>;
    using p_tm_bf_tile = tmem<bf16, 64, 64>;

    const int warpid      = kittens::warpid();
    const int warpgroupid = warpid/kittens::WARPGROUP_WARPS;
    const int qo_blocks   = N / (G::tile_h_qo);
    const int kv_head_idx = (blockIdx.y) / hr; 

    qg_tm_tile qg_tm = all_tmem.subtile<qg_tm_tile>(0, 0);
    kg_tm_tile kg_tm = all_tmem.subtile<kg_tm_tile>(0, G::tile_width + static_cast<int>(warpgroupid*G::tile_width));           
    vg_tm_tile vg_tm = all_tmem.subtile<vg_tm_tile>(0, G::tile_width + (2*G::tile_width) + static_cast<int>(warpgroupid*G::tile_width));

    s_tm_tile       s_tm = all_tmem.subtile<s_tm_tile>(16, 0);
    p_tm_tile       p_tm = all_tmem.subtile<p_tm_tile>(16, 64);
    p_tm_bf_tile p_tm_bf = reinterpret_cast<p_tm_bf_tile&>(p_tm);

    __shared__ kittens::semaphore kv_b, q_b[2], o_b[2], vec_b[2];
    __shared__ kittens::semaphore compute_done[2], qg_ready; 

    __shared__ kittens::semaphore q_mma_semaphore; 
    __shared__ kittens::semaphore kv_mma_semaphore[2]; 

    int tic = 0, toc = 1;
    const int q_start = (is_causal) ? (blockIdx.x * 2) : (0);

    if (threadIdx.x == 0) {
        init_semaphore(kv_b,  0, 1);
        init_semaphore(qg_ready, 1, 0);
        for (int s = 0; s < 2; s++) {
            init_semaphore(q_b[s],   0, 1);
            init_semaphore(o_b[s],   0, 1); 
            init_semaphore(vec_b[s], 0, 1);
            init_semaphore(compute_done[s], 1, 0);
            init_semaphore(kv_mma_semaphore[s], 0, 2); 
        }

        init_semaphore(q_mma_semaphore, 0, 2); 
        
        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * BWD_CONSUMER_WARPGROUPS);
        for (int w = 0; w < BWD_CONSUMER_WARPGROUPS; w++) {
            coord<k_tile> tile_idx = {blockIdx.z, kv_head_idx, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + w, 0};
            tma::load_async(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async(v_smem[w], g.v, tile_idx, kv_b);
        }

        coord<q_tile> tile_idx = {blockIdx.z, blockIdx.y, q_start, 0};
        tma::expect_bytes(q_b[tic],   sizeof(q_smem[0]));
        tma::load_async(q_smem[tic],  g.q,  tile_idx, q_b[tic]);
        tma::expect_bytes(o_b[tic],   sizeof(og_smem[0]));
        tma::load_async(og_smem[tic], g.og, tile_idx, o_b[tic]);

        coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, q_start};
        tma::expect_bytes(vec_b[tic], sizeof(l_smem[0]) + sizeof(d_smem[0]));
        tma::load_async(l_smem[tic], g.l, vec_idx, vec_b[tic]);
        tma::load_async(d_smem[tic], g.d, vec_idx, vec_b[tic]);
    }
    __syncthreads(); 

    if (warpgroupid == BWD_NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<24>();

        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            for (auto qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                if (qo_idx + 1 < qo_blocks) {
                    coord<q_tile> tile_idx = {blockIdx.z, blockIdx.y, qo_idx + 1, 0};
                    tma::expect_bytes(q_b[toc],   sizeof(q_smem[0])); 
                    tma::load_async(q_smem[toc], g.q,  tile_idx, q_b[toc]);
                    tma::expect_bytes(o_b[toc],   sizeof(og_smem[0]));
                    tma::load_async(og_smem[toc], g.og, tile_idx, o_b[toc]);

                    coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, qo_idx + 1};
                    tma::expect_bytes(vec_b[toc], sizeof(l_smem[0]) + sizeof(d_smem[0]));
                    tma::load_async(l_smem[toc], g.l, vec_idx, vec_b[toc]);
                    tma::load_async(d_smem[toc], g.d, vec_idx, vec_b[toc]);
                }
                
                kittens::wait(compute_done[tic], ((qo_idx - q_start)/(2))%2);
            }
        }
        else if(warpid % WARPGROUP_WARPS == 1) {
            for (auto qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                kittens::wait(compute_done[tic], ((qo_idx - q_start)/(2))%2);
                
                coord<qg_tile> tile_idx = {blockIdx.z, blockIdx.y, qo_idx, 0};
                tma::store_add_async(g.qg, qg_smem, tile_idx);
                tma::store_async_wait();
                
                if(laneid() == 0) arrive(qg_ready); 
            }
        }
    }
    else {
        rt_fl<16, G::tile_width> kg_reg, vg_reg;
    
        row_vec<rt_fl<16, 64>> row_reg; 

        rt_fl<16, 64> s_block_t, p_block_t; 
        rt_fl<16, 64> ds_block_t, dp_block_t; 
        rt_bf<16, 64> ds_block_t_mma, p_block_t_mma;

        zero(kg_reg);
        zero(vg_reg);

        if (warpgroupid == 0) {
            warpgroup::increase_registers<256>();
            kittens::wait(kv_b, 0);
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                compute_bwd_loop<is_causal, G::tile_h_qo, G::tile_h, G::tile_width, D>(
                    vec_b, q_b, o_b,
                    s_block_t, dp_block_t, p_block_t, ds_block_t, p_block_t_mma, ds_block_t_mma,
                    qg_tm, kg_tm, vg_tm, s_tm, p_tm, p_tm_bf,
                    kg_reg, vg_reg,
                    q_smem, k_smem, v_smem, og_smem, ds_smem, l_smem, d_smem,
                    kv_mma_semaphore[warpgroupid],
                    qo_idx, q_start, tic, toc
                );

                rt_fl<16, G::tile_width> qg_reg; 
                // warpgroup::mm_AtB(qg_reg, ds_smem[0], k_smem[0]);
                // warpgroup::mma_AtB(qg_reg, ds_smem[1], k_smem[1]);
                if (kittens::warpid() % 4 == 0) {
                    mm_AtB(qg_tm, ds_smem[0], k_smem[0], q_mma_semaphore);
                    mma_AtB(qg_tm, ds_smem[1], k_smem[1], q_mma_semaphore);
                }
    
                kittens::wait(qg_ready, toc);
                if (qo_idx > 0) tma::store_async_wait();

                // warpgroup::mma_async_wait();
                // warpgroup::store(qg_smem, qg_reg);
                // group<4>::sync(warpgroup::groupid()+4);
                kittens::wait(q_mma_semaphore, (qo_idx - q_start)%2);
                warpgroup::load_async(qg_reg, qg_tm);
                warpgroup::store(qg_smem, qg_reg);
                group<4>::sync(warpgroup::groupid()+4);
    
                if (warpgroup::laneid() == 0) arrive(compute_done[tic]);
            }
            kv_store<kg_tile, vg_tile>(kg_smem, kg_reg, 
                                       vg_smem, vg_reg,
                                       g, qg_ready, kv_head_idx, toc);
        }
        else {
            warpgroup::increase_registers<224>();
            kittens::wait(kv_b, 0);
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                compute_bwd_loop<is_causal, G::tile_h_qo, G::tile_h, G::tile_width, D>(
                    vec_b, q_b, o_b,
                    s_block_t, dp_block_t, p_block_t, ds_block_t, p_block_t_mma, ds_block_t_mma,
                    qg_tm, kg_tm, vg_tm, s_tm, p_tm, p_tm_bf,
                    kg_reg, vg_reg,
                    q_smem, k_smem, v_smem, og_smem, ds_smem, l_smem, d_smem,
                    kv_mma_semaphore[warpgroupid],
                    qo_idx, q_start, tic, toc
                );
            }
            kv_store<kg_tile, vg_tile>(kg_smem, kg_reg, 
                                       vg_smem, vg_reg,
                                       g, qg_ready, kv_head_idx, toc);
        }
    }
}

#include "harness.impl"
