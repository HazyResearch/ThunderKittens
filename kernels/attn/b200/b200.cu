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

#include "harness.impl"