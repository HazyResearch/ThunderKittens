#include "../../src/kittens.cuh"
#include <cooperative_groups.h>
using namespace kittens;

constexpr int NUM_WORKERS = 8, NUM_WARPGROUPS = (NUM_WORKERS/WARPGROUP_WARPS);
using layout_q = ducks::st_layout::wgmma_swizzle; 
using layout_k = ducks::st_layout::wgmma_swizzle; 
using layout_v = ducks::st_layout::wgmma_interleave;
using layout_o = ducks::st_layout::swizzle; 
template<int D> struct fwd_attend_ker_tile_dims {
    constexpr static int tile_width = D/kittens::TILE_DIM;
    constexpr static int qo_height  = 4;
    constexpr static int kv_height  = 512/D;
};

// the two cases (D=64, D=128) basically identical, but the barriers have been slightly tuned to eke out that extra few percent.
// current benchmarks (N=4096) are 460 TFLOPs at D=64 and 517 TFLOPs at D=128.
template<int N, int D=64> __global__  __launch_bounds__(NUM_WORKERS*kittens::WARP_THREADS, 2)
void attend_ker_fwd_train(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o, CUtensorMap* tma_l) {
    extern __shared__ int __shm[]; // dynamic shared memory
    tma_swizzle_allocator al((int*)&__shm[0]); // lightweight allocator enforces alignments.

    constexpr int tile_width = fwd_attend_ker_tile_dims<D>::tile_width; // constants
    constexpr int qo_height  = fwd_attend_ker_tile_dims<D>::qo_height;
    constexpr int kv_height  = fwd_attend_ker_tile_dims<D>::kv_height;
    constexpr int kv_blocks  = N / (kv_height*TILE_DIM);

    auto (&q_smem)[NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_q>, NUM_WARPGROUPS>(); // shared tiles
    auto (&k_smem)[2]              = al.allocate<st_bf<kv_height, tile_width, layout_k>, 2>();
    auto (&v_smem)[2]              = al.allocate<st_bf<kv_height, tile_width, layout_v>, 2>();
    auto (&l_smem)[NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_q>::col_vec, NUM_WARPGROUPS>();

    rt_fl<1, kv_height> att_block; // declare registers
    rt_bf<1, kv_height> att_block_mma;
    rt_fl<1, tile_width> o_accum;
    col_vec<rt_fl<1, kv_height>> max_vec_last, max_vec;
    col_vec<rt_fl<1, kv_height>> norm_vec_last, norm_vec;

    int warpid      = kittens::warpid(); // who am i? when am i?
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS; 
    int tic = 0, toc = 1, v_phase = 0; // since we have two barriers for v, we need a half-rate tic as the v phase bit

    __shared__ uint64_t qsmem_barrier, kv_smem_barriers[3]; // init barriers
    // D=64 and D=128 meaningfully (2% perf) prefer different barrier, so this is my hack to prevent splitting into separate kernels.
    uint64_t *kbar, *vbar[2];
    if constexpr (D==64)  { kbar = &kv_smem_barriers[0]; vbar[0] = &kv_smem_barriers[0]; vbar[1] = &kv_smem_barriers[0]; } // set all as aliases
    if constexpr (D==128) { kbar = &kv_smem_barriers[0]; vbar[0] = &kv_smem_barriers[1]; vbar[1] = &kv_smem_barriers[2]; } // separate barriers
    if      (warpid == 0) tma::init_barrier<typeof(q_smem[0]), NUM_WARPGROUPS>(qsmem_barrier);
    else if (warpid == 1) tma::init_barrier<typeof(k_smem[0]), 128/D         >(*kbar);
    if constexpr(D==128) {
        if      (warpid == 2) tma::init_barrier<typeof(v_smem[0])>(*vbar[tic]);
        else if (warpid == 3) tma::init_barrier                   (*vbar[toc]); // will set bytes later anyways.
    }
    __syncthreads();

    if (warpid%4 == 0) { // load q from HBM
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid;
        tma::load_async(q_smem[warpgroupid], tma_q, qsmem_barrier, tile_idx);
    }
    if      (warpid == 0) tma::load_async(k_smem[tic], tma_k,* kbar     , blockIdx.y*kv_blocks); // load initial k, v from HBM
    else if (warpid == 1) tma::load_async(v_smem[tic], tma_v, *vbar[tic], blockIdx.y*kv_blocks);

    neg_infty(max_vec); // zero registers, while we wait
    zero(norm_vec);
    zero(o_accum);

    tma::arrive_and_wait(qsmem_barrier, 0); // wait for memory to arrive
    if constexpr (D==64)  warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f)); // temperature adjustment
    if constexpr (D==128) warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.08838834764f)); // temperature adjustment

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic ^= 1, toc ^= 1) {

        tma::arrive_and_wait(*kbar, tic); // wait for k memory to arrive (and also v, too, if D=64)
        __syncthreads(); // everybody on the same page?
        if (warpid == 0) { // go get the next K from HBM
            tma::set_bytes(*kbar, detail::transfer_bytes<typeof(k_smem[0]), 128/D>::bytes);
            if constexpr (D==128) tma::set_bytes(*vbar[toc], detail::transfer_bytes<typeof(v_smem[0])>::bytes);
            if (kv_idx+1 < kv_blocks) {
                tma::load_async(k_smem[toc], tma_k, *kbar     , (blockIdx.y * kv_blocks) + kv_idx + 1);
                tma::load_async(v_smem[toc], tma_v, *vbar[toc], (blockIdx.y * kv_blocks) + kv_idx + 1);
            }
        }

        warpgroup::mma_fence(att_block); // qk matmul fence
        warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[tic]); // clear registers -- note mm_ABt, not mma_ABt.
        warpgroup::mma_commit_group(); // dew it

        copy(norm_vec_last, norm_vec); // copy registers, while we wait
        copy(max_vec_last,  max_vec);

        warpgroup::mma_async_wait(); // ding dong! matmuls arrived.

        row_max(max_vec, att_block, max_vec); // accumulate new max onto the max_vec
        sub_row(att_block, att_block, max_vec); // ensure all <=0 for numerics
        exp(att_block, att_block); // exponentiate attention block for softmax in FP32
        sub(max_vec_last, max_vec_last, max_vec); // how do we need to normalize previous O's due to new max, in log?
        exp(max_vec_last, max_vec_last); // how do we need to norm previous O's due to new max, actually?
        mul(norm_vec, norm_vec, max_vec_last); // norm previous exp sum using new max
        row_sum(norm_vec, att_block, norm_vec); // accumulate new exp sum onto the norm_vec
        div_row(att_block, att_block, norm_vec); // softmax normalization of existing attention block
        mul(norm_vec_last, norm_vec_last, max_vec_last); // incorporate previous max into norm for o
        div(norm_vec_last, norm_vec_last, norm_vec); // incorporate current norm into new norm for o
        copy(att_block_mma, att_block); // convert to bf16 for mma
        mul_row(o_accum, o_accum, norm_vec_last); // normalize o in advance of mma'ing onto it

        if constexpr (D==128) tma::arrive_and_wait(*vbar[tic], v_phase); // wait for v memory to arrive, if this is a real barrier

        warpgroup::mma_fence(o_accum);  // av matmul fence
        warpgroup::mma_AB(o_accum, att_block_mma, v_smem[tic]); // mm accumulate next attention chunk onto o
        warpgroup::mma_commit_group(); // dew it.

        if(tic) v_phase^=1;
    }

    auto *o_smem = reinterpret_cast<st_bf<qo_height, tile_width, layout_o>*>(&q_smem[0].data[0]); // reuse q memory for store
    warpgroup::store(o_smem[warpgroupid], o_accum); // store from registers to shared mem
    __syncthreads(); // everyone done?
    if (warpid%4 == 0) { // store o to HBM
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid; 
        tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); // dew it
    }

    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    __syncthreads();

    warpgroup::store(l_smem[warpgroupid], norm_vec);
    __syncthreads();
    if (warpid % 4 == 0) { // store l
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid; 
        tma::store_async(tma_l, (l_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    tma::store_async_wait(); // done it.
}

constexpr int WORKERS = 8;

constexpr int th = 4; 
constexpr int tw = 64/16;

using layout_nrow = ducks::st_layout::swizzle;

template<int N> __global__  __launch_bounds__(WORKERS*kittens::WARP_THREADS, 1)
void attend_ker_prep_train(CUtensorMap* tma_o, CUtensorMap* tma_d, CUtensorMap* tma_o_grad) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();

    st_bf<th, tw, layout_nrow>          (&og_smem)[WORKERS] = al.allocate<st_bf<th, tw, layout_nrow>, WORKERS>();
    st_bf<th, tw, layout_nrow>          (&o_smem) [WORKERS] = al.allocate<st_bf<th, tw, layout_nrow>, WORKERS>();
    st_bf<th, tw, layout_nrow>::col_vec (&d_smem) [WORKERS] = al.allocate<st_bf<th, tw, layout_nrow>::col_vec, WORKERS>();

    rt_fl<th, tw> og_reg;
    rt_fl<th, tw> o_reg; 
    rt_fl<th, tw>::col_vec d_reg;

    __shared__ uint64_t ograd_smem_barrier, o_smem_barrier;
    int o_phasebit = 0; 
    int og_phasebit = 0;

    if (threadIdx.x == 0) {
        tma::init_barrier<st_bf<th, tw, layout_o>, WORKERS>(ograd_smem_barrier, 1);
        tma::init_barrier<st_bf<th, tw, layout_o>, WORKERS>(o_smem_barrier, 1);
    }
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < WORKERS; w++) { // load o, o_grad
            int tile_idx = (blockIdx.y * WORKERS * gridDim.x) + (blockIdx.x * WORKERS) + w; 
            tma::load_async((o_smem[w]), tma_o, o_smem_barrier, tile_idx); 
            tma::load_async((og_smem[w]), tma_o_grad, ograd_smem_barrier, tile_idx); 
        }
    }

    tma::arrive_and_wait(ograd_smem_barrier, og_phasebit);
    tma::arrive_and_wait(o_smem_barrier, o_phasebit);

    load(o_reg, o_smem[warpid]);
    load(og_reg, og_smem[warpid]);

    mul(og_reg, og_reg, o_reg);
    row_sum(d_reg, og_reg);
    
    store(d_smem[warpid], d_reg);

    __syncthreads(); 
    if (warpid == 0) {
        for (int w = 0; w < WORKERS; w++) {
            int tile_idx = (blockIdx.y * WORKERS * gridDim.x) + (blockIdx.x * WORKERS) + w; 
            tma::store_async(tma_d, (d_smem[w]), tile_idx); 
        }
        tma::store_commit_group();
    }

    tma::store_async_wait();
}

template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void tile_reduce(ST (&dst)[N_TILES]) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;
    constexpr int RESPONSIBLE_ELEMENTS = (ST::num_elements+STRIDE-1) / STRIDE; // we know in advance this divides evenly.
    float acc[RESPONSIBLE_ELEMENTS];
    #pragma unroll
    for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
        int idx = kittens::laneid() + j*STRIDE;
        if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) acc[j] = __bfloat162float(dst[0].data[idx]); // start
    }
    // then propagate accumulation through
    for(int i = 1; i < N_TILES; i++) {
        #pragma unroll
        for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
            int idx = kittens::laneid() + j*STRIDE;
            if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) acc[j] += __bfloat162float(dst[i].data[idx]); // accumulate
        }
    }
    #pragma unroll
    for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
        int idx = kittens::laneid() + j*STRIDE;
        if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) dst[0].data[idx] = acc[j]; // set
    }
}

constexpr int WORKERS_BWD    = 2; 
constexpr int WORKERS_BWD_QO = 2; 

constexpr int tile_h    = 4;
constexpr int tile_h_qo = 4; 

static_assert(WORKERS_BWD >= WORKERS_BWD_QO, "WORKERS_BWD must be greater than or equal to WORKERS_BWD_QO");
static_assert(tile_h * WORKERS_BWD <= 8, "tile_h * WORKERS_BWD must be less than or equal to 8");
 
constexpr int tile_w = 64/16;

using layout_wgmma     = ducks::st_layout::wgmma_swizzle;
using layout_wgmma_itl = ducks::st_layout::wgmma_interleave;
using layout_tma_swi   = ducks::st_layout::swizzle; 

#define k_smem_tile  st_bf<tile_h, tile_w, layout_wgmma_itl>
#define v_smem_tile  st_bf<tile_h, tile_w, layout_wgmma_itl>

#define q_smem_tile  st_bf<tile_h_qo, tile_w, layout_wgmma_itl>
#define og_smem_tile st_bf<tile_h_qo, tile_w, layout_tma_swi>

#define qg_smem_tile st_bf<tile_h_qo, tile_w, layout_wgmma_itl>
#define kg_smem_tile k_smem_tile
#define vg_smem_tile v_smem_tile

#define l_smem_tile  st_bf<tile_h_qo, tile_w, layout_tma_swi>::col_vec
#define d_smem_tile  st_bf<tile_h_qo, tile_w, layout_tma_swi>::col_vec

template<int N> __global__ __launch_bounds__(WORKERS_BWD*kittens::WARP_THREADS, 1)
void attend_ker_bwd_train(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, 
                            CUtensorMap* tma_l_vec, CUtensorMap* tma_d_vec, 
                            CUtensorMap* tma_og, CUtensorMap* tma_qg, CUtensorMap* tma_kg, CUtensorMap* tma_vg, 
                            const bf16* __restrict__ __l__, const bf16* __restrict__ __d__)
{
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    const bf16 *_l  = __l__ + (blockIdx.y * N);
    const bf16 *_d  = __d__ + (blockIdx.y * N);


    k_smem_tile  (&k_smem) [WORKERS_BWD] = al.allocate<k_smem_tile, WORKERS_BWD>();
    v_smem_tile  (&v_smem) [WORKERS_BWD] = al.allocate<v_smem_tile, WORKERS_BWD>();

    q_smem_tile  (&q_smem)  [WORKERS_BWD_QO]                  = al.allocate<q_smem_tile,  WORKERS_BWD_QO>();
    og_smem_tile (&og_smem) [WORKERS_BWD_QO]                  = al.allocate<og_smem_tile, WORKERS_BWD_QO>();
    qg_smem_tile (&qg_smem) [WORKERS_BWD_QO][WORKERS_BWD + 1] = al.allocate<qg_smem_tile, WORKERS_BWD_QO, WORKERS_BWD + 1>();
    l_smem_tile (&l_smem)   [WORKERS_BWD_QO]                  = al.allocate<l_smem_tile,  WORKERS_BWD_QO>();
    d_smem_tile (&d_smem)   [WORKERS_BWD_QO]                  = al.allocate<d_smem_tile,  WORKERS_BWD_QO>();

    rt_bf<tile_h, tile_w> k_reg;  
    rt_bf<tile_h, tile_w, ducks::rt_layout::col> k_reg_col; 
    rt_bf<tile_h, tile_w> v_reg;
    
    rt_fl<tile_h, tile_w> kg_reg;
    rt_fl<tile_h, tile_w> vg_reg;

    rt_fl<tile_h_qo, tile_w> qg_reg;
    rt_bf<tile_h_qo, tile_w> q_reg;
    rt_bf<tile_h_qo, tile_w> do_reg;

    rt_fl<tile_h_qo, tile_h> att_block; 
    rt_bf<tile_h_qo, tile_h> att_block_mma;
    rt_fl<tile_h_qo, tile_h> temp_block; 
    rt_bf<tile_h_qo, tile_w>::col_vec l_reg_bf; 
    rt_bf<tile_h_qo, tile_w>::col_vec d_reg_bf;
    rt_fl<tile_h_qo, tile_w>::col_vec l_reg_fl; 
    rt_fl<tile_h_qo, tile_w>::col_vec d_reg_fl;

    int warpid = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    constexpr int qo_blocks = N / (q_smem[0].rows * WORKERS_BWD_QO);
    constexpr int kv_blocks = N / (k_smem[0].rows * WORKERS_BWD);

    __shared__ uint64_t kv_b, qo_b;

    int kv_phasebit = 0;
    int qo_phasebit = 0;

    if (threadIdx.x == 0) {
        tma::init_barrier<q_smem_tile,  WORKERS_BWD_QO * 3>(qo_b, 1); // q, og, qg
        tma::init_barrier<k_smem_tile , WORKERS_BWD    * 2>(kv_b, 1); // k, v
    }

    __syncthreads(); 

    for (int kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
        
        if (warpid == 0) {
            // load k and v
            for (int w = 0; w < WORKERS_BWD; w++) {
                int tile_idx = (blockIdx.y * WORKERS_BWD * kv_blocks) + (kv_idx * WORKERS_BWD) + w; 
                tma::load_async((k_smem[w]), tma_k, kv_b, tile_idx); 
                tma::load_async((v_smem[w]), tma_v, kv_b, tile_idx); 
            }
        }
        
        tma::arrive_and_wait(kv_b, kv_phasebit);
        kv_phasebit ^= 1;

        if (threadIdx.x == 0) {
            tma::set_bytes(kv_b, WORKERS_BWD * sizeof(bf16) * k_smem[0].num_elements * 2);
        }

        load(k_reg, k_smem[warpid]);
        load(v_reg, v_smem[warpid]);
        swap_layout(k_reg_col, k_reg);

        zero(kg_reg);
        zero(vg_reg);
        __syncthreads(); 

        for (int qo_idx = 0; qo_idx < qo_blocks; qo_idx++) {
            if (warpid == 0) {
                
                for (int w = 0; w < WORKERS_BWD_QO; w++) {
                    int tile_idx = (blockIdx.y * WORKERS_BWD_QO * qo_blocks) + (qo_idx * WORKERS_BWD_QO) + w;

                    tma::load_async((q_smem[w]),     tma_q,  qo_b, tile_idx); 
                    tma::load_async((og_smem[w]),    tma_og, qo_b, tile_idx); 
                    tma::load_async((qg_smem[w][0]), tma_qg, qo_b, tile_idx);
                }
            }

            if (warpid < WORKERS_BWD_QO) {
                load(l_smem[warpid], _l + (qo_idx * WORKERS_BWD_QO + warpid) * l_smem[0].length);
                load(d_smem[warpid], _d + (qo_idx * WORKERS_BWD_QO + warpid) * d_smem[0].length);
            }

            tma::arrive_and_wait(qo_b,  qo_phasebit);
            qo_phasebit ^= 1;

            if (threadIdx.x == 0) {
                tma::set_bytes(qo_b, WORKERS_BWD_QO * sizeof(bf16) * q_smem[0].num_elements * 3);
            }

            __syncthreads();
            if (threadIdx.x == 0 && blockIdx.y == 0 && qo_idx == 1 && kv_idx == 2) {
                // print out q 
                for (int w = 0; w < WORKERS_BWD_QO; w++) {
                    printf("q_smem[%d]\n", w); 
                    for (int r = 0; r < q_smem[w].rows; r++) {
                        for (int c = 0; c < q_smem[w].cols; c++) {
                            printf("%f ", __bfloat162float(q_smem[w].data[r * q_smem[w].cols + c]));
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
            }
            __syncthreads(); 
            
            mul(q_smem[warpid], q_smem[warpid], __float2bfloat16(0.125f));
            __syncthreads();

            for (int subtile = 0; subtile < WORKERS_BWD_QO; subtile++) {
                load(q_reg, q_smem[subtile]);
                
                zero(att_block);
                mma_ABt(att_block, q_reg, k_reg, att_block);

                load(l_reg_bf, l_smem[subtile]);
                copy(l_reg_fl, l_reg_bf);
                sub_row(att_block, att_block, l_reg_fl);
                exp(att_block, att_block);
                copy(temp_block, att_block);
                copy(att_block_mma, att_block);

                load(do_reg, og_smem[subtile]);
                rt_bf<tile_h_qo, tile_w, ducks::rt_layout::col> &do_reg_col = swap_layout_inplace(do_reg);
                rt_bf<tile_h_qo, tile_h, ducks::rt_layout::col> &att_block_mma_col = swap_layout_inplace(att_block_mma);

                mma_AtB(vg_reg, att_block_mma_col, do_reg_col, vg_reg);

                load(do_reg, og_smem[subtile]);
                zero(att_block);
                mma_ABt(att_block, do_reg, v_reg, att_block);

                load(d_reg_bf, d_smem[subtile]);
                copy(d_reg_fl, d_reg_bf);
                sub_row(att_block, att_block, d_reg_fl);
                mul(temp_block, temp_block, att_block);
                copy(att_block_mma, temp_block);

                zero(qg_reg);
                
                mma_AB(qg_reg, att_block_mma, k_reg_col, qg_reg);
                mul(qg_reg, qg_reg, __float2bfloat16(0.125f));
                store(qg_smem[subtile][1 + warpid], qg_reg);

                rt_bf<tile_h_qo, tile_h, ducks::rt_layout::col> &att_block_mma_col2 = swap_layout_inplace(att_block_mma);
                rt_bf<tile_h_qo, tile_w, ducks::rt_layout::col> &q_reg_col = swap_layout_inplace(q_reg);

                mma_AtB(kg_reg, att_block_mma_col2, q_reg_col, kg_reg);
            }

            __syncthreads();
            if (warpid < WORKERS_BWD_QO) {
                tile_reduce<1, qg_smem_tile, WORKERS_BWD + 1>(qg_smem[warpid]);
            }
            __syncthreads();

            if (warpid == 0) {
                for (int w = 0; w < WORKERS_BWD_QO; w++) {
                    int tile_idx = (blockIdx.y * WORKERS_BWD_QO * qo_blocks) + (qo_idx * WORKERS_BWD_QO) + w; 
                    tma::store_async(tma_qg, (qg_smem[w][0]), tile_idx);
                }
                tma::store_commit_group();
            }
            tma::store_async_wait();
        }

        store(v_smem[warpid], vg_reg);
        store(k_smem[warpid], kg_reg);
        __syncthreads();

        if (warpid == 0) {
            for (int w = 0; w < WORKERS_BWD; w++) {
                int tile_idx = (blockIdx.y * WORKERS_BWD * kv_blocks) + (kv_idx * WORKERS_BWD) + w; 
                tma::store_async(tma_vg, (v_smem[w]), tile_idx);
                tma::store_async(tma_kg, (k_smem[w]), tile_idx);
            }
            tma::store_commit_group();
        }
        tma::store_async_wait();
    }
}

#include "harness_h100_bwd.impl"