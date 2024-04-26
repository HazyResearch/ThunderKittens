

#define KITTENS_HOPPER // we are on an H100
#include "../../src/kittens.cuh"
#include <cooperative_groups.h>

constexpr int NUM_WORKERS = 16;
constexpr int NUM_WARPGROUPS = (NUM_WORKERS/(kittens::WARPGROUP_WARPS));

constexpr int qo_height = 4, kv_height = 4;
constexpr int NUM_WORKERS_KV = 4;
constexpr int tile_width = 64/16;

using namespace kittens;

using layout_q = ducks::st_layout::wgmma_swizzle; // need to make this 128b
using layout_k = ducks::st_layout::wgmma_swizzle; // need to make this 128b
using layout_v = ducks::st_layout::wgmma_interleave; // need to make this 128b
using layout_o = ducks::st_layout::swizzle; 

template<int N> __global__  __launch_bounds__(NUM_WORKERS*kittens::WARP_THREADS, 1)
void attend_ker_fwd_train(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o, CUtensorMap* tma_l) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    st_bf<qo_height, tile_width, layout_q>           (&q_smem)   [NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_q>,          NUM_WARPGROUPS>();
    st_bf<kv_height, tile_width, layout_k>           (&k_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, tile_width, layout_k>, 2,       NUM_WORKERS_KV>();
    st_bf<kv_height, tile_width, layout_v>           (&v_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, tile_width, layout_v>, 2,       NUM_WORKERS_KV>();
    st_bf<qo_height, tile_width, layout_q>::col_vec  (&l_smem)   [NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, tile_width, layout_q>::col_vec, NUM_WARPGROUPS>();

    int tic = 0, toc = 1;
 
    rt_fl<1, kv_height> att_block;
    rt_bf<1, kv_height> att_block_mma;
    rt_fl<1, tile_width> o_prev;
    rt_fl<1, kv_height>::col_vec max_vec_last, max_vec;
    rt_fl<1, kv_height>::col_vec norm_vec_last, norm_vec;

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS; 

    auto block = cooperative_groups::this_thread_block();

    constexpr int qo_tiles  = N / q_smem[0].rows; 
    constexpr int kv_blocks = N / (NUM_WORKERS_KV*k_smem[0][0].rows);

    __shared__ uint64_t qsmem_barrier, ksmem_barrier, vsmem_barrier;

    int q_phasebit = 0;
    int k_phasebit = 0;
    int v_phasebit = 0; 

    if (threadIdx.x == 0) {
        tma::init_barrier<st_bf<qo_height, tile_width, layout_q>, NUM_WARPGROUPS>(qsmem_barrier, 1);
        tma::init_barrier<st_bf<kv_height, tile_width, layout_k>, NUM_WORKERS_KV>(ksmem_barrier, 1); 
        tma::init_barrier<st_bf<kv_height, tile_width, layout_v>, NUM_WORKERS_KV>(vsmem_barrier, 1);
    }
    __syncthreads();

    if (warpid == 0) {
        for (int wg = 0; wg < NUM_WORKERS/kittens::WARPGROUP_WARPS; wg++) { // load q
            int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + wg;
            tma::load_async((q_smem[wg]), tma_q, qsmem_barrier, tile_idx); 
        }
        for (int w = 0; w < NUM_WORKERS_KV; w++) { // load k, v      
            int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + (0 * NUM_WORKERS_KV) + w; 
            tma::load_async((k_smem[tic][w]), tma_k, ksmem_barrier, tile_idx); 
            tma::load_async((v_smem[tic][w]), tma_v, vsmem_barrier, tile_idx); 
        }
    }

    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);

    tma::arrive_and_wait(qsmem_barrier, q_phasebit);
    q_phasebit ^= 1;
    __syncthreads();

    warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f));

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic ^= 1, toc ^= 1) {

        tma::arrive_and_wait(ksmem_barrier, k_phasebit);
        tma::arrive_and_wait(vsmem_barrier, v_phasebit);
        k_phasebit ^= 1;
        v_phasebit ^= 1;

        if ((threadIdx.x == 0)) {
            tma::set_bytes(ksmem_barrier, NUM_WORKERS_KV * sizeof(bf16) * k_smem[0][0].num_elements);
            tma::set_bytes(vsmem_barrier, NUM_WORKERS_KV * sizeof(bf16) * v_smem[0][0].num_elements);
        }
        __syncthreads();

        if ((kv_idx + 1 < kv_blocks) && (warpid == 0)) {
            for (int w = 0; w < NUM_WORKERS_KV; w++) {        
                int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS_KV) + w; 
                tma::load_async((k_smem[toc][w]), tma_k, ksmem_barrier, tile_idx); 
                tma::load_async((v_smem[toc][w]), tma_v, vsmem_barrier, tile_idx); 
            }
        }

        for(int subtile = 0; subtile < NUM_WORKERS_KV; subtile++) {
            warpgroup::mma_fence(att_block);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[tic][subtile]);
            warpgroup::mma_commit_group();

            copy(norm_vec_last, norm_vec);
            copy(max_vec_last,  max_vec);

            warpgroup::mma_async_wait();

            row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
            sub_row(att_block, att_block, max_vec);
            exp(att_block, att_block);

            sub(max_vec_last, max_vec_last, max_vec);
            exp(max_vec_last, max_vec_last);
            mul(norm_vec, norm_vec, max_vec_last);

            row_sum(norm_vec, att_block, norm_vec); // accumulate onto the norm_vec
            div_row(att_block, att_block, norm_vec);

            mul(norm_vec_last, norm_vec_last, max_vec_last);
            div(norm_vec_last, norm_vec_last, norm_vec);

            copy(att_block_mma, att_block); // convert to bf16 for mma
            mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma'ing onto it

            warpgroup::mma_fence(o_prev);
            warpgroup::mma_AB(o_prev, att_block_mma, v_smem[tic][subtile]);
            warpgroup::mma_commit_group();
        }
    }

    auto *o_smem = reinterpret_cast<st_bf<qo_height, tile_width, layout_o>*>(&q_smem[0].data[0]); // reuse q memory
    warpgroup::store(o_smem[warpgroupid], o_prev); 
    __syncthreads();
    if (warpid % 4 == 0) { // store o
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid; 
        tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    __syncthreads();

    warpgroup::store(l_smem[warpgroupid], norm_vec);
    __syncthreads();
    if (warpid % 4 == 0) { // store l
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * blockDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid; 
        tma::store_async(tma_l, (l_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    tma::store_async_wait();
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
            int tile_idx = (blockIdx.y * WORKERS * blockDim.x) + (blockIdx.x * WORKERS) + w; 
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
            int tile_idx = (blockIdx.y * WORKERS * blockDim.x) + (blockIdx.x * WORKERS) + w; 
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

constexpr int WORKERS_BWD = 8; 
constexpr int WORKERS_BWD_QO = 8; 

constexpr int tile_h = 1; // can go up to 8
constexpr int tile_h_qo = 1; 
constexpr int tile_w = 64/16;

using layout_swi = ducks::st_layout::swizzle; 

#define k_smem_tile  st_bf<tile_h, tile_w, layout_swi>
#define v_smem_tile  st_bf<tile_h, tile_w, layout_swi>

#define q_smem_tile  st_bf<tile_h, tile_w, layout_swi>
#define og_smem_tile st_bf<tile_h, tile_w, layout_swi>
#define qg_smem_tile st_bf<tile_h, tile_w, layout_swi>
#define l_smem_tile  st_bf<tile_h, tile_w, layout_swi>::col_vec
#define d_smem_tile  st_bf<tile_h, tile_w, layout_swi>::col_vec

#define scratch_pad  st_bf<tile_h, tile_h, layout_swi>

template<int N> __global__ __launch_bounds__(WORKERS_BWD*kittens::WARP_THREADS, 1)
void attend_ker_bwd_train(const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, 
                          const bf16* __restrict__ __l__, const bf16* __restrict__ __d__, 
                          const bf16* __restrict__ __og__, bf16* __qg__, bf16* __kg__, bf16* __vg__) 
{
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    
    // shared_allocator al((int*)&__shm[0]);
    shared_allocator al((int*)&__shm[0]);

    const bf16 *_q = __q__   + (blockIdx.y * N * 64); 
    const bf16 *_k = __k__   + (blockIdx.y * N * 64);
    const bf16 *_v = __v__   + (blockIdx.y * N * 64);
    const bf16 *_l = __l__   + (blockIdx.y * N);
    const bf16 *_d = __d__   + (blockIdx.y * N);
    const bf16 *_og = __og__ + (blockIdx.y * N * 64);

    bf16 *_qg = __qg__ + (blockIdx.y * N * 64);
    bf16 *_kg = __kg__ + (blockIdx.y * N * 64);
    bf16 *_vg = __vg__ + (blockIdx.y * N * 64);

    q_smem_tile  (&q_smem)  [WORKERS_BWD_QO]                 = al.allocate<q_smem_tile,  WORKERS_BWD_QO>();
    og_smem_tile (&og_smem) [WORKERS_BWD_QO]                 = al.allocate<og_smem_tile, WORKERS_BWD_QO>();
    qg_smem_tile (&qg_smem)[WORKERS_BWD_QO][WORKERS_BWD + 1] = al.allocate<qg_smem_tile, WORKERS_BWD_QO, WORKERS_BWD + 1>();
    l_smem_tile (&l_smem)  [WORKERS_BWD_QO]                  = al.allocate<l_smem_tile,  WORKERS_BWD_QO>();
    d_smem_tile (&d_smem)  [WORKERS_BWD_QO]                  = al.allocate<d_smem_tile,  WORKERS_BWD_QO>();

    rt_bf<tile_h, tile_w> k_reg; 
    rt_bf<tile_h, tile_w, ducks::rt_layout::col> k_reg_col;
    rt_bf<tile_h, tile_w> v_reg;
    rt_fl<tile_h, tile_w> kg_reg;
    rt_fl<tile_h, tile_w> vg_reg;

    rt_fl<tile_h_qo, tile_w> qg_reg;
    rt_bf<tile_h_qo, tile_w> q_reg;
    rt_bf<tile_h_qo, tile_w> do_reg;

    rt_bf<tile_h_qo, tile_w>::col_vec l_reg_bf; 
    rt_bf<tile_h_qo, tile_w>::col_vec d_reg_bf;
    rt_fl<tile_h_qo, tile_w>::col_vec l_reg_fl; 
    rt_fl<tile_h_qo, tile_w>::col_vec d_reg_fl; 

    rt_fl<tile_h_qo, tile_h> att_block; 
    rt_bf<tile_h_qo, tile_h> att_block_mma;
    rt_fl<tile_h_qo, tile_h> temp_block;
    
    int warpid = kittens::warpid();

    constexpr int qo_blocks = N / (q_smem[0].rows * WORKERS_BWD_QO);
    constexpr int kv_blocks = N / (v_reg.rows * WORKERS_BWD);

    for (int kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

        load(k_reg, _k + (kv_idx * WORKERS_BWD + warpid) * k_reg.num_elements, k_reg.cols);
        load(v_reg, _v + (kv_idx * WORKERS_BWD + warpid) * v_reg.num_elements, v_reg.cols);
        swap_layout(k_reg_col, k_reg);
        __syncthreads();

        zero(kg_reg);
        zero(vg_reg); 

        for (int qo_idx = 0; qo_idx < qo_blocks; qo_idx++) {

            if (warpid < WORKERS_BWD_QO) {
                load(q_smem[warpid], _q + (qo_idx * WORKERS_BWD_QO + warpid) * q_smem[0].num_elements, q_smem[0].cols);
                load(og_smem[warpid], _og + (qo_idx * WORKERS_BWD_QO + warpid) * og_smem[0].num_elements, og_smem[0].cols);
                load(qg_smem[warpid][0], _qg + (qo_idx * WORKERS_BWD_QO + warpid) * qg_smem[0][0].num_elements, qg_smem[0][0].cols);
                load(l_smem[warpid], _l + (qo_idx * WORKERS_BWD_QO + warpid) * l_smem[0].length);
                load(d_smem[warpid], _d + (qo_idx * WORKERS_BWD_QO + warpid) * d_smem[0].length);
            }
            __syncthreads();

            for (int subtile = 0; subtile < WORKERS_BWD_QO; subtile++) {
                load(q_reg, q_smem[subtile]);
                mul(q_reg, q_reg, __float2bfloat16(0.125f));
                
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
                store(_qg + (qo_idx * WORKERS_BWD_QO + warpid) * qg_reg.num_elements, qg_smem[warpid][0], qg_smem[0][0].cols);
            }
        }

        store(_vg + (kv_idx * WORKERS_BWD + warpid) * vg_reg.num_elements, vg_reg, vg_reg.cols); 
        store(_kg + (kv_idx * WORKERS_BWD + warpid) * kg_reg.num_elements, kg_reg, kg_reg.cols);
    }
}

#include "harness_naive_bwd.impl"