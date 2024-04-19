

#define KITTENS_HOPPER // we are on an H100
#include "../../src/kittens.cuh"
#include <cooperative_groups.h>

constexpr int NUM_WORKERS = 16;
constexpr int NUM_WARPGROUPS = (NUM_WORKERS/(kittens::WARPGROUP_WARPS));

constexpr int qo_height = 4, kv_height = 4;
constexpr int NUM_WORKERS_KV = 4;
constexpr int tile_width = 64/16;

using namespace kittens;

using layout_q = ducks::st_layout::wgmma_row_0b;
using layout_k = ducks::st_layout::wgmma_row_0b;
using layout_v = ducks::st_layout::wgmma_col_t_0b;
using layout_o = ducks::st_layout::naive; 

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
            warpgroup::dot_reset(att_block, q_smem[warpgroupid], k_smem[tic][subtile]);
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
            warpgroup::mma_accum(o_prev, att_block_mma, v_smem[tic][subtile]);
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

using layout_nrow = ducks::st_layout::naive;

template<int N> __global__  __launch_bounds__(WORKERS*kittens::WARP_THREADS, 1)
void attend_ker_prep_train(CUtensorMap* tma_o, CUtensorMap* tma_d, CUtensorMap* tma_o_grad) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_allocator al((int*)&__shm[0]);

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

constexpr int WORKERS_BWD = 4;

constexpr int tile_h = 4; 
constexpr int tile_w = 64/16;

template<int N> __global__  __launch_bounds__(WORKERS_BWD*kittens::WARP_THREADS, 1)
void attend_ker_bwd_train(CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, 
                            CUtensorMap* tma_l_vec, CUtensorMap* tma_d_vec, 
                            CUtensorMap* tma_og, CUtensorMap* tma_qg, CUtensorMap* tma_kg, CUtensorMap* tma_vg)
{
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_allocator al((int*)&__shm[0]);

    st_bf<tile_h, tile_w, layout_nrow> (&q_smem) [WORKERS_BWD] = al.allocate<st_bf<tile_h, tile_w, layout_nrow>, WORKERS_BWD>();
    st_bf<tile_h, tile_w, layout_nrow> (&k_smem) [WORKERS_BWD] = al.allocate<st_bf<tile_h, tile_w, layout_nrow>, WORKERS_BWD>();
    st_bf<tile_h, tile_w, layout_nrow> (&v_smem) [WORKERS_BWD] = al.allocate<st_bf<tile_h, tile_w, layout_nrow>, WORKERS_BWD>();
    st_bf<tile_h, tile_w, layout_nrow> (&og_smem)[WORKERS_BWD] = al.allocate<st_bf<tile_h, tile_w, layout_nrow>, WORKERS_BWD>();
    st_bf<tile_h, tile_w, layout_nrow> (&qg_smem)[WORKERS_BWD] = al.allocate<st_bf<tile_h, tile_w, layout_nrow>, WORKERS_BWD>();

    st_bf<tile_h, tile_w, layout_nrow>::col_vec (&l_smem)[WORKERS_BWD] = al.allocate<st_bf<tile_h, tile_w, layout_nrow>::col_vec, WORKERS_BWD>();
    st_bf<tile_h, tile_w, layout_nrow>::col_vec (&d_smem)[WORKERS_BWD] = al.allocate<st_bf<tile_h, tile_w, layout_nrow>::col_vec, WORKERS_BWD>();

    int warpid = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    auto block = cooperative_groups::this_thread_block();

    constexpr int qo_blocks = N / (q_smem[0].rows*WORKERS_BWD);
    constexpr int kv_blocks = N / (k_smem[0].rows*WORKERS_BWD);

    __shared__ uint64_t qsmem_barrier, ksmem_barrier, vsmem_barrier, lsmem_barrier, dsmem_barrier, ogsmem_barrier, qgsmem_barrier;

    int kv_phasebit = 0;
    int qo_phasebit = 0;

    if (threadIdx.x == 0) {
        tma::init_barrier<st_bf<tile_h, tile_w, layout_nrow>, WORKERS_BWD>(ksmem_barrier, 1);
        tma::init_barrier<st_bf<tile_h, tile_w, layout_nrow>, WORKERS_BWD>(vsmem_barrier, 1);

        tma::init_barrier<st_bf<tile_h, tile_w, layout_nrow>::col_vec, WORKERS_BWD>(lsmem_barrier, 1);
        tma::init_barrier<st_bf<tile_h, tile_w, layout_nrow>::col_vec, WORKERS_BWD>(dsmem_barrier, 1);

        tma::init_barrier<st_bf<tile_h, tile_w, layout_nrow>, WORKERS_BWD>(qsmem_barrier,  1);
        tma::init_barrier<st_bf<tile_h, tile_w, layout_nrow>, WORKERS_BWD>(ogsmem_barrier, 1);
        tma::init_barrier<st_bf<tile_h, tile_w, layout_nrow>, WORKERS_BWD>(qgsmem_barrier, 1);
    }
    __syncthreads();

    for (int kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
        
        if (warpid == 0) {
            // load k and v
            for (int w = 0; w < WORKERS_BWD; w++) {
                int tile_idx = (blockIdx.y * WORKERS_BWD * kv_blocks) + (kv_idx * WORKERS_BWD) + w; 
                tma::load_async((k_smem[w]), tma_k, ksmem_barrier, tile_idx); 
                tma::load_async((v_smem[w]), tma_v, vsmem_barrier, tile_idx); 
            }
        }

        tma::arrive_and_wait(ksmem_barrier, kv_phasebit);
        tma::arrive_and_wait(vsmem_barrier, kv_phasebit);
        kv_phasebit ^= 1;

        if (threadIdx.x == 0) {
            tma::set_bytes(ksmem_barrier, WORKERS_BWD * sizeof(bf16) * k_smem[0].num_elements);
            tma::set_bytes(vsmem_barrier, WORKERS_BWD * sizeof(bf16) * v_smem[0].num_elements);
        }

        rt_bf<tile_h, tile_w> k_reg; 
        rt_bf<tile_h, tile_w, ducks::rt_layout::col> k_reg_col; 
        rt_bf<tile_h, tile_w> v_reg;

        rt_fl<tile_h, tile_w> kg_reg;
        rt_fl<tile_h, tile_w> vg_reg;

        load(k_reg, k_smem[warpid]);
        load(v_reg, v_smem[warpid]);

        swap_layout(k_reg_col, k_reg);
        mul(k_reg_col, k_reg_col, __float2bfloat16(0.125f)); // temperature adjustment

        zero(kg_reg);
        zero(vg_reg);

        for (int qo_idx = 0; qo_idx < qo_blocks; qo_idx++) {

            if (warpid == 0) {
                for (int w = 0; w < WORKERS_BWD; w++) {
                    int tile_idx = (blockIdx.y * WORKERS_BWD * qo_blocks) + (qo_idx * WORKERS_BWD) + w; 
                    tma::load_async((q_smem[w]),  tma_q,     qsmem_barrier,  tile_idx); 
                    tma::load_async((og_smem[w]), tma_og,    ogsmem_barrier, tile_idx); 
                    tma::load_async((qg_smem[w]), tma_qg,    qgsmem_barrier, tile_idx);
                    tma::load_async((l_smem[w]),  tma_l_vec, lsmem_barrier,  tile_idx); 
                    tma::load_async((d_smem[w]),  tma_d_vec, dsmem_barrier,  tile_idx); 
                }
            }

            tma::arrive_and_wait(qsmem_barrier,  qo_phasebit);
            tma::arrive_and_wait(ogsmem_barrier, qo_phasebit);
            tma::arrive_and_wait(qgsmem_barrier, qo_phasebit);
            tma::arrive_and_wait(lsmem_barrier,  qo_phasebit);
            tma::arrive_and_wait(dsmem_barrier,  qo_phasebit);
            qo_phasebit ^= 1;

            if (threadIdx.x == 0) {
                tma::set_bytes(qsmem_barrier,  WORKERS_BWD * sizeof(bf16) * q_smem[0].num_elements);
                tma::set_bytes(ogsmem_barrier, WORKERS_BWD * sizeof(bf16) * og_smem[0].num_elements);
                tma::set_bytes(qgsmem_barrier, WORKERS_BWD * sizeof(bf16) * qg_smem[0].num_elements);
                tma::set_bytes(lsmem_barrier,  WORKERS_BWD * sizeof(bf16) * l_smem[0].length);
                tma::set_bytes(dsmem_barrier,  WORKERS_BWD * sizeof(bf16) * d_smem[0].length);
            }

            rt_bf<tile_h, tile_w> q_reg; 
            rt_bf<tile_h, tile_w, ducks::rt_layout::col> q_reg_col; 
            rt_bf<tile_h, tile_w> do_reg; 

            rt_fl<tile_h, tile_w> qg_reg_out;
            rt_fl<tile_h, tile_w> qg_reg_fl; 
            rt_bf<tile_h, tile_w> qg_reg;

            rt_bf<tile_h, tile_h>::col_vec l_reg;  
            rt_bf<tile_h, tile_h>::col_vec d_reg;
            rt_fl<tile_h, tile_h>::col_vec sub_reg;

            rt_fl<tile_h, tile_h> att_block; 
            rt_bf<tile_h, tile_h> att_block_mma;

            rt_fl<tile_h, tile_h> dP; 
            rt_bf<tile_h, tile_h> dP_mma;

            load(qg_reg_out, qg_smem[warpid]);
            copy(qg_reg, qg_reg_out);

            for (int subtile = 0; subtile < WORKERS_BWD; subtile++) {
                load(q_reg, q_smem[subtile]);
                mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment
                
                zero(att_block);
                dot(att_block, q_reg, k_reg, att_block);


                load(l_reg, l_smem[subtile]);
                copy(sub_reg, l_reg);
                sub_row(att_block, att_block, sub_reg);
                exp(att_block, att_block);
                copy(dP, att_block); 


                load(do_reg, og_smem[subtile]);
                rt_bf<tile_h, tile_w, ducks::rt_layout::col> &do_reg_col = swap_layout_inplace(do_reg);
                
                copy(att_block_mma, att_block);
                transpose_inplace(att_block_mma);
                
                mma(vg_reg, att_block_mma, do_reg_col, vg_reg);

                
                load(do_reg, og_smem[subtile]);
                zero(att_block);
                dot(att_block, do_reg, v_reg, att_block);


                load(d_reg, d_smem[subtile]);
                copy(sub_reg, d_reg);
                sub_row(att_block, att_block, sub_reg);
                mul(dP, dP, att_block);
                copy(dP_mma, dP);

                load(qg_reg, qg_smem[subtile]);
                copy(qg_reg_fl, qg_reg);
                mma(qg_reg_fl, dP_mma, k_reg_col, qg_reg_fl);
                store(qg_smem[subtile], qg_reg_fl);
                
                transpose_inplace(dP_mma);
                swap_layout(q_reg_col, q_reg);
                mma(kg_reg, dP_mma, q_reg_col, kg_reg);
            }

            __syncthreads();
            store(qg_smem[warpid], qg_reg_fl);
            __syncthreads();
            if (warpid == 0) {
                for (int w = 0; w < WORKERS_BWD; w++) {
                    int tile_idx = (blockIdx.y * WORKERS_BWD * qo_blocks) + (qo_idx * WORKERS_BWD) + w; 
                    tma::store_async(tma_qg, (qg_smem[w]), tile_idx);
                }
                tma::store_commit_group();
            }
            tma::store_async_wait();
        }

        store(v_smem[warpid], vg_reg);
        store(k_smem[warpid], kg_reg);
        __syncthreads();

        // store vg
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

#include "harness_t.impl"
