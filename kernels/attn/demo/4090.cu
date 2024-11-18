#include "kittens.cuh"

constexpr int ATTN_B = 16;
constexpr int ATTN_H = 16;
constexpr int ATTN_N = 1024;
constexpr int ATTN_D = 128;
constexpr int ITER   = 10;

using namespace kittens;

constexpr int NUM_WORKERS = 4;
constexpr int PIPE_STAGES = 3; 

template<int D> constexpr size_t ROWS = 16*(128/D); // height of each worker tile (rows)
template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, ROWS<D>, D, L>;
template<int D, typename T=float> using attn_tile = rt<T, ROWS<D>, ROWS<D>>;
template<int D> using shared_tile = st_bf<ROWS<D>, D>;
template<int D> using global_layout = gl<bf16, -1, -1, -1, D>; // B, H, g.Qg.rows specified at runtime, D=64 known at compile time for this kernel
template<int D> struct globals { global_layout<D> Qg, Kg, Vg, Og; };

template<int D> __launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__global__ void attend_ker(const __grid_constant__ globals<D> g) {
    
    using load_group = kittens::group<2>; // pairs of workers collaboratively load k, v tiles
    int loadid = load_group::groupid(), workerid = kittens::warpid(); // which worker am I?
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;
    const int batch = blockIdx.z, head  = blockIdx.y, q_seq = blockIdx.x * NUM_WORKERS + workerid;

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    
    shared_tile<D> (&k_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D> (&v_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    
    shared_tile<D> (&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);
    
    qkvo_tile<D, bf16> q_reg, k_reg;
    qkvo_tile<D, bf16, col_l> v_reg; 
    qkvo_tile<D, float> o_reg; 
    attn_tile<D, float> att_block;
    attn_tile<D, bf16> att_block_mma;

    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec;

    if (q_seq*ROWS<D> < g.Qg.rows) {
        load(qo_smem[workerid], g.Qg, {batch, head, q_seq, 0}); 
        __syncwarp();
        load(q_reg, qo_smem[workerid]);
    }
    __syncthreads();

    if constexpr(D == 64) mul(q_reg, q_reg, __float2bfloat16(0.125f * 1.44269504089));
    else if constexpr(D == 128) mul(q_reg, q_reg, __float2bfloat16(0.08838834764f * 1.44269504089));

    neg_infty(max_vec);
    zero(norm_vec);
    zero(o_reg);

    int kv_blocks = g.Qg.rows / (LOAD_BLOCKS*ROWS<D>), tic = 0;
    load_group::load_async(k_smem[loadid][0], g.Kg, {batch, head, loadid, 0});
    load_group::load_async(v_smem[loadid][0], g.Vg, {batch, head, loadid, 0});

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic=(tic+1)%PIPE_STAGES) {
        int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;
        
        if(next_load_idx*ROWS<D> < g.Kg.rows) {
            int next_tic = (tic+1)%PIPE_STAGES;
            load_group::load_async(k_smem[loadid][next_tic], g.Kg, {batch, head, next_load_idx, 0});
            load_group::load_async(v_smem[loadid][next_tic], g.Vg, {batch, head, next_load_idx, 0});
            load_async_wait<2>();
        }
        else load_async_wait();
        __syncthreads();

        #pragma unroll LOAD_BLOCKS
        for(int subtile = 0; subtile < LOAD_BLOCKS && (kv_idx*LOAD_BLOCKS + subtile) < g.Qg.rows; subtile++) {
            
            load(k_reg, k_smem[subtile][tic]);
            zero(att_block); 
            mma_ABt(att_block, q_reg, k_reg, att_block); 
            
            copy(max_vec_last,  max_vec);
            row_max(max_vec, att_block, max_vec); 
            sub_row(att_block, att_block, max_vec); 
            exp2(att_block, att_block);
            sub(max_vec_last, max_vec_last, max_vec); 
            exp2(max_vec_last, max_vec_last); 
            mul(norm_vec, norm_vec, max_vec_last); 
            row_sum(norm_vec, att_block, norm_vec); 
            copy(att_block_mma, att_block); 
            
            load(v_reg, v_smem[subtile][tic]); 
            mul_row(o_reg, o_reg, max_vec_last); 
            mma_AB(o_reg, att_block_mma, v_reg, o_reg);
        }
    }

    div_row(o_reg, o_reg, norm_vec);
    __syncthreads();

    if (q_seq*ROWS<D> < g.Qg.rows) {
        store(qo_smem[workerid], o_reg);
        __syncwarp();
        store(g.Og, qo_smem[workerid], {batch, head, q_seq, 0});
    }
}

#include "4090_harness.impl"