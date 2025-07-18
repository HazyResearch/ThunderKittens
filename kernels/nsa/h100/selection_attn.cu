// # Define TORCH_COMPILE macro

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>
#include "cuda_bf16.h"

using namespace kittens;
namespace cg = cooperative_groups;

namespace NSA_SELECTION_ATTN{

static inline int host_ceil_div(int a, int b){
    return (a+b-1)/b;
}

__device__ static inline int ceil_div(int a, int b){
    return (a+b-1)/b;
}


// SRC [16 (Block of Width), qo_height] to DST [qo_height, width]
template<ducks::st::all ST, ducks::rt::all RT>
__device__ static inline void warpgroup_store_t(ST &dst, const RT &src, int width_offset){
    using T2 = RT::dtype;
    using U  = ST::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int local_warpid = warpid()%4;
    int warp_laneid = ::kittens::laneid();
    int row = warp_laneid%16;
    int col = warp_laneid/16*8+local_warpid*src.tile_size_row+width_offset;
    U2 tmp[4];
    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][0].data[0]);
    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][0].data[1]);
    tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[0][0].data[2]);
    tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[0][0].data[3]);
    move<U2>::stsm4t(dst.idx(shared_addr, {row, col}), tmp[0], tmp[2], tmp[1], tmp[3]);
}

// [BLOCKKV, WIDTH] to [WIDTH/2, BLOCKKV] and [WIDTH/2, BLOCKKV]
template<ducks::st::all ST, ducks::rt::all RT>
__device__ static inline void warpgroup_load_t_split(RT &dst0, RT &dst1, const ST &src){
    int local_warpid = warpid()%4;
    using T2 = RT::dtype;
    using U  = ST::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = ::kittens::laneid();
    uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    int row =  (warp_laneid % 16);
    int col = (warp_laneid / 16) * 8 + local_warpid*16;

    U2 tmp[4];
    #pragma unroll
    for(int j=0; j<4; j++, row+=16){
        
        move<U2>::ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3], src.idx(src_addr, {row, col}));
        dst0.tiles[0][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
        dst0.tiles[0][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
        dst0.tiles[0][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
        dst0.tiles[0][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
        move<U2>::ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3], src.idx(src_addr, {row, col+64}));
        dst1.tiles[0][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
        dst1.tiles[0][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
        dst1.tiles[0][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
        dst1.tiles[0][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
    }
}


template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int qo_height  = (16);
    constexpr static int kv_height  = (16);
    constexpr static int stages     = (2);
};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (16);
    constexpr static int kv_height  = (16);
    constexpr static int stages     = (2);
};

template<int D> struct fwd_globals {
    using qo_tile   =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using k_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using v_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>>;
    using indices_vec =       sv<int, 16>;

    using q_gl = gl<bf16,  -1, -1, -1, -1, tma::descriptor<qo_tile, dim::ROW>>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, tma::descriptor<k_tile, dim::DEPTH>>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, tma::descriptor<v_tile, dim::DEPTH>>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, tma::descriptor<qo_tile, dim::ROW>>;
    using indices_gl = gl<int, -1, -1, -1, -1>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;
    indices_gl indices;

    const int N;
    const int KV_N;
    const int hr;
    const int block_size;
    const int block_count;
};

template<int D, bool is_causal>
__global__  __launch_bounds__(12*32, 12)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid();
    using K = fwd_attend_ker_tile_dims<D>;
    using qo_tile     =         fwd_globals<D>::qo_tile;
    using k_tile      =         fwd_globals<D>::k_tile;
    using v_tile      =         fwd_globals<D>::v_tile;
    using l_col_vec   =         fwd_globals<D>::l_col_vec;
    using indices_vec =         fwd_globals<D>::indices_vec;
    

    k_tile    (&k_smem)[K::stages]        = al.allocate<k_tile, K::stages>();
    v_tile    (&v_smem)[K::stages]        = al.allocate<v_tile, K::stages>();
    l_col_vec (*l_smem)        = reinterpret_cast<l_col_vec(*)>(&v_smem[0].data[0]);
    //indices_vec (&indices_smem)  = al.allocate<indices_vec>();
    qo_tile   (*qo_smem)       = reinterpret_cast<qo_tile(*)>(&k_smem[0].data[0]);
    int kv_blocks   = g.KV_N / (K::kv_height);

    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages];
    if(threadIdx.x == 0) {        
        init_semaphore(qsmem_semaphore, 0, 1);
        for(int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1);
            init_semaphore(v_smem_arrived[j], 0, 1);
        }
    }
    __syncthreads();

    rt_bf<K::qo_height, K::tile_width> q_reg;
    rt_bf<K::kv_height, K::tile_width> k_reg;
    rt_bf<K::kv_height, K::tile_width, col_l> v_reg;
    rt_fl<K::qo_height, K::kv_height>  att_block;
    rt_bf<K::qo_height, K::kv_height>  att_block_mma;
    rt_fl<K::qo_height, K::tile_width> o_reg;
    // [1, qo_height]
    col_vec<rt_fl<K::qo_height, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

    tma::expect_bytes(qsmem_semaphore, sizeof(qo_tile));
    coord<qo_tile> q_tile_idx = {blockIdx.z, blockIdx.x, blockIdx.y, 0};
    tma::load_async<dim::ROW, cache_policy::NORMAL>(qo_smem[0], g.q, q_tile_idx, qsmem_semaphore);

    wait(qsmem_semaphore, 0);
    load(q_reg, qo_smem[0]);

    neg_infty(max_vec);
    zero(norm_vec);
    zero(o_reg);
    zero(att_block);
    int kv_iters = g.block_count;
    int num_chunk = g.block_size / K::kv_height;
    
    for (auto kv_idx = 0; kv_idx < kv_iters; kv_idx++) {
        int kv_chunk_idx = g.indices[{blockIdx.z, blockIdx.x, blockIdx.y, kv_idx}];
        if(blockIdx.x<kv_chunk_idx*g.block_size) {
            continue;
        }

        for (int j = 0; j < K::stages - 1; j++) {
            coord<k_tile> kv_tile_idx = {blockIdx.z, kv_chunk_idx*num_chunk+j, blockIdx.y, 0};
            tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(k_smem[j], g.k, kv_tile_idx, k_smem_arrived[j]);
            tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(v_smem[j], g.v, kv_tile_idx, v_smem_arrived[j]);
        }

        for(int inner_idx = 0; inner_idx < num_chunk; inner_idx++) {
            wait(k_smem_arrived[(inner_idx)%K::stages], (inner_idx/K::stages)%2);
            load(k_reg, k_smem[(inner_idx)%K::stages]);
            mma<transpose::N, transpose::T>(att_block, q_reg, k_reg, att_block);
            copy(max_vec_last_scaled, max_vec);
            if constexpr (D == 64) { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
            else                   { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }

            if constexpr (is_causal) {
                int  kv_row_idx = (kv_chunk_idx*num_chunk + inner_idx) * K::kv_height;
                if (blockIdx.x < kv_row_idx+K::kv_height){
                    int kv_row_idx_lower = kv_row_idx + (laneid() % 4)*2;
                    if(blockIdx.x < kv_row_idx_lower){
                        att_block.tiles[0][0].data[0].x = base_types::constants<float>::neg_infty();
                        att_block.tiles[0][0].data[1].x = base_types::constants<float>::neg_infty();
                    }
                    if(blockIdx.x < kv_row_idx_lower+1){
                        att_block.tiles[0][0].data[0].y = base_types::constants<float>::neg_infty();
                        att_block.tiles[0][0].data[1].y = base_types::constants<float>::neg_infty();
                    }
                    if(blockIdx.x < kv_row_idx_lower+8){
                        att_block.tiles[0][0].data[2].x = base_types::constants<float>::neg_infty();
                        att_block.tiles[0][0].data[3].x = base_types::constants<float>::neg_infty();
                    }
                    if(blockIdx.x < kv_row_idx_lower+9){
                        att_block.tiles[0][0].data[2].y = base_types::constants<float>::neg_infty();
                        att_block.tiles[0][0].data[3].y = base_types::constants<float>::neg_infty();
                    }
                    __syncwarp();
                }
            }

            row_max(max_vec, att_block, max_vec);
                
            if constexpr (D == 64) { 
                mul(att_block, att_block,    1.44269504089f*0.125f); 
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
            }
            else                   { 
                mul(att_block, att_block,    1.44269504089f*0.08838834764f); 
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
            }

            sub_row(att_block, att_block, max_vec_scaled);
            exp2(att_block, att_block);
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last_scaled,       max_vec_last_scaled);
            mul(norm_vec,            norm_vec,     max_vec_last_scaled);
            row_sum(norm_vec,  att_block, norm_vec);
            add(att_block, att_block, 0.f);
            copy(att_block_mma, att_block); 
            mul_row(o_reg, o_reg, max_vec_last_scaled); 


            wait(v_smem_arrived[(inner_idx)%K::stages], (inner_idx/K::stages)%2);
            load(v_reg, v_smem[(inner_idx)%K::stages]);
            if(inner_idx+1 < num_chunk && inner_idx+2 >= K::stages) {
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_chunk_idx*num_chunk + inner_idx + 1, blockIdx.y, 0};
                tma::expect_bytes(k_smem_arrived[(inner_idx+1)%K::stages], sizeof(k_tile));
                tma::load_async<dim::DEPTH, cache_policy::NORMAL>(k_smem[(inner_idx+1)%K::stages], g.k, kv_tile_idx, k_smem_arrived[(inner_idx+1)%K::stages]);
                tma::expect_bytes(v_smem_arrived[(inner_idx+1)%K::stages], sizeof(v_tile));
                tma::load_async<dim::DEPTH, cache_policy::NORMAL>(v_smem[(inner_idx+1)%K::stages], g.v, kv_tile_idx, v_smem_arrived[(inner_idx+1)%K::stages]);
            }
            mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_reg, o_reg);
        }
    }

    div_row(o_reg, o_reg, norm_vec);
    store(qo_smem[0], o_reg);
    __syncwarp();
    coord<qo_tile> o_tile_idx = {blockIdx.z, blockIdx.x, blockIdx.y, 0};
    tma::store_async<dim::ROW, cache_policy::NORMAL>(g.o, qo_smem[0], o_tile_idx);

    mul(max_vec_scaled,   max_vec_scaled, 0.69314718056f);
    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec_scaled);


    if constexpr (D == 64) { mul(norm_vec, norm_vec, -8.0f); }
    else                   { mul(norm_vec, norm_vec, -11.313708499f); }

    store(l_smem[0], norm_vec);
    __syncwarp();
    coord<l_col_vec> tile_idx = {blockIdx.z, blockIdx.x, 0, blockIdx.y};
    tma::store_async(g.l, l_smem[0], tile_idx);
    tma::store_async_wait();

}

// ---------------------------------------------------------------------------------------------------
// ----------------------------------- Backward preparation kernel -----------------------------------
// ---------------------------------------------------------------------------------------------------

template<int D>
struct bwd_prep_globals {
    using og_tile = st_bf<16, D>;
    using o_tile  = st_bf<16, D>;
    using d_tile  = col_vec<st_fl<16, D>>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, tma::descriptor<og_tile, dim::ROW>>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1, tma::descriptor<o_tile, dim::ROW>>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;
    constexpr static int seq_block_size = 4;

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

    using og_tile = st_bf<16, D>;
    using o_tile  = st_bf<16, D>;
    using d_tile  = col_vec<st_fl<16, D>>;

    og_tile (&og_smem)[g.seq_block_size] = al.allocate<og_tile, g.seq_block_size>();
    o_tile  (&o_smem) [g.seq_block_size] = al.allocate<o_tile , g.seq_block_size>();
    d_tile  (&d_smem) [g.seq_block_size] = al.allocate<d_tile, g.seq_block_size>();


    rt_fl<16, D> og_reg, o_reg;
    col_vec<rt_fl<16, D>> d_reg;

    __shared__ kittens::semaphore smem_semaphore;

    if (threadIdx.x == 0) {
        init_semaphore(smem_semaphore, 0, 1);
        tma::expect_bytes(smem_semaphore, sizeof(og_smem[0]) * g.seq_block_size * 2);
    }
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < g.seq_block_size; w++) {
            coord<o_tile> tile_idx = {blockIdx.z, (blockIdx.x * g.seq_block_size) + w, blockIdx.y, 0};
            tma::load_async<dim::ROW, cache_policy::NORMAL>(o_smem[w],  g.o,  tile_idx, smem_semaphore);
            tma::load_async<dim::ROW, cache_policy::NORMAL>(og_smem[w], g.og, tile_idx, smem_semaphore);
        }
    }

    wait(smem_semaphore, 0);
    load(o_reg, o_smem[warpid]);
    load(og_reg, og_smem[warpid]);
    mul(og_reg, og_reg, o_reg);
    row_sum(d_reg, og_reg);
    store(d_smem[warpid], d_reg);
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < g.seq_block_size; w++) {
            coord<d_tile> tile_idx = {blockIdx.z, (blockIdx.x * g.seq_block_size) + w, 0, blockIdx.y};
            tma::store_async(g.d, d_smem[w], tile_idx);
        }
    }
    tma::store_async_wait();
}

struct prepare_mask_globals {
    using indices_gl = gl<int,  -1, -1, -1, -1>;
    constexpr static int seq_blocks = 128;
    constexpr static int threads = 128;
    indices_gl indices;
    bool*    mask;
    int stride_b;
    int stride_h;
    int block_size;
};

__global__ void prepare_mask(const __grid_constant__ prepare_mask_globals g)
{
 
    const int inner_id = threadIdx.x % g.indices.cols();
    const int seq_id = (blockIdx.z * g.threads + threadIdx.x) / g.indices.cols();
    const int seq_offset = g.seq_blocks * g.threads / g.indices.cols();
    int*  indices_ptr = g.indices.raw_ptr+blockIdx.x*g.indices.template stride<0>() + blockIdx.y*g.indices.template stride<2>() + inner_id;
    bool* mask_ptr = g.mask + blockIdx.x * g.stride_b + blockIdx.y * g.stride_h;

    for(int i=seq_id; i<g.indices.depth(); i+=seq_offset){
        int index = *(indices_ptr+i*g.indices.template stride<1>());
        mask_ptr[index*g.indices.depth()+i] = i >= index*g.block_size;
    }

}

template<int D> struct bwd_attend_ker_tile_dims {};
template<> struct bwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int tile_h     = (4*16);
    constexpr static int tile_h_qo  = (16);
    constexpr static int blocks_sm = 1;
    constexpr static int num_k_chunk = 1;
};
template<> struct bwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int tile_h     = (4*16);
    constexpr static int tile_h_qo  = (16);
    constexpr static int blocks_sm = 1;
    constexpr static int num_k_chunk = 2;
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
    using qg_tile =         st_bf<G::tile_h_qo, G::tile_width>;
    using kg_tile =         st_bf<G::tile_h,    G::tile_width>;
    using vg_tile =         st_bf<G::tile_h,    G::tile_width>;
    using attn_tile =       st_bf<G::tile_h, G::tile_h_qo>; 
    using l_tile  = col_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile  = col_vec<st_fl<G::tile_h_qo, G::tile_h>>;

    using q_gl  = gl<bf16,  -1, -1, -1, -1, tma::descriptor<q_tile, dim::ROW>>;
    using k_gl  = gl<bf16,  -1, -1, -1, -1, tma::descriptor<k_tile, dim::DEPTH>>;
    using v_gl  = gl<bf16,  -1, -1, -1, -1, tma::descriptor<v_tile, dim::DEPTH>>;
    using og_gl = gl<bf16,  -1, -1, -1, -1, tma::descriptor<og_tile, dim::ROW>>;
    using qg_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<qg_tile, dim::ROW>>;
    using kg_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<kg_tile, dim::DEPTH>>;
    using vg_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<vg_tile, dim::DEPTH>>;
    using indices_gl = gl<int, -1, -1, -1, -1>;
    using mask_gl    = gl<bool, -1, -1, -1, -1>;

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
    bool* mask;

    const int N;
    const int hr;
    const int block_size;
    const int block_count;
    const int mask_stride_b;
    const int mask_stride_h;
};

__device__ static inline void
stream_tile(auto &reg_tile, auto &smem_vec, int tic) {
    int base_col = 2*(kittens::laneid()%4);
    reg_tile.tiles[0][0].data[0] = *(float2*)&smem_vec[tic][base_col + 0];
    reg_tile.tiles[0][0].data[1] = *(float2*)&smem_vec[tic][base_col + 0];
    reg_tile.tiles[0][0].data[2] = *(float2*)&smem_vec[tic][base_col + 8];
    reg_tile.tiles[0][0].data[3] = *(float2*)&smem_vec[tic][base_col + 8];
}

__device__ static inline void
stream_sub_tile(auto &reg_tile, auto &smem_vec, int tic) {
    int base_col = 2*(laneid()%4);
    reg_tile.tiles[0][0].data[0] = base_ops::sub::template op<float2>(reg_tile.tiles[0][0].data[0], *(float2*)&smem_vec[tic][base_col + 0]);
    reg_tile.tiles[0][0].data[1] = base_ops::sub::template op<float2>(reg_tile.tiles[0][0].data[1], *(float2*)&smem_vec[tic][base_col + 0]);
    reg_tile.tiles[0][0].data[2] = base_ops::sub::template op<float2>(reg_tile.tiles[0][0].data[2], *(float2*)&smem_vec[tic][base_col + 8]);
    reg_tile.tiles[0][0].data[3] = base_ops::sub::template op<float2>(reg_tile.tiles[0][0].data[3], *(float2*)&smem_vec[tic][base_col + 8]);
}

template<int tile_h_qo, int tile_h>
__device__ static inline void 
causal_mask(auto &reg_tile, int qo_idx) {
    int k_blk = blockIdx.x * BWD_CONSUMER_WARPGROUPS * tile_h + kittens::warpid()*kittens::TILE_ROW_DIM<bf16>;

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(reg_tile.tiles[0][j]);
        if      (qo_idx  < k_blk+j*kittens::TILE_ROW_DIM<bf16>) { neg_infty(attn_subtile); }
        else if (qo_idx < k_blk+(j+1)*kittens::TILE_ROW_DIM<bf16>-1) {
            int k_idx_lower = k_blk + laneid() / 4;
            int k_idx_upper = k_idx_lower + 8;
            if(k_idx_lower > qo_idx) { 
                attn_subtile.tiles[0][0].data[0] = base_types::constants<float2>::neg_infty(); 
                attn_subtile.tiles[0][0].data[2] = base_types::constants<float2>::neg_infty(); 
            }
            if(k_idx_upper > qo_idx) { 
                attn_subtile.tiles[0][1].data[1] = base_types::constants<float2>::neg_infty(); 
                attn_subtile.tiles[0][0].data[3] = base_types::constants<float2>::neg_infty(); 
            }
        }
    }
}

template<bool is_causal, int tile_h_qo, int tile_h, int tile_width, int D>
__device__ static inline void
compute_bwd_loop(
        kittens::semaphore *vec_b, kittens::semaphore *q_b, kittens::semaphore *o_b, 
        rt_fl<16, tile_h_qo> &s_block_t, rt_fl<16, tile_h_qo> &dp_block_t, 
        rt_fl<16, tile_h_qo> &p_block_t, rt_fl<16, tile_h_qo> &ds_block_t,  
        rt_bf<16, tile_h_qo> &p_block_t_mma,  rt_bf<16, tile_h_qo> &ds_block_t_mma,
        rt_fl<16, tile_width> &kg_reg, rt_fl<16, tile_width> &vg_reg,
        auto &q_smem, auto &k_smem, auto &v_smem, 
        auto &og_smem, auto &ds_smem, auto &l_smem, auto &d_smem,
        int qo_idx, int q_start, int tic, int toc) 
{
    group<8>::sync(11); 
    wait(q_b[tic], ((qo_idx - q_start)/2)%2);
    // wait(vec_b[tic], ((qo_idx - q_start)/2)%2);
    stream_tile(s_block_t, l_smem, tic);
    
    // [BlockKV, Width] * [BlockQ, Width] -> [BlockKV, BlockQ]
    // [BlockKV, Width] * [G, Width] -> [BlockKV, G]
    warpgroup::mma_ABt(s_block_t, k_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], q_smem[tic]);
    warpgroup::mma_commit_group();

    wait(o_b[tic], ((qo_idx - q_start)/2)%2);
    // [BlockKV, Width] * [BlockQ, Width] -> [BlockKV, BlockQ]
    // [BlockKV, Width] * [G, Width] -> [BlockKV, G]
    warpgroup::mm_ABt(dp_block_t, v_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], og_smem[tic]);
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

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

    // [BlockKV, G] * [G, Width] -> [BlockKV, Width]
    warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[tic]);
    warpgroup::mma_commit_group();
    
    copy(ds_block_t_mma, ds_block_t);
    warpgroup::store(ds_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], ds_block_t_mma);

    // [BlockKV, G] * [G, Width] -> [BlockKV, Width]
    warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[tic]);
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();
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
        coord<kg_tile> tile_idx = {blockIdx.z, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + (kittens::warpid()/kittens::WARPGROUP_WARPS), kv_head_idx, 0};
        tma::store_async<dim::DEPTH, cache_policy::NORMAL>(dst.kg, kg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], tile_idx);
        tma::store_commit_group();
    }

    wait(bar, toc);
    warpgroup::store(vg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], vg_reg);
    group<4>::sync(warpgroup::groupid()+4);

    if (kittens::warpid() % 4 == 0) {
        coord<vg_tile> tile_idx = {blockIdx.z, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + (kittens::warpid()/kittens::WARPGROUP_WARPS), kv_head_idx, 0};
        tma::store_async<dim::DEPTH, cache_policy::NORMAL>(dst.vg, vg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], tile_idx);
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
    
    using kg_tile   = bwd_globals<D>::kg_tile;
    using vg_tile   = bwd_globals<D>::vg_tile;
    using k_tile    = bwd_globals<D>::k_tile;
    using v_tile    = bwd_globals<D>::v_tile;
    using q_tile    = bwd_globals<D>::q_tile;
    using og_tile   = bwd_globals<D>::og_tile;
    using qg_tile   = bwd_globals<D>::qg_tile;
    using l_tile    = bwd_globals<D>::l_tile;
    using d_tile    = bwd_globals<D>::d_tile;
    using attn_tile = bwd_globals<D>::attn_tile; 

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
    const int warpid      = kittens::warpid();
    const int warpgroupid = warpid/kittens::WARPGROUP_WARPS;
    const int qo_blocks   = N;
    const int kv_head_idx = blockIdx.y; 

    __shared__ kittens::semaphore kv_b, q_b[2], o_b[2], vec_b[2];
    __shared__ kittens::semaphore compute_done[2], qg_ready; 

    int tic = 0, toc = 1;
    const int q_start = (is_causal) ? (blockIdx.x*G::tile_h*BWD_CONSUMER_WARPGROUPS) : (0);
    // bool* mask_ptr = g.mask + blockIdx.z*g.mask_stride_b + blockIdx.y*g.mask_stride_h + (blockIdx.x+;

    if (threadIdx.x == 0) {
        init_semaphore(kv_b,  0, 1);
        init_semaphore(qg_ready, 1, 0);
        for (int s = 0; s < 2; s++) {
            init_semaphore(q_b[s],   0, 1);
            init_semaphore(o_b[s],   0, 1); 
            init_semaphore(vec_b[s], 0, 1);
            init_semaphore(compute_done[s], 1, 0);
        }
        
        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * BWD_CONSUMER_WARPGROUPS);
        for (int w = 0; w < BWD_CONSUMER_WARPGROUPS; w++) {
            coord<k_tile> tile_idx = {blockIdx.z, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + w, kv_head_idx, 0};
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(v_smem[w], g.v, tile_idx, kv_b);
        }

        coord<q_tile> tile_idx = {blockIdx.z, q_start, blockIdx.y, 0};
        tma::expect_bytes(q_b[tic],   sizeof(q_smem[0]));
        tma::load_async<dim::ROW, cache_policy::NORMAL>(q_smem[tic],  g.q,  tile_idx, q_b[tic]);
        tma::expect_bytes(o_b[tic],   sizeof(og_smem[0]));
        tma::load_async<dim::ROW, cache_policy::NORMAL>(og_smem[tic], g.og, tile_idx, o_b[tic]);

        // Don't use tma::load_async for l_smem and d_smem because it sames have
        // some synchronize problem.
        // coord<l_tile> vec_idx = {blockIdx.z, q_start, 0, blockIdx.y};
        // tma::expect_bytes(vec_b[tic], sizeof(l_smem[0])+sizeof(d_smem[0]));
        // tma::load_async(l_smem[tic], g.l, vec_idx, vec_b[tic]);
        // tma::load_async(d_smem[tic], g.d, vec_idx, vec_b[tic]);
    }
    if(warpid==0){
        coord<l_tile> vec_idx = {blockIdx.z, q_start, 0, blockIdx.y};
        load(l_smem[tic], g.l, vec_idx);
        load(d_smem[tic], g.d, vec_idx);
    }
    __syncthreads(); 

    if (warpgroupid == BWD_NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<24>();

        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx+=1) {
                if (qo_idx + 1 < qo_blocks) {
                    bool skip = mask_ptr[qo_idx+];
                    coord<l_tile> vec_idx = {blockIdx.z, qo_idx+1, 0, blockIdx.y};
                    load(l_smem[toc], g.l, vec_idx);
                    load(d_smem[toc], g.d, vec_idx);

                    coord<q_tile> tile_idx = {blockIdx.z, qo_idx+1, blockIdx.y, 0};
                    tma::expect_bytes(q_b[toc],   sizeof(q_smem[0])); 
                    tma::load_async(q_smem[toc], g.q,  tile_idx, q_b[toc]);
                    tma::expect_bytes(o_b[toc],   sizeof(og_smem[0]));
                    tma::load_async(og_smem[toc], g.og, tile_idx, o_b[toc]);
                    tic ^= 1;
                    toc ^= 1;

                    // Don't use tma::load_async for l_smem and d_smem because it sames have
                    // some synchronize problem.
                    // coord<l_tile> vec_idx = {blockIdx.z, qo_idx+1, 0, blockIdx.y};
                    // tma::expect_bytes(vec_b[toc], sizeof(l_smem[0])+sizeof(d_smem[0]));
                    // tma::load_async(l_smem[toc], g.l, vec_idx, vec_b[toc]);
                    // tma::load_async(d_smem[toc], g.d, vec_idx, vec_b[toc]);
                }
                
                wait(compute_done[tic], ((qo_idx - q_start)/(2))%2);
            }
        }
        else if(warpid % WARPGROUP_WARPS == 1) {
            for (auto qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                wait(compute_done[tic], ((qo_idx - q_start)/(2))%2);
                
                coord<qg_tile> tile_idx = {blockIdx.z, qo_idx, blockIdx.y, 0};
                tma::store_add_async<dim::ROW, cache_policy::NORMAL>(g.qg, qg_smem, tile_idx);
                tma::store_async_wait();
                
                if(laneid() == 0) arrive(qg_ready); 
            }
        }
    }
    else {
        rt_fl<16, G::tile_width> kg_reg, vg_reg;
        // [BlockKV, G]
        rt_fl<16, G::tile_h_qo> s_block_t,  p_block_t; 
        rt_fl<16, G::tile_h_qo> ds_block_t, dp_block_t; 
        rt_bf<16, G::tile_h_qo> ds_block_t_mma, p_block_t_mma;
        zero(kg_reg);
        zero(vg_reg);

        if (warpgroupid == 0) {
            warpgroup::increase_registers<256>();
            wait(kv_b, 0);
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                compute_bwd_loop<is_causal, G::tile_h_qo, G::tile_h, G::tile_width, D>(
                    vec_b, q_b, o_b,
                    s_block_t, dp_block_t, p_block_t, ds_block_t, p_block_t_mma, ds_block_t_mma,
                    kg_reg, vg_reg,
                    q_smem, k_smem, v_smem, og_smem, ds_smem, l_smem, d_smem,
                    qo_idx, q_start, tic, toc
                );
                rt_fl<16, G::tile_h_qo> qg_reg[G::num_k_chunk];
                 
                if constexpr(G::num_k_chunk==1){
                    // [BlockKV, Width] * [BlockKV, G] -> [Width, G]
                    warpgroup::mm_AtB(qg_reg[0], k_smem[0], ds_smem[0]);
                    warpgroup::mma_AtB(qg_reg[0], k_smem[1], ds_smem[1]);
                    warpgroup::mma_commit_group(); 
                } else{
                    rt_bf<16, 64> k_reg_chunk[4];
                    warpgroup_load_t_split(k_reg_chunk[0], k_reg_chunk[1], k_smem[0]);
                    warpgroup::sync(warpgroupid+8);
                    warpgroup::mm_AB(qg_reg[0], k_reg_chunk[0], ds_smem[0]);
                    warpgroup::mm_AB(qg_reg[1], k_reg_chunk[1], ds_smem[0]);
                    warpgroup_load_t_split(k_reg_chunk[2], k_reg_chunk[3], k_smem[1]);
                    warpgroup::sync(warpgroupid+8);
                    warpgroup::mma_AB(qg_reg[0], k_reg_chunk[2], ds_smem[1]);
                    warpgroup::mma_AB(qg_reg[1], k_reg_chunk[3], ds_smem[1]);

                    warpgroup::mma_commit_group(); 
                }
    
                wait(qg_ready, toc);
                if (qo_idx > 0) tma::store_async_wait();

                warpgroup::mma_async_wait();

                #pragma unroll
                for(int i=0; i<G::num_k_chunk; i++){
                    warpgroup_store_t(qg_smem, qg_reg[i], i*64);
                }
                warpgroup::sync(warpgroupid+4);
    
                if (warpgroup::laneid() == 0) arrive(compute_done[tic]);
            }
            kv_store<kg_tile, vg_tile>(kg_smem, kg_reg, vg_smem, vg_reg, g, qg_ready, kv_head_idx, toc);
        }
        else {
            warpgroup::increase_registers<224>();
            wait(kv_b, 0);
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                compute_bwd_loop<is_causal, G::tile_h_qo, G::tile_h, G::tile_width, D>(
                    vec_b, q_b, o_b,
                    s_block_t, dp_block_t, p_block_t, ds_block_t, p_block_t_mma, ds_block_t_mma,
                    kg_reg, vg_reg,
                    q_smem, k_smem, v_smem, og_smem, ds_smem, l_smem, d_smem,
                    qo_idx, q_start, tic, toc
                );
            }
            kv_store<kg_tile, vg_tile>(kg_smem, kg_reg, vg_smem, vg_reg, g, qg_ready, kv_head_idx, toc);
        }
    }
}
}

#ifdef TORCH_COMPILE

#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

using namespace NSA_SELECTION_ATTN;

std::vector<torch::Tensor>
nsa_selection_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor indices, int block_count, int block_size, bool causal)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    auto batch    = q.size(0);
    auto seq_len  = q.size(1);
    auto kv_len = k.size(1);
    auto head_dim = q.size(3);
    auto is_causal = causal;
    auto qo_heads = q.size(2);
    auto kv_heads = k.size(2);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(indices.scalar_type() == at::kInt, "Indices tensor must be of type int32");
    TORCH_CHECK(block_size == 64, "Only support block size of 64");

    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(indices.size(0) == batch, "Indices batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(1) == seq_len, "Q sequence length dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1) == seq_len, "K sequence length dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_len, "V sequence length dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(indices.size(1) == seq_len, "Indices sequence length dimension - idx 1 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(indices.size(3) == block_count, "Indices head dimension - idx 3 - must match for all non-vector inputs");

    TORCH_CHECK(qo_heads/kv_heads == 16, "QO_heads/KV_heads must equal 16");
    TORCH_CHECK(qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");
    TORCH_CHECK(q.size(2) == qo_heads, "QO head dimension - idx 2 - must match for QO inputs");
    TORCH_CHECK(k.size(2) == kv_heads, "KV head dimension - idx 2 - must match for KV inputs");
    TORCH_CHECK(v.size(2) == kv_heads, "KV head dimension - idx 2 - must match for KV inputs");
    TORCH_CHECK(indices.size(2) == kv_heads, "Indices head dimension - idx 2 - must match for KV inputs");

    auto hr = qo_heads / kv_heads;

    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();
    int* indices_ptr = indices.data_ptr<int>();

    bf16*  d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v = reinterpret_cast<bf16*>(v_ptr);

    // for the returned outputs
    torch::Tensor o     = torch::empty({static_cast<const uint>(batch),
                                        static_cast<const uint>(seq_len),
                                        static_cast<const uint>(qo_heads),
                                        static_cast<const uint>(head_dim)}, v.options());

    torch::Tensor l_vec = torch::empty({static_cast<const uint>(batch),
                                        static_cast<const uint>(seq_len),
                                        static_cast<const uint>(qo_heads),
                                        static_cast<const uint>(1)},
                                        torch::TensorOptions().dtype(torch::kFloat).device(q.device()).memory_format(at::MemoryFormat::Contiguous));
    bf16*  o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16*  d_o   = reinterpret_cast<bf16*>(o_ptr);

    float* l_ptr = reinterpret_cast<float*>(l_vec.data_ptr<float>());
    float* d_l   = reinterpret_cast<float*>(l_ptr);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (head_dim == 64) {
        using globals = fwd_globals<64>;
        using tile_args = fwd_attend_ker_tile_dims<64>;
        globals::q_gl qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads),  64U};
        globals::k_gl kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_len), static_cast<unsigned int>(kv_heads), 64U};
        globals::v_gl vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_len), static_cast<unsigned int>(kv_heads), 64U};
        globals::l_gl lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), 1U,  static_cast<unsigned int>(qo_heads)};
        globals::o_gl og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 64U};
        globals::indices_gl indices_arg{indices_ptr, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(kv_heads), 
                                   static_cast<unsigned int>(block_count)};

        globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, indices_arg, static_cast<int>(seq_len), static_cast<int>(kv_len), static_cast<int>(hr), 
                  static_cast<int>(block_size), static_cast<int>(block_count)};

        auto mem_size = sizeof(globals::qo_tile) + (sizeof(globals::k_tile) + sizeof(globals::v_tile))*tile_args::stages;
        auto threads  = 32;

        // TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 192");
        dim3 grid(seq_len, kv_heads, batch);

        if (is_causal) {
            cudaFuncSetAttribute(
                fwd_attend_ker<64, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<64, true><<<grid, threads, mem_size, stream>>>(g);
        }
        else {
            cudaFuncSetAttribute(
                fwd_attend_ker<64, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<64, false><<<grid, threads, mem_size, stream>>>(g);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    if (head_dim == 128) {
        using globals      = fwd_globals<128>;
        using tile_args = fwd_attend_ker_tile_dims<128>;
        globals::q_gl qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 128U};
        globals::k_gl kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_len), static_cast<unsigned int>(kv_heads), 128U};
        globals::v_gl vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_len), static_cast<unsigned int>(kv_heads), 128U};
        globals::o_gl og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 128U};
        globals::l_gl lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), 1U,   static_cast<unsigned int>(qo_heads)};
        globals::indices_gl indices_arg{indices_ptr, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(kv_heads), 
                                   static_cast<unsigned int>(block_count)};
        globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, indices_arg, static_cast<int>(seq_len), static_cast<int>(kv_len), static_cast<int>(hr), 
                  static_cast<int>(block_size), static_cast<int>(block_count)};

        auto mem_size = sizeof(globals::qo_tile) + (sizeof(globals::k_tile) + sizeof(globals::v_tile))*tile_args::stages;
        auto threads  = 32;

        // TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 192");
        dim3 grid(seq_len, kv_heads, batch);

        if (is_causal) {
            cudaFuncSetAttribute(
                fwd_attend_ker<128, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<128, true><<<grid, threads, mem_size, stream>>>(g);
        }
        else {
            cudaFuncSetAttribute(
                fwd_attend_ker<128, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<128, false><<<grid, threads, mem_size, stream>>>(g);
        }

        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    return {o, l_vec};
}

std::vector<torch::Tensor>
nsa_selection_attention_backward(torch::Tensor q,
                   torch::Tensor k,
                   torch::Tensor v,
                   torch::Tensor o,
                   torch::Tensor l_vec,
                   torch::Tensor og,
                   torch::Tensor indices,
                   int block_count,
                   int block_size,
                   bool causal)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(l_vec);
    CHECK_INPUT(o);
    CHECK_INPUT(og);

    auto batch    = q.size(0);
    auto seq_len  = q.size(1);
    auto head_dim = q.size(3);
    auto kv_len   = k.size(1);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(block_size == 64, "Only support block size of 64");
    TORCH_CHECK(indices.size(0) == batch, "Indices batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(q.size(0)     == batch, "Q  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0)     == batch, "K  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0)     == batch, "V  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(l_vec.size(0) == batch, "L  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(o.size(0)     == batch, "O  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(og.size(0)    == batch, "OG batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(1)     == seq_len, "Q  sequence length dimension - idx 2 - must match for all inputs");
    // TORCH_CHECK(k.size(1)     == seq_len, "K  sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(v.size(1)     == kv_len, "V  sequence length dimension - idx 2 - must match for all inputs");
    //TORCH_CHECK(l_vec.size(1) == seq_len, "L  sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(o.size(1)     == seq_len, "O  sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(og.size(1)    == seq_len, "OG sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(indices.size(1) == seq_len, "Indices sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3)  == head_dim, "Q  head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3)  == head_dim, "K  head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3)  == head_dim, "V  head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(o.size(3)  == head_dim, "O  head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(og.size(3) == head_dim, "OG head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(indices.size(3) == block_count, "Indices head dimension - idx 3 - must equal Block Count inputs");

    // check if causal
    auto is_causal = causal;

    auto qo_heads = q.size(2);
    auto kv_heads = k.size(2);

    TORCH_CHECK(qo_heads >= kv_heads,     "Q heads must be greater than or equal to K and V heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "Q heads must be divisible by KV heads");

    TORCH_CHECK(q.size(2)     == qo_heads, "Q  heads dimension - idx 1 - must match for QO inputs");
    TORCH_CHECK(l_vec.size(2) == qo_heads, "L  heads dimension - idx 1 - must match for QO inputs");
    TORCH_CHECK(o.size(2)     == qo_heads, "O  heads dimension - idx 1 - must match for QO inputs");
    TORCH_CHECK(og.size(2)    == qo_heads, "OG heads dimension - idx 1 - must match for KV inputs");
    TORCH_CHECK(k.size(2)  == kv_heads, "K  heads dimension - idx 1 - must match for KV inputs");
    TORCH_CHECK(v.size(2)  == kv_heads, "V  heads dimension - idx 1 - must match for KV inputs");
    TORCH_CHECK(indices.size(2) == kv_heads, "Indices heads dimension - idx 1 - must match for QO inputs");

    auto hr = qo_heads / kv_heads;

    c10::BFloat16* q_ptr  = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr  = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr  = v.data_ptr<c10::BFloat16>();
    c10::BFloat16* o_ptr  = o.data_ptr<c10::BFloat16>();
    c10::BFloat16* og_ptr = og.data_ptr<c10::BFloat16>();
    float*         l_ptr  = l_vec.data_ptr<float>();
    int*     indices_ptr  = indices.data_ptr<int>();

    torch::Tensor qg = torch::zeros({static_cast<const uint>(batch),
                                    static_cast<const uint>(seq_len),
                                     static_cast<const uint>(qo_heads),
                                     static_cast<const uint>(head_dim)},   q.options());
    torch::Tensor kg = torch::zeros({static_cast<const uint>(batch),
                                     static_cast<const uint>(kv_len),
                                     static_cast<const uint>(kv_heads),
                                     static_cast<const uint>(head_dim)},   q.options());
    torch::Tensor vg = torch::zeros({static_cast<const uint>(batch),
                                     static_cast<const uint>(kv_len),
                                     static_cast<const uint>(kv_heads),
                                     static_cast<const uint>(head_dim)},   q.options());

    torch::Tensor d_vec = torch::empty({static_cast<const uint>(batch),
                                        static_cast<const uint>(seq_len),
                                        static_cast<const uint>(qo_heads),
                                        static_cast<const uint>(1)},       l_vec.options());
    torch::Tensor mask = torch::zeros({static_cast<const uint>(batch),
                                       static_cast<const uint>(kv_heads),
                                       static_cast<const uint>(seq_len / block_size),
                                       static_cast<const uint>(seq_len)},
                                       torch::TensorOptions().dtype(torch::kBool).
                                       device(l_vec.device()).memory_format(at::MemoryFormat::Contiguous)
    );

    auto           qg_ptr = qg.data_ptr<c10::BFloat16>();
    auto*          kg_ptr = kg.data_ptr<c10::BFloat16>();
    auto*          vg_ptr = vg.data_ptr<c10::BFloat16>();
    float*         d_ptr  = d_vec.data_ptr<float>();
    bool*          mask_ptr = mask.data_ptr<bool>();

    bf16*  d_q  = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k  = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v  = reinterpret_cast<bf16*>(v_ptr);
    bf16*  d_o  = reinterpret_cast<bf16*>(o_ptr);
    bf16*  d_og = reinterpret_cast<bf16*>(og_ptr);
    float* d_l  = reinterpret_cast<float*>(l_ptr);
    float* d_d  = reinterpret_cast<float*>(d_ptr);
    bf16* d_qg  = reinterpret_cast<bf16*>(qg_ptr);
    bf16* d_kg  = reinterpret_cast<bf16*>(kg_ptr);
    bf16* d_vg  = reinterpret_cast<bf16*>(vg_ptr);

    auto mem_size = kittens::MAX_SHARED_MEMORY;
    auto threads  = 4 * kittens::WARP_THREADS;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (head_dim == 64)  {
        using og_tile = st_bf<16, 64>;
        using o_tile  = st_bf<16, 64>;
        using d_tile  = col_vec<st_fl<16, 64>>;

        using og_global = gl<bf16,  -1, -1, -1, -1, tma::descriptor<og_tile, dim::ROW>>;
        using o_global  = gl<bf16,  -1, -1, -1, -1, tma::descriptor<o_tile, dim::ROW>>;
        using d_global  = gl<float, -1, -1, -1, -1, d_tile>;

        using bwd_prep_globals = bwd_prep_globals<64>;
        dim3 grid_bwd(host_ceil_div(seq_len, bwd_prep_globals::seq_block_size), kv_heads, batch);

        og_global prep_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 64U};
        o_global  prep_o_arg {d_o,  static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 64U};
        d_global  prep_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), 1U,  static_cast<unsigned int>(qo_heads)};

        bwd_prep_globals bwd_g{prep_og_arg, prep_o_arg, prep_d_arg};

        cudaFuncSetAttribute(
            bwd_attend_prep_ker<64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        bwd_attend_prep_ker<64><<<grid_bwd, bwd_prep_globals::seq_block_size*kittens::WARP_THREADS, mem_size, stream>>>(bwd_g);
        using bwd_global_args = bwd_globals<64>;
        using bwd_q_global  = bwd_global_args::q_gl;
        using bwd_k_global  = bwd_global_args::k_gl;
        using bwd_v_global  = bwd_global_args::v_gl;
        using bwd_og_global = bwd_global_args::og_gl;
        using bwd_qg_global = bwd_global_args::qg_gl;
        using bwd_kg_global = bwd_global_args::kg_gl;
        using bwd_vg_global = bwd_global_args::vg_gl;
        using bwd_l_global  = bwd_global_args::l_gl;
        using bwd_d_global  = bwd_global_args::d_gl;
        using bwd_indices_global = bwd_global_args::indices_gl;
        int mask_stride_b = kv_heads*seq_len*seq_len/block_size;
        int mask_stride_h = seq_len*seq_len/block_size;

        bwd_q_global  bwd_q_arg {d_q,  static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 64U};
        bwd_k_global  bwd_k_arg {d_k,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_len), static_cast<unsigned int>(kv_heads), 64U};
        bwd_v_global  bwd_v_arg {d_v,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_len), static_cast<unsigned int>(kv_heads), 64U};
        bwd_og_global bwd_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 64U};
        bwd_qg_global bwd_qg_arg{d_qg, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 64U};
        bwd_kg_global bwd_kg_arg{d_kg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_len), static_cast<unsigned int>(kv_heads), 64U};
        bwd_vg_global bwd_vg_arg{d_vg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_len), static_cast<unsigned int>(kv_heads), 64U};
        bwd_l_global  bwd_l_arg {d_l,  static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), 1U,  static_cast<unsigned int>(qo_heads)};
        bwd_d_global  bwd_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), 1U,  static_cast<unsigned int>(qo_heads)};
        bwd_indices_global bwd_indices_arg{indices_ptr, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(block_count)};
        prepare_mask_globals prep_mask_arg{bwd_indices_arg, mask_ptr, mask_stride_b, mask_stride_h, static_cast<int>(block_size)};
        dim3 grid_prep_mask(batch, kv_heads, prepare_mask_globals::seq_blocks);
        prepare_mask<<<grid_prep_mask, prepare_mask_globals::threads, 0, stream>>>(prep_mask_arg);

        bwd_global_args bwd_global{bwd_q_arg,
                        bwd_k_arg,
                        bwd_v_arg,
                        bwd_og_arg,
                        bwd_qg_arg,
                        bwd_kg_arg,
                        bwd_vg_arg,
                        bwd_l_arg,
                        bwd_d_arg,
                        mask_ptr,
                        static_cast<int>(seq_len),
                        static_cast<int>(hr),
                        static_cast<int>(block_size),
                        static_cast<int>(block_count),
                        mask_stride_b,
                        mask_stride_h};

        dim3 grid_bwd_2(host_ceil_div(seq_len, 4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>), kv_heads, batch);
        threads = kittens::WARP_THREADS * BWD_NUM_WORKERS;

        cudaDeviceSynchronize();

        if (is_causal) {
            cudaFuncSetAttribute(
                bwd_attend_ker<64, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                194000
            );
            cudaFuncSetAttribute(
                bwd_attend_ker<64, true>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                85
            );
            bwd_attend_ker<64, true><<<grid_bwd_2, threads, 194000, stream>>>(bwd_global);
        }
        else {
            cudaFuncSetAttribute(
                bwd_attend_ker<64, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                194000
            );
            cudaFuncSetAttribute(
                bwd_attend_ker<64, false>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                85
            );

            bwd_attend_ker<64, false><<<grid_bwd_2, threads, 194000, stream>>>(bwd_global);
        }
    }

    // if (head_dim == 128) {
    //     using og_tile = st_bf<16, 128>;
    //     using o_tile  = st_bf<16, 128>;
    //     using d_tile  = col_vec<st_fl<16, 128>>;

    //     using og_global = gl<bf16,  -1, -1, -1, -1, tma::descriptor<og_tile, dim::ROW>>;
    //     using o_global  = gl<bf16,  -1, -1, -1, -1, tma::descriptor<o_tile, dim::ROW>>;
    //     using d_global  = gl<float, -1, -1, -1, -1, d_tile>;

    //     using bwd_prep_globals = bwd_prep_globals<128>;
    //     dim3 grid_bwd(host_ceil_div(seq_len, bwd_prep_globals::seq_block_size), kv_heads, batch);

    //     og_global prep_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 128U};
    //     o_global  prep_o_arg {d_o,  static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 128U};
    //     d_global  prep_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), 1U,   static_cast<unsigned int>(qo_heads)};

    //     bwd_prep_globals bwd_g{prep_og_arg, prep_o_arg, prep_d_arg};

    //     cudaFuncSetAttribute(
    //         bwd_attend_prep_ker<128>,
    //         cudaFuncAttributeMaxDynamicSharedMemorySize,
    //         mem_size
    //     );

    //     bwd_attend_prep_ker<128><<<grid_bwd, bwd_prep_globals::seq_block_size*kittens::WARP_THREADS, mem_size, stream>>>(bwd_g);

    //     using bwd_q_tile    =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>;
    //     using bwd_k_tile    =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
    //     using bwd_v_tile    =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
    //     using bwd_og_tile   =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>;
    //     using bwd_qg_tile   =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>;
    //     using bwd_kg_tile   =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
    //     using bwd_vg_tile   =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
    //     using bwd_l_tile    = col_vec<st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_h>>;
    //     using bwd_d_tile    = col_vec<st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_h>>;

    //     using bwd_q_global  = gl<bf16,  -1, -1, -1, -1, tma::descriptor<bwd_q_tile, dim::ROW>>;
    //     using bwd_k_global  = gl<bf16,  -1, -1, -1, -1, tma::descriptor<bwd_k_tile, dim::DEPTH>>;
    //     using bwd_v_global  = gl<bf16,  -1, -1, -1, -1, tma::descriptor<bwd_v_tile, dim::DEPTH>>;

    //     using bwd_og_global = gl<bf16,  -1, -1, -1, -1, tma::descriptor<bwd_og_tile, dim::ROW>>;

    //     using bwd_qg_global = gl<bf16, -1, -1, -1, -1, tma::descriptor<bwd_qg_tile, dim::ROW>>;
    //     using bwd_kg_global = gl<bf16, -1, -1, -1, -1, tma::descriptor<bwd_kg_tile, dim::DEPTH>>;
    //     using bwd_vg_global = gl<bf16, -1, -1, -1, -1, tma::descriptor<bwd_vg_tile, dim::DEPTH>>;

    //     using bwd_l_global  = gl<float, -1, -1, -1, -1, bwd_l_tile>;
    //     using bwd_d_global  = gl<float, -1, -1, -1, -1, bwd_d_tile>;

    //     using bwd_global_args = bwd_globals<128>;

    //     bwd_q_global  bwd_q_arg {d_q,  static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 128U};
    //     bwd_k_global  bwd_k_arg {d_k,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_len), static_cast<unsigned int>(kv_heads), 128U};
    //     bwd_v_global  bwd_v_arg {d_v,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_len), static_cast<unsigned int>(kv_heads), 128U};
    //     bwd_og_global bwd_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 128U};
    //     bwd_qg_global bwd_qg_arg{d_qg, static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), static_cast<unsigned int>(qo_heads), 128U};
    //     bwd_kg_global bwd_kg_arg{d_kg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_len), static_cast<unsigned int>(kv_heads), 128U};
    //     bwd_vg_global bwd_vg_arg{d_vg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_len), static_cast<unsigned int>(kv_heads), 128U};
    //     bwd_l_global  bwd_l_arg {d_l,  static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), 1U,   static_cast<unsigned int>(qo_heads)};
    //     bwd_d_global  bwd_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(seq_len), 1U,   static_cast<unsigned int>(qo_heads)};

    //     bwd_global_args bwd_global{
    //                     bwd_q_arg,
    //                     bwd_k_arg,
    //                     bwd_v_arg,
    //                     bwd_og_arg,
    //                     bwd_qg_arg,
    //                     bwd_kg_arg,
    //                     bwd_vg_arg,
    //                     bwd_l_arg,
    //                     bwd_d_arg,
    //                     static_cast<int>(seq_len),
    //                     static_cast<int>(hr)};

    //     // TORCH_CHECK(seq_len % (4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_DIM) == 0, "sequence length must be divisible by 128");
    //     dim3 grid_bwd_2(host_ceil_div(seq_len, 4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>), kv_heads, batch);
    //     threads = kittens::WARP_THREADS * BWD_NUM_WORKERS;


    //     if (is_causal) {
    //         cudaFuncSetAttribute(
    //             bwd_attend_ker<128, true>,
    //             cudaFuncAttributeMaxDynamicSharedMemorySize,
    //             194000
    //         );
    //         cudaFuncSetAttribute(
    //             bwd_attend_ker<128, true>,
    //             cudaFuncAttributePreferredSharedMemoryCarveout,
    //             85
    //         );

    //         bwd_attend_ker<128, true><<<grid_bwd_2, threads, 194000, stream>>>(bwd_global);
    //     }
    //     else {
    //         cudaFuncSetAttribute(
    //             bwd_attend_ker<128, false>,
    //             cudaFuncAttributeMaxDynamicSharedMemorySize,
    //             194000
    //         );
    //         cudaFuncSetAttribute(
    //             bwd_attend_ker<128, false>,
    //             cudaFuncAttributePreferredSharedMemoryCarveout,
    //             85
    //         );

    //         bwd_attend_ker<128, false><<<grid_bwd_2, threads, 194000, stream>>>(bwd_global);
    //     }

    // }

    return {qg, kg, vg, mask};
}

#else

#include "harness.impl"

#endif
