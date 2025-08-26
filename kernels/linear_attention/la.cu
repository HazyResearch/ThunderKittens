#include "kittens.cuh"
#include <tuple>

#define NUM_WORKERS (8)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define NUM_WARPGROUPS (NUM_WORKERS/kittens::WARPGROUP_WARPS)

using namespace kittens;

#define CHUNK_SIZE 64
#define ATTN_D 128
#define ATTN_F 128

struct la_globals { 
    // shapes    
    using q_tile       = st_bf<CHUNK_SIZE, ATTN_F>;
    using k_tile       = st_bf<CHUNK_SIZE, ATTN_F>;
    using k_tile_split = st_bf<CHUNK_SIZE, ATTN_F/2>;
    using v_tile       = st_bf<CHUNK_SIZE, ATTN_D>;
    using o_tile       = st_bf<CHUNK_SIZE, ATTN_D>;

    // global layouts
    using q_gl       = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl       = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using k_split_gl = gl<bf16,  -1, -1, -1, -1, k_tile_split>;
    using v_gl       = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using o_gl       = gl<bf16,  -1, -1, -1, -1, o_tile>;

    // pointers
    q_gl       q;
    k_gl       k;
    k_split_gl k_split;
    v_gl       v;
    o_gl       o;

    float *slopes;
};

template<ducks::rt::row_layout RT>
__device__ static inline void wg_mask(RT &dst, const RT &src, float slope) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(0.0f);

    int i = warpid() % kittens::WARPGROUP_WARPS;
    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        if(j < i) { // below the diagonal, copy
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                const int row = (i * dst.tile_size_row) + ((k % 2) * 8) + ((laneid() / 4)); 
                const int col_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int col_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;

                const float decay_x = __expf(-1.0f * slope * (row - col_x));
                const float decay_y = __expf(-1.0f * slope * (row - col_y));

                dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x * decay_x;
                dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y * decay_y;
            }
        }
        else if(j > i) { // above the diagonal, zero
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = packed_val;
            }
        }
        else { // on the diagonal, interesting!
            constexpr uint32_t MASK_X = 0xFF773311, MASK_Y = 0xF7733110; // magic numbers for on-diagonal core matrices

            const int row   = (i * dst.tile_size_row) + ((1 % 2) * 8) + ((laneid() / 4)); 
            const int col_x = (j * dst.tile_size_col) + ((1 / 2) * 8) + ((laneid() % 4) * 2);
            const int col_y = (j * dst.tile_size_col) + ((1 / 2) * 8) + ((laneid() % 4) * 2) + 1;

            const float decay_x = __expf(-1.0f * slope * (row - col_x));
            const float decay_y = __expf(-1.0f * slope * (row - col_y));

            dst.tiles[i][j].data[1].x = src.tiles[i][j].data[1].x * decay_x;
            dst.tiles[i][j].data[1].y = src.tiles[i][j].data[1].y * decay_y;

            dst.tiles[i][j].data[2] = packed_val; // above diagonal, zero
            
            if((MASK_X >> laneid()) & 1) {
                int row       = (i * dst.tile_size_row) + ((0 % 2) * 8) + ((laneid() / 4)); 
                int col_x     = (j * dst.tile_size_col) + ((0 / 2) * 8) + ((laneid() % 4) * 2);
                float decay_x = __expf(-1.0f * slope * (row - col_x));
                dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x * decay_x;

                row     = (i * dst.tile_size_row) + ((3 % 2) * 8) + ((laneid() / 4)); 
                col_x   = (j * dst.tile_size_col) + ((3 / 2) * 8) + ((laneid() % 4) * 2);
                decay_x = __expf(-1.0f * slope * (row - col_x));
                dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x * decay_x;
            }
            else {
                dst.tiles[i][j].data[0].x = 0.0f;
                dst.tiles[i][j].data[3].x = 0.0f;
            }
            if((MASK_Y >> laneid()) & 1) {
                int row       = (i * dst.tile_size_row) + ((0 % 2) * 8) + ((laneid() / 4)); 
                int col_y     = (j * dst.tile_size_col) + ((0 / 2) * 8) + ((laneid() % 4) * 2) + 1;
                float decay_y = __expf(-1.0f * slope * (row - col_y));
                dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y * decay_y;

                row     = (i * dst.tile_size_row) + ((3 % 2) * 8) + ((laneid() / 4)); 
                col_y   = (j * dst.tile_size_col) + ((3 / 2) * 8) + ((laneid() % 4) * 2) + 1;
                decay_y = __expf(-1.0f * slope * (row - col_y));    
                dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y * decay_y;
            }
            else {
                dst.tiles[i][j].data[0].y = 0.0f;
                dst.tiles[i][j].data[3].y = 0.0f;
            }
        }
        __syncwarp();
    }
    group<4>::sync(10 + warpgroupid());
}

__device__ static inline void wg_arange(auto &vec) {
    #pragma unroll
    for(int i = 0; i < vec.length; i++) {
        float val = static_cast<float>(i) + ((warpid() % kittens::WARPGROUP_WARPS) * vec.length); 
        vec.data[i] = val; 
    }
    group<4>::sync(5 + warpgroupid());
}


__global__ __launch_bounds__(NUM_THREADS, 1)
void la_kernel (const __grid_constant__ la_globals g, int N)  
{
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int batch = blockIdx.y;
    const int head  = blockIdx.x;

    float slope = g.slopes[head];

    // smem
    using q_tile              = st_bf<CHUNK_SIZE, ATTN_F>;

    using k_tile              = st_bf<CHUNK_SIZE, ATTN_F>;
    using k_tile_split        = st_bf<CHUNK_SIZE, ATTN_F/2>; 

    using v_tile              = st_bf<CHUNK_SIZE, ATTN_D>;
    using o_tile              = st_bf<CHUNK_SIZE, ATTN_D>;
    using kv_state_tile       = st_bf<ATTN_F,     ATTN_D>;

    q_tile             (&q_smem)[2]    = al.allocate<q_tile, 2>();          // 64 x 128 x 2 x 2 = 32k
    k_tile             (&k_smem)[2]    = al.allocate<k_tile, 2>();          // 64 x 128 x 2 x 2 = 32k
    k_tile_split (&k_smem_split)[2][2] = al.allocate<k_tile_split, 2, 2>(); // 64 x 128 x 2 x 2 = 32k
    v_tile             (&v_smem)[2]    = al.allocate<v_tile, 2>();          // 64 x 128 x 2 x 2 = 32k

    kv_state_tile (&kv_smem)                 = al.allocate<kv_state_tile>();          // 128 x 128 x 2 = 32k
    o_tile         (&o_smem)[NUM_WARPGROUPS] = al.allocate<o_tile, NUM_WARPGROUPS>(); // 64 x 128 x 2 x 2 = 32k

    col_vec<st_fl<CHUNK_SIZE, ATTN_D>> (&q_decay) = al.allocate<col_vec<st_fl<CHUNK_SIZE, ATTN_D>>>();
    col_vec<st_fl<CHUNK_SIZE, ATTN_D>> (&k_decay) = al.allocate<col_vec<st_fl<CHUNK_SIZE, ATTN_D>>>();

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/4;
    int blocks      = N / (CHUNK_SIZE);

    int tic = 0, toc = 1;

    __shared__ semaphore qkv_semaphore;
    if (warpid == 0) {
        init_semaphore(qkv_semaphore, 0, 1);
        tma::expect_bytes(qkv_semaphore, 
            size_bytes<typeof(q_smem[0])> + 
            size_bytes<typeof(k_smem[0])> + 
            size_bytes<typeof(k_smem_split[0][0])> + 
            size_bytes<typeof(k_smem_split[0][1])> + 
            size_bytes<typeof(v_smem[0])>);
        
        tma::load_async(q_smem[tic],          g.q,       {batch, head, 0, 0}, qkv_semaphore);
        tma::load_async(k_smem[tic],          g.k,       {batch, head, 0, 0}, qkv_semaphore);
        tma::load_async(k_smem_split[tic][0], g.k_split, {batch, head, 0, 0}, qkv_semaphore);
        tma::load_async(k_smem_split[tic][1], g.k_split, {batch, head, 0, 1}, qkv_semaphore);
        tma::load_async(v_smem[tic],          g.v,       {batch, head, 0, 0}, qkv_semaphore);
    }

    if (warpgroupid == 0) {
        wg_arange(q_decay);
        warpgroup::mul(q_decay, q_decay, -1.0f * slope);
        warpgroup::exp(q_decay, q_decay);
    }
    if (warpgroupid == 1) {
        wg_arange(k_decay);
        warpgroup::mul(k_decay, k_decay, -1.0f); 
        warpgroup::add(k_decay, k_decay, CHUNK_SIZE);
        warpgroup::mul(k_decay, k_decay, -1.0f * slope);
        warpgroup::exp(k_decay, k_decay);
    }

    zero(kv_smem);

    for (int block = 0; block < blocks; block++, tic^=1, toc^=1) {

        wait(qkv_semaphore, tic);
        __syncthreads();

        if (warpid == 0 && block < blocks-1) {
            tma::expect_bytes(qkv_semaphore,
                size_bytes<typeof(q_smem[0])> + 
                size_bytes<typeof(k_smem[0])> + 
                size_bytes<typeof(k_smem_split[0][0])> + 
                size_bytes<typeof(k_smem_split[0][1])> + 
                size_bytes<typeof(v_smem[0])>
            );
            tma::load_async(q_smem[toc],          g.q,       {batch, head, block+1, 0}, qkv_semaphore); 
            tma::load_async(k_smem[toc],          g.k,       {batch, head, block+1, 0}, qkv_semaphore); 
            tma::load_async(k_smem_split[toc][0], g.k_split, {batch, head, block+1, 0}, qkv_semaphore);
            tma::load_async(k_smem_split[toc][1], g.k_split, {batch, head, block+1, 1}, qkv_semaphore);
            tma::load_async(v_smem[toc],          g.v,       {batch, head, block+1, 0}, qkv_semaphore);
        }
        __syncthreads();

        if (warpgroupid == 0) {
            rt_fl<CHUNK_SIZE/kittens::WARPGROUP_WARPS, ATTN_D> linear_o;  

            rt_fl<CHUNK_SIZE/kittens::WARPGROUP_WARPS, CHUNK_SIZE> qk; 
            rt_bf<CHUNK_SIZE/kittens::WARPGROUP_WARPS, CHUNK_SIZE> qk_bf;

            warpgroup::mm_ABt(qk, q_smem[tic], k_smem[tic]);
            warpgroup::mma_async_wait();

            wg_mask(qk, qk, slope);
            copy(qk_bf, qk);

            warpgroup::mm_AB(linear_o, qk_bf, v_smem[tic]);
            warpgroup::mma_async_wait();

            warpgroup::store(o_smem[warpgroupid], linear_o);

            if (block != 0 && warpid == 0) { tma::store_async_wait(); }
            if               (warpid == 0) { tma::store_add_async(g.o, o_smem[warpgroupid], {batch, head, block, 0}); }
        }

        if (warpgroupid == 1) {
            rt_fl<CHUNK_SIZE/kittens::WARPGROUP_WARPS, ATTN_D> linear_o;  

            static_assert(NUM_WARPGROUPS == 2, "NUM_WARPGROUPS must be 2");
            rt_fl<ATTN_F/(kittens::WARPGROUP_WARPS * NUM_WARPGROUPS), ATTN_D> local_kv_0; 
            rt_fl<ATTN_F/(kittens::WARPGROUP_WARPS * NUM_WARPGROUPS), ATTN_D> local_kv_1;

            rt_fl<CHUNK_SIZE/kittens::WARPGROUP_WARPS, ATTN_F/2> local_k_0;
            rt_fl<CHUNK_SIZE/kittens::WARPGROUP_WARPS, ATTN_F/2> local_k_1;

            col_vec<rt_fl<CHUNK_SIZE/kittens::WARPGROUP_WARPS, ATTN_D>> decay; 

            auto kv_subtile_0 = subtile_inplace<ATTN_F/2, ATTN_D>(kv_smem, {0, 0});
            auto kv_subtile_1 = subtile_inplace<ATTN_F/2, ATTN_D>(kv_smem, {1, 0});

            warpgroup::mm_AB(linear_o, q_smem[tic], kv_smem);
            warpgroup::load(decay, q_decay);
            warpgroup::mma_async_wait();
            mul_row(linear_o, linear_o, decay);
            
            warpgroup::store(o_smem[warpgroupid], linear_o);

            warpgroup::load(local_k_0, k_smem_split[tic][0]);
            warpgroup::load(local_k_1, k_smem_split[tic][1]);
            warpgroup::load(decay, k_decay);
            mul_row(local_k_0, local_k_0, decay);
            mul_row(local_k_1, local_k_1, decay);
            warpgroup::store(k_smem_split[toc][0], local_k_0);
            warpgroup::store(k_smem_split[toc][1], local_k_1);

            if (block != 0 && warpid == 4) { tma::store_async_wait(); }
            if               (warpid == 4) { tma::store_add_async(g.o, o_smem[warpgroupid], {batch, head, block, 0}); }

            float block_decay = __expf(-slope * static_cast<float>(CHUNK_SIZE));

            warpgroup::load(local_kv_0, kv_subtile_0); 
            mul(local_kv_0, local_kv_0, block_decay);
            warpgroup::mma_AtB(local_kv_0, k_smem_split[tic][0], v_smem[tic]);
            
            warpgroup::load(local_kv_1, kv_subtile_1); 
            mul(local_kv_1, local_kv_1, block_decay);
            warpgroup::mma_AtB(local_kv_1, k_smem_split[tic][1], v_smem[tic]);
    
            warpgroup::mma_async_wait();

            warpgroup::store(kv_subtile_0, local_kv_0);
            warpgroup::store(kv_subtile_1, local_kv_1);
        }
    }

    tma::store_async_wait();
    __syncthreads();
}

la_globals la_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    float *d_slopes,
    int B, int H, int N
) {
    // global pointers. 
    using q_tile       = st_bf<CHUNK_SIZE,   ATTN_F>;
    using k_tile       = st_bf<CHUNK_SIZE,   ATTN_F>;
    using k_split_tile = st_bf<CHUNK_SIZE,   ATTN_F/2>;
    using v_tile       = st_bf<CHUNK_SIZE,   ATTN_D>;
    using o_tile       = st_bf<CHUNK_SIZE,   ATTN_D>;
    
    using q_global       = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_global       = gl<bf16, -1, -1, -1, -1, k_tile>;
    using k_split_global = gl<bf16, -1, -1, -1, -1, k_split_tile>;
    using v_global       = gl<bf16, -1, -1, -1, -1, v_tile>;
    using o_global       = gl<bf16, -1, -1, -1, -1, o_tile>;

    using globals = la_globals;
    q_global             q_arg{d_q, B, H, N, ATTN_F};
    k_global             k_arg{d_k, B, H, N, ATTN_F};
    k_split_global k_split_arg{d_k, B, H, N, ATTN_F}; 
    v_global             v_arg{d_v, B, H, N, ATTN_D};
    o_global             o_arg{d_o, B, H, N, ATTN_D};

    globals g{
        q_arg, k_arg, k_split_arg, v_arg, o_arg,
        d_slopes
    };

    return g;
}

#include "harness.impl"