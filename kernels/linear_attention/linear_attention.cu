#include "kittens.cuh"
#include <tuple>

static constexpr int NUM_WORKERS = (8);
static constexpr int NUM_THREADS = (NUM_WORKERS*kittens::WARP_THREADS);
static constexpr int NUM_WARPGROUPS = (NUM_WORKERS/kittens::WARPGROUP_WARPS);

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
    // Each warp in the warpgroup writes to its own portion to avoid races
    // vec.length = 64 (CHUNK_SIZE), 4 warps per warpgroup -> 16 elements per warp
    constexpr int WARPS_PER_WG = kittens::WARPGROUP_WARPS;
    int warp_in_group = warpid() % WARPS_PER_WG;
    int elements_per_warp = vec.length / WARPS_PER_WG;
    int start = warp_in_group * elements_per_warp;
    #pragma unroll
    for(int i = 0; i < elements_per_warp; i++) {
        float val = static_cast<float>(start + i);
        vec.data[start + i] = val;
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
        warp::tma::expect_bytes(qkv_semaphore, 
            size_bytes<typeof(q_smem[0])> + 
            size_bytes<typeof(k_smem[0])> + 
            size_bytes<typeof(k_smem_split[0][0])> + 
            size_bytes<typeof(k_smem_split[0][1])> + 
            size_bytes<typeof(v_smem[0])>);
        
        warp::tma::load_async(q_smem[tic],          g.q,       {batch, head, 0, 0}, qkv_semaphore);
        warp::tma::load_async(k_smem[tic],          g.k,       {batch, head, 0, 0}, qkv_semaphore);
        warp::tma::load_async(k_smem_split[tic][0], g.k_split, {batch, head, 0, 0}, qkv_semaphore);
        warp::tma::load_async(k_smem_split[tic][1], g.k_split, {batch, head, 0, 1}, qkv_semaphore);
        warp::tma::load_async(v_smem[tic],          g.v,       {batch, head, 0, 0}, qkv_semaphore);
    }

    if (warpgroupid == 0) {
        wg_arange(q_decay);
        warp::mul(q_decay, q_decay, -1.0f * slope);
        warp::exp(q_decay, q_decay);
    }
    if (warpgroupid == 1) {
        wg_arange(k_decay);
        warp::mul(k_decay, k_decay, -1.0f); 
        warp::add(k_decay, k_decay, CHUNK_SIZE);
        warp::mul(k_decay, k_decay, -1.0f * slope);
        warp::exp(k_decay, k_decay);
    }

    warp::zero(kv_smem);

    for (int block = 0; block < blocks; block++, tic^=1, toc^=1) {

        wait(qkv_semaphore, tic);
        __syncthreads();

        if (warpid == 0 && block < blocks-1) {
            warp::tma::expect_bytes(qkv_semaphore,
                size_bytes<typeof(q_smem[0])> + 
                size_bytes<typeof(k_smem[0])> + 
                size_bytes<typeof(k_smem_split[0][0])> + 
                size_bytes<typeof(k_smem_split[0][1])> + 
                size_bytes<typeof(v_smem[0])>
            );
            warp::tma::load_async(q_smem[toc],          g.q,       {batch, head, block+1, 0}, qkv_semaphore); 
            warp::tma::load_async(k_smem[toc],          g.k,       {batch, head, block+1, 0}, qkv_semaphore); 
            warp::tma::load_async(k_smem_split[toc][0], g.k_split, {batch, head, block+1, 0}, qkv_semaphore);
            warp::tma::load_async(k_smem_split[toc][1], g.k_split, {batch, head, block+1, 1}, qkv_semaphore);
            warp::tma::load_async(v_smem[toc],          g.v,       {batch, head, block+1, 0}, qkv_semaphore);
        }
        __syncthreads();

        if (warpgroupid == 0) {
            rt_fl<CHUNK_SIZE/kittens::WARPGROUP_WARPS, ATTN_D> linear_o;  

            rt_fl<CHUNK_SIZE/kittens::WARPGROUP_WARPS, CHUNK_SIZE> qk; 
            rt_bf<CHUNK_SIZE/kittens::WARPGROUP_WARPS, CHUNK_SIZE> qk_bf;

            warpgroup::mm_ABt(qk, q_smem[tic], k_smem[tic]);
            warpgroup::mma_async_wait();

            wg_mask(qk, qk, slope);
            warp::copy(qk_bf, qk);

            warpgroup::mm_AB(linear_o, qk_bf, v_smem[tic]);
            warpgroup::mma_async_wait();

            warpgroup::store(o_smem[warpgroupid], linear_o);

            if (block != 0 && warpid == 0) { warp::tma::store_async_wait(); }
            if               (warpid == 0) { warp::tma::store_add_async(g.o, o_smem[warpgroupid], {batch, head, block, 0}); }
        }

        if (warpgroupid == 1) {
            rt_fl<CHUNK_SIZE/kittens::WARPGROUP_WARPS, ATTN_D> linear_o;  

            static_assert(NUM_WARPGROUPS == 2, "NUM_WARPGROUPS must be 2");
            rt_fl<ATTN_F/(kittens::WARPGROUP_WARPS * NUM_WARPGROUPS), ATTN_D> local_kv_0; 
            rt_fl<ATTN_F/(kittens::WARPGROUP_WARPS * NUM_WARPGROUPS), ATTN_D> local_kv_1;

            rt_fl<CHUNK_SIZE/kittens::WARPGROUP_WARPS, ATTN_F/2> local_k_0;
            rt_fl<CHUNK_SIZE/kittens::WARPGROUP_WARPS, ATTN_F/2> local_k_1;

            col_vec<rt_fl<CHUNK_SIZE/kittens::WARPGROUP_WARPS, ATTN_D>> decay; 

            auto kv_subtile_0 = kv_smem.subtile<ATTN_F/2, ATTN_D>({0, 0});
            auto kv_subtile_1 = kv_smem.subtile<ATTN_F/2, ATTN_D>({1, 0});

            warpgroup::mm_AB(linear_o, q_smem[tic], kv_smem);
            warpgroup::load(decay, q_decay);
            warpgroup::mma_async_wait();
            warp::mul_row(linear_o, linear_o, decay);
            
            warpgroup::store(o_smem[warpgroupid], linear_o);

            warpgroup::load(local_k_0, k_smem_split[tic][0]);
            warpgroup::load(local_k_1, k_smem_split[tic][1]);
            warpgroup::load(decay, k_decay);
            warp::mul_row(local_k_0, local_k_0, decay);
            warp::mul_row(local_k_1, local_k_1, decay);
            // Store decayed K back to tic (safe since TMA is done with it)
            // NOT to toc, which races with TMA loading the next block
            warpgroup::store(k_smem_split[tic][0], local_k_0);
            warpgroup::store(k_smem_split[tic][1], local_k_1);

            if (block != 0 && warpid == 4) { warp::tma::store_async_wait(); }
            if               (warpid == 4) { warp::tma::store_add_async(g.o, o_smem[warpgroupid], {batch, head, block, 0}); }

            float block_decay = __expf(-slope * static_cast<float>(CHUNK_SIZE));

            warpgroup::load(local_kv_0, kv_subtile_0); 
            warp::mul(local_kv_0, local_kv_0, block_decay);
            warpgroup::mma_AtB(local_kv_0, k_smem_split[tic][0], v_smem[tic]);
            
            warpgroup::load(local_kv_1, kv_subtile_1); 
            warp::mul(local_kv_1, local_kv_1, block_decay);
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
    size_t B, size_t H, size_t N
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

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <chrono>

#define ATTN_B (1)
#define ATTN_H (8)
#define ATTN_N (1024)

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line ) {
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}

int main(int argc, char **argv) {
    std::cout << "Entered main!" << std::endl;

    constexpr int TOTAL_ELEMENTS_QK = ATTN_B*ATTN_H*ATTN_N*ATTN_D;
    constexpr int TOTAL_ELEMENTS_VO = ATTN_B*ATTN_H*ATTN_N*ATTN_F;
    
    float *slopes      = new float[ATTN_H];
    float *q           = new float[TOTAL_ELEMENTS_QK];
    float *k           = new float[TOTAL_ELEMENTS_QK];
    float *v           = new float[TOTAL_ELEMENTS_VO];
    float *o_ref       = new float[TOTAL_ELEMENTS_VO];
    float *o           = new float[TOTAL_ELEMENTS_VO];
    
    bf16 *q_bf        = new bf16[TOTAL_ELEMENTS_QK];
    bf16 *k_bf        = new bf16[TOTAL_ELEMENTS_QK];
    bf16 *v_bf        = new bf16[TOTAL_ELEMENTS_VO];
    bf16 *o_bf        = new bf16[TOTAL_ELEMENTS_VO];

    if(argc > 1) {
        std::ifstream infile(argv[1]);
        std::cout << "Reading input file: " << argv[1] << std::endl;

        // 1. Read slopes
        for(int i = 0; i < ATTN_H; i++) {
            infile >> slopes[i];
            printf("slopes[%d] = %f\n", i, slopes[i]);
        }
        std::cout << "Finished loading " << ATTN_H << " slopes" << std::endl;

        // 2. Read Q
        for(int i = 0; i < TOTAL_ELEMENTS_QK; i++) infile >> q[i];
        std::cout << "Finished loading " << TOTAL_ELEMENTS_QK << " elements of Q" << std::endl;

        // 3. Read K
        for(int i = 0; i < TOTAL_ELEMENTS_QK; i++) infile >> k[i];
        std::cout << "Finished loading " << TOTAL_ELEMENTS_QK << " elements of K" << std::endl;

        // 4. Read V
        for(int i = 0; i < TOTAL_ELEMENTS_VO; i++) infile >> v[i];
        std::cout << "Finished loading " << TOTAL_ELEMENTS_VO << " elements of V" << std::endl;

        // 5. Read O reference
        for(int i = 0; i < TOTAL_ELEMENTS_VO; i++) infile >> o_ref[i];
        std::cout << "Finished loading " << TOTAL_ELEMENTS_VO << " elements of O_REF" << std::endl;
    }

    // Convert to bf16
    for(uint64_t i = 0; i < TOTAL_ELEMENTS_QK; i++) {
        q_bf[i] = __float2bfloat16(q[i]);
        k_bf[i] = __float2bfloat16(k[i]);
    }
    for(uint64_t i = 0; i < TOTAL_ELEMENTS_VO; i++) {
        v_bf[i] = __float2bfloat16(v[i]);
    }

    bf16 *d_q, *d_k, *d_v, *d_o;
    float *d_slopes;
    
    cudaMalloc(&d_slopes,   ATTN_H            * sizeof(float));
    cudaMalloc(&d_q,        TOTAL_ELEMENTS_QK * sizeof(bf16));
    cudaMalloc(&d_k,        TOTAL_ELEMENTS_QK * sizeof(bf16));
    cudaMalloc(&d_v,        TOTAL_ELEMENTS_VO * sizeof(bf16));
    cudaMalloc(&d_o,        TOTAL_ELEMENTS_VO * sizeof(bf16));

    cudaMemcpy(d_slopes, slopes,   ATTN_H            * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q,      q_bf,     TOTAL_ELEMENTS_QK * sizeof(bf16),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_k,      k_bf,     TOTAL_ELEMENTS_QK * sizeof(bf16),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,      v_bf,     TOTAL_ELEMENTS_VO * sizeof(bf16),  cudaMemcpyHostToDevice);
    
    // zero out d_o
    cudaMemset(d_o, 0, TOTAL_ELEMENTS_VO * sizeof(bf16));

    cudaDeviceSynchronize();
    CudaCheckError();

    // Set up kernel configuration
    unsigned long mem_size = kittens::MAX_SHARED_MEMORY - 1024; 

    // Initialize kernel configuration
    la_globals g = la_init(
        d_q, d_k, d_v, d_o,
        d_slopes,
        ATTN_B, ATTN_H, ATTN_N
    );

    cudaFuncSetAttribute(
        la_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    // Run kernel
    const int ITER = 1;
    cudaDeviceSynchronize();
    CudaCheckError();

    std::cout << "Starting kernel with " << ATTN_B*ATTN_H << " blocks and " << NUM_THREADS << " threads\n";
    float avg_us = 0;
    for(int i = 0; i < ITER; i++) {
        // zero out d_o
        cudaMemset(d_o, 0, TOTAL_ELEMENTS_VO * sizeof(bf16));
        cudaDeviceSynchronize();
        CudaCheckError();

        const auto start = std::chrono::high_resolution_clock::now();
        la_kernel<<<dim3(ATTN_H,ATTN_B), NUM_THREADS, mem_size>>>(g, ATTN_N);
        cudaDeviceSynchronize();
        const auto finish = std::chrono::high_resolution_clock::now();
        CudaCheckError();
        avg_us += std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    }
    avg_us /= ITER;
    std::cout << "Average execution time: " << avg_us << " us" << std::endl;

    // Copy results back and compare
    cudaMemcpy(o_bf, d_o, TOTAL_ELEMENTS_VO * sizeof(bf16), cudaMemcpyDeviceToHost);
    
    // Convert output to float
    for(int i = 0; i < TOTAL_ELEMENTS_VO; i++) {
        o[i] = __bfloat162float(o_bf[i]);
    }

    // Write results to files for analysis
    std::ofstream o_ref_file("printouts/o_ref.txt");
    std::ofstream o_file("printouts/o.txt");
    std::ofstream diff_file("printouts/diff.txt");

    float max_diff = 0, total_diff = 0, total_abs = 0;
    for(int i = 0; i < TOTAL_ELEMENTS_VO; i++) {
        float diff = o[i] - o_ref[i];
        
        o_ref_file << o_ref[i] << ' ';
        o_file << o[i] << ' ';
        diff_file << diff << ' ';
        
        if(i % 64 == 63) {
            o_ref_file << std::endl;
            o_file << std::endl;
            diff_file << std::endl;
        }

        if(abs(diff) > max_diff || isnan(diff)) {
            max_diff = abs(diff);
            if(isnan(diff)) {
                printf("NAN detected idx=%d, o = %f, o_ref = %f, diff = %f\n", i, o[i], o_ref[i], diff);
                break;
            }
        }

        total_abs += abs(o_ref[i]);
        total_diff += abs(diff);
    }

    // Print error metrics
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);
    std::cout.width(12);
    std::cout << "O | avg_diff=" << (total_diff/TOTAL_ELEMENTS_VO) 
              << ", avg_abs=" << (total_abs/TOTAL_ELEMENTS_VO)
              << ", rel_diff=" << 100*(total_diff/total_abs) 
              << "%, max_diff=" << max_diff << std::endl;

    // Cleanup
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_slopes);

    delete[] slopes;
    delete[] q;
    delete[] k;
    delete[] v;
    delete[] o;
    delete[] o_ref;
    delete[] q_bf;
    delete[] k_bf;
    delete[] v_bf;
    delete[] o_bf;

    return 0;
}
