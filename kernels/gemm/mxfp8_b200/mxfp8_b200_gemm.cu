#include "kittens.cuh"

using namespace kittens;

namespace mxfp8_gemm {

template <int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH, int _SUPERGROUP_SIZE, int _NUM_D_TILES>
struct config {
    static_assert(_LOAD_PIPE_DEPTH > 0, "LOAD_PIPE_DEPTH must be greater than 0");
    static_assert(_EPI_PIPE_DEPTH > 0, "EPI_PIPE_DEPTH must be greater than 0");
    static_assert(_SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be greater than 0");
    static_assert(_NUM_D_TILES > 0, "NUM_D_TILES must be greater than 0");
    static_assert(_EPI_PIPE_DEPTH <= 1 || _NUM_D_TILES >= 2, "NUM_D_TILES must be at least 2 if EPI_PIPE_DEPTH > 1");

    static constexpr int CLUSTER_SIZE = 2;

    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = 128;

    static constexpr int NUM_D_TILES = _NUM_D_TILES;
};

template <typename C>
struct globals {
    using A_fp8_tile = st_fp8e4m3<C::Mb / 2, C::Kb>;
    using A_sc_tile  = st_fp8e8m0<32, 16, false>;
    using B_fp8_tile = st_fp8e4m3<C::Nb / 2, C::Kb>;
    using B_sc_tile  = st_fp8e8m0<32, 16, false>;
    using D_tile     = st_bf<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>;

    using A_gl    = gl<fp8e4m3,  1,  1, -1, -1, A_fp8_tile>;
    using A_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, A_sc_tile>;
    using B_gl    = gl<fp8e4m3,  1,  1, -1, -1, B_fp8_tile>;
    using B_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, B_sc_tile>;
    using D_gl    = gl<bf16,     1,  1, -1, -1, D_tile>;

    A_gl A;       // M x K
    A_sc_gl A_sc; // (M // 128) x (K // 128) x 32 x 16
    B_gl B;       // N x K
    B_sc_gl B_sc; // (N // 128) x (K // 128) x 32 x 16
    D_gl D;       // M x N

    __host__ inline dim3 grid() const {
        return dim3(min((D.rows()/(C::Mb/2))*(D.cols()/C::Nb), num_sms()));
    }
};

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    struct input_tiles_t {
        typename G::A_fp8_tile A;
        typename G::B_fp8_tile B;
    };
    struct input_scales_t {
        typename G::A_sc_tile A;
        typename G::B_sc_tile B[2];
    };
    struct outputs_t {
        typename G::D_tile D[C::NUM_D_TILES];
    };

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    static_assert(sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH +
                  sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                  sizeof(outputs_t) <= C::DYNAMIC_SHARED_MEMORY);
    input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<input_tiles_t, C::LOAD_PIPE_DEPTH>();
    input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<input_scales_t, C::LOAD_PIPE_DEPTH>();
    outputs_t &output_tiles = sm_allocator.allocate<outputs_t>();

    // Declare tensor memory
    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    // Set up mbarriers
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore inputs_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    // Thread metadata
    int lane_id = warp::laneid();
    int warp_id = warpgroup::warpid();
    int warpgroup_id = warpgroup::groupid();
    int cta_id = cluster_ctarank();
    int cluster_id = clusterIdx().x;

    // Block dimensions
    const int num_blocks_per_row = g.D.cols() / C::Nb;
    const int num_blocks_per_col = g.D.rows() / C::Mb;
    const int num_blocks = num_blocks_per_row * num_blocks_per_col;
    const int num_iters_per_block = g.A.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_blocks_per_row;

    // Declare stage and phasebits for semaphore waits
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Main divergence
    if (warpgroup_id == C::NUM_WARPGROUPS - 1) {
        // Producer group
        if (warp_id == 3 && lane_id == 0) {
            // Load input matrices and scales to shared memory
            pdl::wait();
            everyone::tma::cluster::wait_aligned();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_blocks_per_col - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                for (int i = 0; i < num_iters_per_block; ++i) {
                    tma::cluster::wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    tma::cluster::load_async(input_tiles[stage].A,          g.A,    {row_block_idx * 2 + cta_id, i},       inputs_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B,          g.B,    {col_block_idx * 2 + cta_id, i},       inputs_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    tma::cluster::load_async(input_scales[stage].A,         g.A_sc, {row_block_idx * 2 + cta_id, i, 0, 0}, inputs_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    tma::cluster::load_async(input_scales[stage].B[cta_id], g.B_sc, {col_block_idx * 2 + cta_id, i, 0, 0}, inputs_arrived[stage], (uint16_t)(0b11), 0);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 1 && lane_id == 0) {
            // Load A and B scales from shared memory to tensor memory
            everyone::tma::cluster::wait_aligned();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32*C::LOAD_PIPE_DEPTH>>(384);
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::cluster::expect_bytes(inputs_arrived[stage], 2 * (sizeof(input_tiles_t) + sizeof(input_scales_t)));
                    tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * 16);
                    load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[stage].A);
                    auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * 32);
                    auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * 32 + 16);
                    load_mxnv_scale_async2(B_sc_tm_subtile_0, input_scales[stage].B[0]);
                    load_mxnv_scale_async2(B_sc_tm_subtile_1, input_scales[stage].B[1]);
                    kittens::detail::tcgen05::commit<2>(scales_arrived[stage], 0b1);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0 && lane_id == 0) {
            // Launch tensor core matrix multiply
            everyone::tma::cluster::wait_aligned();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32*C::LOAD_PIPE_DEPTH>>(384);
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                tma::cluster::wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                update_phasebit<1>(phasebits, 0);
                for (int i = 0; i < num_iters_per_block; i++) {
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    if (i == 0) mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * 16),
                                        B_sc_tm.template subtile<full_tt_fp8e8m0<32>>(stage * 32),
                                        inputs_finished[stage]);
                    else mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                  A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * 16),
                                  B_sc_tm.template subtile<full_tt_fp8e8m0<32>>(stage * 32),
                                  inputs_finished[stage]);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                kittens::detail::tcgen05::commit<2>(outputs_arrived);
            }
        }
    } else {
        // Consumer group
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_blocks_per_col - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            // Wait for the last matmul to complete
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);

            // Load the output from tensor memory into registers
            rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; i++)
                warpgroup::load_async(D_reg[i], out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * i));
            tensor_load_wait();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1); // signal CTA 0

            // Store to HBM with pipelined epilogue
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg[i]);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx * 2 + cta_id, col_block_idx * C::EPI_PIPE_DEPTH + i});
            }
        }
        warpgroup::sync(1);
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

} // namespace mxfp8_gemm

namespace mxfp8_quantize {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPS = 2; // 64 threads, 2 rows per thread
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int TILE_SIZE = 128;   // This should not change
    static constexpr int K_BLOCK_SIZE = 32; // This should not change

    using A_bf16_tile = st_bf<TILE_SIZE, TILE_SIZE, false>;
    using A_fp8_tile  = st_fp8e4m3<TILE_SIZE, TILE_SIZE, false>;
    using A_sc_tile   = st_fp8e8m0<32, 16, false>;

    using A_bf16_gl = gl<bf16, 1, 1, -1, -1, A_bf16_tile>;
    using A_fp8_gl = gl<fp8e4m3, 1, 1, -1, -1, A_fp8_tile>;
    using A_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, A_sc_tile>;

    A_bf16_gl A_bf16; // M x N
    A_fp8_gl A_fp8;   // M x N
    A_sc_gl A_sc;     // (M // 128) x (N // 128) x 32 x 16

    __host__ inline dim3 grid() const {
        return dim3(A_bf16.cols() / TILE_SIZE, A_bf16.rows() / TILE_SIZE);
    }
    __host__ inline int dynamic_shared_memory() const {
        return TILE_SIZE * TILE_SIZE * sizeof(bf16) + 1024;
    }
};

__device__ inline void kernel(const globals &G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    globals::A_bf16_tile &A_bf16_smem = sm_allocator.allocate<globals::A_bf16_tile>();
    globals::A_fp8_tile  &A_fp8_smem = *reinterpret_cast<globals::A_fp8_tile *>(&A_bf16_smem);
    globals::A_sc_tile   &A_sc_smem = *reinterpret_cast<globals::A_sc_tile *>(
        reinterpret_cast<uint64_t>(&A_fp8_smem) + sizeof(A_fp8_smem));

    // Calculate indices
    const int tid = threadIdx.x;
    const int row = blockIdx.y;
    const int col = blockIdx.x;

    // Initialize mbarrier and initiate TMA load
    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect(inputs_arrived, A_bf16_smem);
        tma::load_async(A_bf16_smem, G.A_bf16, {row, col}, inputs_arrived);
    }

    // Wait for the TMA load to complete
    __syncthreads();
    wait(inputs_arrived, 0);

    // We have 64 threads per block. Each thread handles 2 rows of 128 elements
    constexpr int ROWS_PER_THREAD = 2;
    constexpr int NUM_K_BLOCKS = globals::TILE_SIZE / globals::K_BLOCK_SIZE; // 4
    constexpr int N_PER_K_BLOCK = globals::TILE_SIZE / 2 / NUM_K_BLOCKS;     // 16
    bf16_2 A_bf16_reg[ROWS_PER_THREAD][NUM_K_BLOCKS][N_PER_K_BLOCK];
    fp8e8m0 A_sc_reg[ROWS_PER_THREAD][NUM_K_BLOCKS];

    // Load input matrix from shared memory (custom swizzling)
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int tile_row = tid + (r*64);
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS; i++) {
            int k_block_idx = (i + tid/8) % NUM_K_BLOCKS; // 8 SMEM banks per K-block
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                int tile_col = k_block_idx*globals::K_BLOCK_SIZE + ((tid+j)*2)%globals::K_BLOCK_SIZE;
                int offset = (tile_row*globals::TILE_SIZE + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(A_bf16_reg[r][i][j], static_cast<uint32_t>(__cvta_generic_to_shared(&A_bf16_smem)) + offset);
            }
        }
    }
    __syncthreads();

    // Perform MXFP8 quantization
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int tile_row = tid + (r*64);
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS; i++) {
            int k_block_idx = (i + tid/8) % NUM_K_BLOCKS; // 8 SMEM banks per K-block

            // Calculate absolute maximum
            bf16_2 amax = __habs2(A_bf16_reg[r][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; j++)
                amax = __hmax2(amax, __habs2(A_bf16_reg[r][i][j]));

            // Compute scales
            // Must narrow to e8m0, rounding towards positive infinity and saturating to finite, then clamp
            // https://arxiv.org/pdf/2506.08027
            float scale = max(__bfloat162float(__hmax(amax.x, amax.y)) * 0.002232142857f, 0.000000000001f); // in theory lower clamp is not needed
            A_sc_reg[r][k_block_idx].__x = __nv_cvt_float_to_e8m0(scale, __NV_SATFINITE, cudaRoundPosInf); // causes stack frame, but ignorable
            float scale_inv = 1.0f / static_cast<float>(A_sc_reg[r][k_block_idx]); // utilizes the float() operator defined in __nv_fp8x2_e8m0

            // Quantize and store to shared memory
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                int tile_col = k_block_idx*globals::K_BLOCK_SIZE + ((tid+j)*2)%globals::K_BLOCK_SIZE;
                int offset = (tile_row*globals::TILE_SIZE + tile_col) * sizeof(fp8e4m3);
                fp8e4m3 A_fp8_reg[2] = {
                    __nv_fp8_e4m3(__bfloat162float(A_bf16_reg[r][i][j].x) * scale_inv),
                    __nv_fp8_e4m3(__bfloat162float(A_bf16_reg[r][i][j].y) * scale_inv)
                };
                asm volatile("{st.shared.b16 [%0], %1;}"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_fp8_smem)) + offset)
                       "h"(*reinterpret_cast<uint16_t *>(&A_fp8_reg[0])));
            }
        }

        // Store the scales to shared memory. Each thread will access 1 bank, so no need to swizzle,
        // but we do have to follow this complicated layout pattern made by NVIDIA:
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
        int scale_offset = (tile_row % 32) * 16 + // row
                           (tile_row / 32) * 4;   // column
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_sc_smem)) + scale_offset)
               "r"(*reinterpret_cast<uint32_t *>(&A_sc_reg[r][0])));
    }

    // Store to global memory
    __syncthreads();
    if (tid == 0) {
        tma::store_async(G.A_fp8, A_fp8_smem, {row, col});
        tma::store_async(G.A_sc,  A_sc_smem,  {row, col, 0, 0});
    }
}

} // namespace mxfp8_quantize

#ifndef TORCH_COMPILE

#include "../common.cuh"

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE) __launch_bounds__(C::NUM_THREADS)
__global__ void kernel_entrypoint(const __grid_constant__ mxfp8_gemm::globals<C> g) {
    mxfp8_gemm::kernel<C>(g);
}

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    using G = mxfp8_gemm::globals<C>;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Mb=" << C::Mb << " Nb=" << C::Nb << " Kb=" << C::Kb << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE
              << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH << " EPI_PIPE_DEPTH=" << C::EPI_PIPE_DEPTH << " NUM_D_TILES=" << C::NUM_D_TILES << "\n";

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = size_t(M) * K + size_t(N) * K + size_t(M) * N * 2;
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory
    std::vector<__nv_fp8_e4m3*> d_A(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_B(arg_group_count);
    std::vector<__nv_fp8_e8m0*> d_A_sc(arg_group_count);
    std::vector<__nv_fp8_e8m0*> d_B_sc(arg_group_count);
    std::vector<__nv_bfloat16*> d_D(arg_group_count);
    __nv_bfloat16* d_D_ref;
    for (int i = 0; i < arg_group_count; i++) {
        cudaMalloc(&d_A[i], M*K*sizeof(__nv_fp8_e4m3));
        cudaMalloc(&d_B[i], N*K*sizeof(__nv_fp8_e4m3));
        cudaMalloc(&d_A_sc[i], M*K*sizeof(__nv_fp8_e8m0)/32);
        cudaMalloc(&d_B_sc[i], N*K*sizeof(__nv_fp8_e8m0)/32);
        cudaMalloc(&d_D[i], M*N*sizeof(__nv_bfloat16));
    }
    cudaMalloc(&d_D_ref, M*N*sizeof(__nv_bfloat16));

    // Initialize matrices with random values on device
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_A[i], M*K, seed + i*100, -448.0f, 448.0f);
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_B[i], N*K, seed + i*100 + 1, -448.0f, 448.0f);
        fill<__nv_fp8_e8m0, FillMode::RANDOM>(d_A_sc[i], M*K/32, seed + i*100 + 2, 0.1f, 10.0f);
        fill<__nv_fp8_e8m0, FillMode::RANDOM>(d_B_sc[i], N*K/32, seed + i*100 + 3, 0.1f, 10.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_D_ref, M*N, 0.0f);

    // Compute reference GEMM on device
    reference_blockscaled_gemm<__nv_fp8_e4m3, __nv_fp8_e8m0, __nv_bfloat16, 32>(
        d_D_ref, d_A[0], d_B[0], d_A_sc[0], d_B_sc[0], M, N, K);
    cudaDeviceSynchronize();

    // Prepare kernel inputs
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::A_gl Ag{d_A[i], nullptr, nullptr, M, K};
        typename G::A_sc_gl Asg{d_A_sc[i], M/128, K/128, nullptr, nullptr};
        typename G::B_gl Bg{d_B[i], nullptr, nullptr, N, K};
        typename G::B_sc_gl Bsg{d_B_sc[i], N/128, K/128, nullptr, nullptr};
        typename G::D_gl Dg{d_D[i], nullptr, nullptr, M, N};
        g.push_back(G{Ag, Asg, Bg, Bsg, Dg});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(kernel_entrypoint<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::DYNAMIC_SHARED_MEMORY));

    // Prepare kernel launch configuration
    LaunchConfig<true, true> launch_config(g[0].grid(), C::NUM_THREADS, C::DYNAMIC_SHARED_MEMORY, 0, C::CLUSTER_SIZE);

    // Number of iterations
    int num_warmups = ncu ? 0 : 5;
    int num_iters = ncu ? 1 : 10;

    // Warmup
    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[idx]);
    }

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[idx]);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Check correctness
    check_correctness(d_D[0], d_D_ref, M * N);

    // Cleanup
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_A_sc[i]);
        cudaFree(d_B_sc[i]);
        cudaFree(d_D[i]);
    }
    cudaFree(d_D_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

int main() {
    int N;
    bool ncu = false;

    // Template parameters: LOAD_PIPE_DEPTH, EPI_PIPE_DEPTH, SUPERGROUP_SIZE, NUM_D_TILES
    N = 1024;
    run_benchmark<mxfp8_gemm::config<6, 16, 12, 4>>(N, N, N, ncu);
    N = 2048;
    run_benchmark<mxfp8_gemm::config<6, 16, 12, 4>>(N, N, N, ncu);
    N = 4096;
    run_benchmark<mxfp8_gemm::config<6, 16, 8, 2>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<mxfp8_gemm::config<6, 16, 16, 4>>(N, N, N, ncu);
    N = 16384;
    run_benchmark<mxfp8_gemm::config<5, 8, 8, 4>>(N, N, N, ncu);

    return 0;
}

#else

#include "pyutils/torchutils.cuh"

void mxfp8_gemm_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    using C = mxfp8_gemm::config<6, 16, 12, 4>;
    using G = mxfp8_gemm::globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D)
    };
    kittens::py::launch_kernel<C, G, mxfp8_gemm::kernel<C>>(g);
}

void mxfp8_quantize_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &A_fp8,
    at::Tensor &A_sc
) {
    using C = mxfp8_quantize::config;
    using G = mxfp8_quantize::globals;

    G g {
        .A_bf16 = kittens::py::tensor_to_gl<G::A_bf16_gl>(A_bf16),
        .A_fp8 = kittens::py::tensor_to_gl<G::A_fp8_gl>(A_fp8),
        .A_sc = kittens::py::tensor_to_gl<G::A_sc_gl>(A_sc)
    };
    kittens::py::launch_kernel<C, G, mxfp8_quantize::kernel>(g);
}

PYBIND11_MODULE(_C, m) {
    m.def("mxfp8_gemm", &mxfp8_gemm_entrypoint);
    m.def("mxfp8_quantize", &mxfp8_quantize_entrypoint);
}

#endif
