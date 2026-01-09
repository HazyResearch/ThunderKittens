#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace nvfp4_gemm {

struct config {
    static constexpr int CLUSTER_SIZE = 2;

    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 256;
    static constexpr int CONSUMER_REGISTERS = 256;

    static constexpr int LOAD_PIPE_DEPTH = 4;
    static constexpr int EPI_PIPE_DEPTH = 8;

    static constexpr int SUPERGROUP_BLOCKS = 1;
    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 256;
    static constexpr int MMA_PER_TILE = RED_BLOCK/64;

    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1;
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::ROW_BLOCK/2, C::RED_BLOCK/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::COL_BLOCK/2, C::RED_BLOCK/2>;
    using B_sc_tile    = st_hf<4, 256, false>;
    using D_tile       = st_bf<C::ROW_BLOCK/2, C::COL_BLOCK/C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,       1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using D_gl           = gl<bf16,       1,  1, -1, -1, D_tile>;

    A_fp4x2_gl     A;           // M x (N // 2)
    A_sc_gl        A_sc;        // (M // 128) x (N // 64) x 256
    A_sc_global_gl A_sc_global; // (1,)
    B_fp4x2_gl     B;           // M x (N // 2)
    B_sc_gl        B_sc;        // (M // 128) x (N // 64) x 256
    B_sc_global_gl B_sc_global; // (1,)
    D_gl           D;           // M x N
};

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    struct input_tiles_t {
        typename G::A_fp4x2_tile A;
        typename G::B_fp4x2_tile B;
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
    static_assert(sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH +
                  sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                  sizeof(outputs_t) <= C::DYNAMIC_SHARED_MEMORY);
    input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<input_tiles_t, C::LOAD_PIPE_DEPTH>();
    input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<input_scales_t, C::LOAD_PIPE_DEPTH>();
    outputs_t       &output_tiles                      = sm_allocator.allocate<outputs_t>();

    // Allocate tensor memory
    tensor_allocator<1, C::CLUSTER_SIZE> tm_allocator;
    auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::COL_BLOCK>>(0);
    auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
    auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(384);

    // Set up mbarriers
    __shared__ semaphore inputs_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 32) {
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 2);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::sync();

    // Thread metadata
    int lane_id = warp::laneid();
    int warp_id = warpgroup::warpid();
    int warpgroup_id = warpgroup::groupid();
    int cta_id = cluster_ctarank();
    int cluster_id = clusterIdx().x;

    // Block dimensions
    const int num_row_blocks = g.D.rows() / C::ROW_BLOCK;
    const int num_col_blocks = g.D.cols() / C::COL_BLOCK;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_red_blocks = 2 * g.A.cols() / C::RED_BLOCK;
    const int num_blocks_per_supergroup = C::SUPERGROUP_BLOCKS * num_col_blocks;

    // Declare stage and phasebits for semaphore waits
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Main divergence
    if (warpgroup_id == C::NUM_WARPGROUPS - 1) {
        // Producer group
        warpgroup::increase_registers<C::PRODUCER_REGISTERS>();

        if (warp_id == 3 && lane_id == 0) {
            // Load input matrices to shared memory
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_BLOCKS, num_row_blocks - supergroup_idx * C::SUPERGROUP_BLOCKS);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_BLOCKS + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                for (int i = 0; i < num_red_blocks; ++i) {
                    tma::cluster::wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    tma::cluster::load_async(input_tiles[stage].A, g.A, {row_block_idx*2 + cta_id, i}, inputs_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx*2 + cta_id, i}, inputs_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    tma::cluster::load_async(input_scales[stage].A, g.A_sc, {row_block_idx*2 + cta_id, i, 0}, inputs_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    tma::cluster::load_async(input_scales[stage].B[cta_id], g.B_sc, {col_block_idx*2 + cta_id, i, 0}, inputs_arrived[stage], (uint16_t)(0b11), 0);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 1 && lane_id == 0) {
            // Load A scales from shared memory to tensor memory
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                #pragma unroll 4
                for (int i = 0; i < num_red_blocks; i++) {
                    tma::cluster::expect_bytes(inputs_arrived[stage], 2 * (sizeof(input_tiles_t) + sizeof(input_scales_t)));
                    tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ii++) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*16 + ii*16 +  0);
                        auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16*32*ii);
                        load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);
                    }
                    kittens::detail::tcgen05::commit<2>(scales_arrived[stage], 0b1);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 2 && lane_id == 0) {
            // Load B scales from shared memory to tensor memory
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                #pragma unroll 4
                for (int i = 0; i < num_red_blocks; i++) {
                    tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ii++) {
                        auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32 + ii*32 +  0);
                        auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32 + ii*32 + 16);
                        auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16*32*ii);
                        auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0]) + 16*32*ii);
                        load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);
                        load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                    }
                    kittens::detail::tcgen05::commit<2>(scales_arrived[stage], 0b1);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0 && lane_id == 0) {
            // Launch tensor core matrix multiplies
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                tma::cluster::wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                update_phasebit<1>(phasebits, 0);
                for (int i = 0; i < num_red_blocks; i++) {
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    if (i == 0) mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                        inputs_finished[stage]);
                    else       mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                        inputs_finished[stage]);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                kittens::detail::tcgen05::commit<2>(outputs_arrived);
            }
        }
    } else {
        // Consumer group
        warpgroup::increase_registers<C::CONSUMER_REGISTERS>();
        const bf16 global_scale_bf16 = __float2bfloat16(g.A_sc_global[{0}] * g.B_sc_global[{0}]);
        const bf16_2 global_scale = {global_scale_bf16, global_scale_bf16};

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_BLOCKS, num_row_blocks - supergroup_idx * C::SUPERGROUP_BLOCKS);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_BLOCKS + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            // Wait for the last matmul to complete
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);

            // Load the output from tensor memory into registers
            rt_bf<C::ROW_BLOCK / 8, C::COL_BLOCK/C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; i++)
                warpgroup::load_async(D_reg[i], out_tm.template subtile<full_tt_fl<C::COL_BLOCK/C::EPI_PIPE_DEPTH>>(0, C::COL_BLOCK/C::EPI_PIPE_DEPTH*i));
            tensor_load_wait();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1); // signal CTA 0

            // Decode with global scale and save to HBM (interleaved)
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                // Scale this tile
                #pragma unroll
                for (int ii = 0; ii < D_reg[i].height; ii++) {
                    #pragma unroll
                    for (int jj = 0; jj < D_reg[i].width; jj++) {
                        #pragma unroll
                        for (int kk = 0; kk < D_reg[i].packed_per_tile; kk++) {
                            D_reg[i].tiles[ii][jj].data[kk] = __hmul2(D_reg[i].tiles[ii][jj].data[kk], global_scale);
                        }
                    }
                }

                // Store this tile
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg[i]);
                warpgroup::sync(1);
                warpgroup::tma::store_async(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + i});
            }
        }
    }
}

void entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    using C = config;
    using G = globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.size(0), B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D)
    };
    kittens::py::launch_kernel<config, G, kernel<config>>(g);
}

} // namespace nvfp4_gemm

namespace nvfp4_quantize {

struct absmax_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 148 * 4;
    static constexpr int NUM_WARPGROUPS = 4;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct quantize_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = 2;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int TILE_M = 128;   // This should not change
    static constexpr int TILE_N = 64;   // This should not change
    static constexpr int K_BLOCK_SIZE = 16; // This should not change

    using A_bf16_tile  = st_bf<TILE_M, TILE_N, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<TILE_M, TILE_N/2, false>;
    using A_sc_vec     = sv_hf<256>;

    using A_bf16_gl      = gl<bf16,      1,  1, -1, -1, A_bf16_tile>;
    using A_fp4x2_gl     = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,      1, -1, -1, 256, A_sc_vec>;
    using A_sc_global_gl = gl<float,     1,  1,  1,  1>;

    A_bf16_gl      A_bf16;      // M x N
    A_fp4x2_gl     A_fp4x2;     // M x (N // 2)
    A_sc_gl        A_sc;        // (M // 128) x (N // 64) x 32 x 16
    A_sc_global_gl A_sc_global; // (1,)

    __host__ inline dim3 grid() const {
        return dim3(A_bf16.cols() / TILE_N, A_bf16.rows() / TILE_M);
    }
    __host__ inline int dynamic_shared_memory() const {
        return TILE_M * TILE_N * sizeof(bf16) + 1024;
    }
};

__global__ void zero_kernel(const globals g) {
    g.A_sc_global.raw_ptr[0] = 0.0f;
}

__global__ void absmax_kernel(const globals g) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = gridDim.x * blockDim.x;
    const size_t numel = g.A_bf16.rows() * g.A_bf16.cols();

    bf16 local_max = __float2bfloat16(0.0f);
    bf16_2 *base_ptr = reinterpret_cast<bf16_2*>(g.A_bf16.raw_ptr);

    for (size_t i = tid; i < numel / 8; i += num_threads) {
        bf16_2 v0, v1, v2, v3;
        asm volatile(
            "ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
            : "=r"(*(uint32_t*)&v0), "=r"(*(uint32_t*)&v1), "=r"(*(uint32_t*)&v2), "=r"(*(uint32_t*)&v3)
            : "l"(base_ptr + i*4)
        );

        bf16_2 abs0 = __habs2(v0);
        bf16_2 abs1 = __habs2(v1);
        bf16_2 abs2 = __habs2(v2);
        bf16_2 abs3 = __habs2(v3);

        bf16_2 max01 = __hmax2(abs0, abs1);
        bf16_2 max23 = __hmax2(abs2, abs3);
        bf16_2 max0123 = __hmax2(max01, max23);

        bf16 curr_max = __hmax(max0123.x, max0123.y);
        local_max = __hmax(local_max, curr_max);
    }

    for (size_t i = (numel / 8) * 8 + tid; i < numel; i += num_threads)
        local_max = __hmax(local_max, __habs(g.A_bf16.raw_ptr[i]));

    #pragma unroll
    for (int offset = WARP_THREADS / 2; offset > 0; offset /= 2) {
        uint32_t local_bits = *reinterpret_cast<unsigned short*>(&local_max);
        uint32_t other_bits = __shfl_xor_sync(0xffffffff, local_bits, offset);
        local_max = __hmax(local_max, *reinterpret_cast<bf16*>(&other_bits));
    }

    __shared__ bf16 shared_max[absmax_config::NUM_WARPS];
    if (laneid() == 0) shared_max[warpid()] = local_max;
    __syncthreads();

    if (warpid() == 0) {
        bf16 val = (laneid() < absmax_config::NUM_WARPS) ? shared_max[laneid()] : __float2bfloat16(0.0f);

        #pragma unroll
        for (int offset = absmax_config::NUM_WARPS / 2; offset > 0; offset /= 2) {
            uint32_t val_bits = *reinterpret_cast<unsigned short*>(&val);
            uint32_t other_bits = __shfl_xor_sync(0xffffffff, val_bits, offset);
            val = __hmax(val, *reinterpret_cast<bf16*>(&other_bits));
        }

        if (laneid() == 0) {
            float val_fl = __bfloat162float(val); // Positive float values keep bit ordering
            atomicMax(reinterpret_cast<uint32_t*>(g.A_sc_global.raw_ptr), *reinterpret_cast<uint32_t*>(&val_fl));
        }
    }
}

__global__ void divide_kernel(const globals g) {
    g.A_sc_global.raw_ptr[0] = g.A_sc_global.raw_ptr[0] / (6.0f * 448.0f);
}

__device__ inline void quantize_kernel(const globals &G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    globals::A_bf16_tile &A_bf16_smem = sm_allocator.allocate<globals::A_bf16_tile>();
    globals::A_fp4x2_tile &A_fp4x2_smem = *reinterpret_cast<globals::A_fp4x2_tile *>(&A_bf16_smem);
    globals::A_sc_vec &A_sc_smem = *reinterpret_cast<globals::A_sc_vec *>(
        reinterpret_cast<uint64_t>(&A_fp4x2_smem) + sizeof(A_fp4x2_smem));

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

    // Fetch pre-calculated global scales
    float s_global_dec = G.A_sc_global[{0}];
    float s_global_enc = 1.0f / s_global_dec;

    // We have 64 threads per block. Each thread handles 2 rows of 64 elements / 16 elements per block = 8 K blocks
    constexpr int ROWS_PER_THREAD = 2;
    constexpr int NUM_K_BLOCKS = globals::TILE_N / globals::K_BLOCK_SIZE; // 4 (per row)
    constexpr int N_PER_K_BLOCK = globals::K_BLOCK_SIZE / 2;              // 8 (bf16x2 per K block)
    bf16_2 A_bf16_reg[ROWS_PER_THREAD][NUM_K_BLOCKS][N_PER_K_BLOCK];
    fp8e4m3 A_sc_reg[ROWS_PER_THREAD][NUM_K_BLOCKS];

    // Wait for the inputs to arrive
    __syncthreads();
    wait(inputs_arrived, 0);

    // Load input matrix from shared memory (custom swizzling to avoid bank conflicts)
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        const int tile_row = tid + r*64;
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS; i++) {
            const int k_block_idx = (i + tid/8) % NUM_K_BLOCKS; // each block takes 8 SMEM banks
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int tile_col = k_block_idx*globals::K_BLOCK_SIZE + ((tid+j)*2)%globals::K_BLOCK_SIZE;
                const int offset = (tile_row*globals::TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(A_bf16_reg[r][i][j], static_cast<uint32_t>(__cvta_generic_to_shared(&A_bf16_smem)) + offset);
            }
        }
    }
    __syncthreads();

    // Perform NVFP4 quantization
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        const int tile_row = tid + r*64;
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS; i++) {
            const int k_block_idx = (i + tid/8) % NUM_K_BLOCKS;

            // Calculate absolute maximum for this K block
            bf16_2 amax = __habs2(A_bf16_reg[r][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; j++)
                amax = __hmax2(amax, __habs2(A_bf16_reg[r][i][j]));

            // Compute the local scale
            float s_local_enc = 6.0f / (s_global_enc * __bfloat162float(__hmax(amax.x, amax.y)));
            float s_local_dec = 1.0f / s_local_enc;
            A_sc_reg[r][k_block_idx] = __nv_fp8_e4m3(s_local_dec); // round-to-even

            // Quantize input matrix to FP4 and store to shared memory
            const int offset_base = tile_row*globals::TILE_N/2 + k_block_idx*globals::K_BLOCK_SIZE/2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int offset = offset_base + ((tid+j)&7);
                const float2 scaled = {
                    __bfloat162float(A_bf16_reg[r][i][j].x)*s_global_enc*s_local_enc,
                    __bfloat162float(A_bf16_reg[r][i][j].y)*s_global_enc*s_local_enc
                };
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_fp4x2_smem)) + offset)
                       "r"(static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest))));
            }
        }

        // Store the scales to shared memory following NVIDIA's scale swizzle layout
        const int scale_offset = (tile_row%32) * 16 + (tile_row/32) * 4;

        // Store 4 scales (one per K block)
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_sc_smem)) + scale_offset)
               "r"(*reinterpret_cast<uint32_t *>(&A_sc_reg[r][0])));
    }

    // Store to global memory
    __syncthreads();
    if (tid == 0) {
        tma::store_async(G.A_fp4x2, A_fp4x2_smem, {row, col});
        tma::store_async(G.A_sc,    A_sc_smem,    {row, col, 0});
    }
}

__host__ void entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &A_fp4x2,
    at::Tensor &A_sc,
    at::Tensor &A_sc_global
) {
    globals G {
        .A_bf16 = kittens::py::tensor_to_gl<globals::A_bf16_gl>(A_bf16),
        .A_fp4x2 = kittens::py::tensor_to_gl<globals::A_fp4x2_gl>(A_fp4x2),
        .A_sc = kittens::py::tensor_to_gl<globals::A_sc_gl, false>(A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<globals::A_sc_global_gl>(A_sc_global)
    };

    zero_kernel<<<1, 1>>>(G);
    absmax_kernel<<<absmax_config::NUM_BLOCKS, absmax_config::NUM_THREADS>>>(G);
    divide_kernel<<<1, 1>>>(G);
    kittens::py::launch_kernel<quantize_config, globals, quantize_kernel>(G);
}

} // namespace nvfp4_quantize

#include "ATen/Functions.h"

namespace nvfp4_utils {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1024; // arbitrary
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    using A_fp32_gl = gl<float, 1, 1, -1, -1>;
    using A_fp4x2_gl = gl<fp4e2m1_2, 1, 1, -1, -1>;

    A_fp32_gl A_fp32;
    A_fp4x2_gl A_fp4x2;
};

__device__ inline void fp32_to_fp4x2_kernel(const globals &G) {
    // This kernel is for testing purposes only
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < G.A_fp32.numel() / 2; i += blockDim.x * gridDim.x) {
        float2 A_fp32x2 = {G.A_fp32.raw_ptr[i * 2 + 0], G.A_fp32.raw_ptr[i * 2 + 1]};
        G.A_fp4x2.raw_ptr[i].__x = __nv_cvt_float2_to_fp4x2(A_fp32x2, __NV_E2M1, cudaRoundNearest);
    }
}

__device__ inline void fp4x2_to_fp32_kernel(const globals &G) {
    // This kernel is for testing purposes only
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < G.A_fp32.numel() / 2; i += blockDim.x * gridDim.x) {
        float2 A_fp32x2 = static_cast<float2>(G.A_fp4x2.raw_ptr[i]);
        G.A_fp32.raw_ptr[i * 2 + 0] = A_fp32x2.x;
        G.A_fp32.raw_ptr[i * 2 + 1] = A_fp32x2.y;
    }
}

__host__ at::Tensor fp32_to_fp4x2(at::Tensor A_fp32) {
    auto options = A_fp32.options().dtype(at::kFloat4_e2m1fn_x2).requires_grad(false);
    at::Tensor A_fp4x2 = at::empty({A_fp32.size(0), A_fp32.size(1) / 2}, options);

    globals G {
        .A_fp32 = kittens::py::tensor_to_gl<globals::A_fp32_gl>(A_fp32),
        .A_fp4x2 = kittens::py::tensor_to_gl<globals::A_fp4x2_gl>(A_fp4x2),
    };
    kittens::py::launch_kernel<config, globals, fp32_to_fp4x2_kernel>(G);

    return A_fp4x2;
}

__host__ at::Tensor fp4x2_to_fp32(at::Tensor A_fp4x2) {
    auto options = A_fp4x2.options().dtype(at::kFloat).requires_grad(false);
    at::Tensor A_fp32 = at::empty({A_fp4x2.size(0), A_fp4x2.size(1) * 2}, options);

    globals G {
        .A_fp32 = kittens::py::tensor_to_gl<globals::A_fp32_gl>(A_fp32),
        .A_fp4x2 = kittens::py::tensor_to_gl<globals::A_fp4x2_gl>(A_fp4x2),
    };
    kittens::py::launch_kernel<config, globals, fp4x2_to_fp32_kernel>(G);   

    return A_fp32;
}

}

PYBIND11_MODULE(_C, m) {
    m.def("nvfp4_gemm", &nvfp4_gemm::entrypoint);
    m.def("nvfp4_quantize", &nvfp4_quantize::entrypoint);
    m.def("fp32_to_fp4x2", &nvfp4_utils::fp32_to_fp4x2);
    m.def("fp4x2_to_fp32", &nvfp4_utils::fp4x2_to_fp32);
}
