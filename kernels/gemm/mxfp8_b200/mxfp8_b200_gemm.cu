#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace mxfp8_gemm {

struct config {
    static constexpr int CLUSTER_SIZE = 2;

    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 2;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;
};

struct globals {
    static constexpr int PIPELINE_STAGES = 5;
    static constexpr int SUPERGROUP_BLOCKS = 12;
    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 256;
    static constexpr int REDUCTION_BLOCK = 128;

    using A_fp8_tile = st_fp8e4m3<ROW_BLOCK / 2, REDUCTION_BLOCK>; // CTA distributed
    using A_sc_tile  = st_fp8e8m0<32, 16, false>;
    using B_fp8_tile = st_fp8e4m3<COL_BLOCK / 2, REDUCTION_BLOCK>; // CTA distributed
    using B_sc_tile  = st_fp8e8m0<32, 16, false>;
    using C_tile     = st_bf<ROW_BLOCK / 2, COL_BLOCK / 2>;        // CTA/WG distributed

    using A_gl    = gl<fp8e4m3,  1,  1, -1, -1, A_fp8_tile>;
    using A_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, A_sc_tile>;
    using B_gl    = gl<fp8e4m3,  1,  1, -1, -1, B_fp8_tile>;
    using B_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, B_sc_tile>;
    using C_gl    = gl<bf16,     1,  1, -1, -1, C_tile>;

    A_gl A;       // M x K
    A_sc_gl A_sc; // (M // 128) x (K // 128) x 32 x 16
    B_gl B;       // N x K
    B_sc_gl B_sc; // (M // 128) x (K // 128) x 32 x 16
    C_gl C;       // M x N
};

struct pipeline_input_tiles {
    globals::A_fp8_tile A;
    globals::B_fp8_tile B;
};

struct pipeline_input_scales {
    globals::A_sc_tile A;
    globals::B_sc_tile B[2];
};

struct pipeline_outputs {
    globals::C_tile C;
};

__device__ inline void kernel(const globals &G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    static_assert(sizeof(pipeline_input_tiles) * globals::PIPELINE_STAGES +
                  sizeof(pipeline_input_scales) * globals::PIPELINE_STAGES + 1024 +
                  sizeof(pipeline_outputs) <= config::DYNAMIC_SHARED_MEMORY);
    pipeline_input_tiles  (&input_tiles) [globals::PIPELINE_STAGES] = sm_allocator.allocate<pipeline_input_tiles, globals::PIPELINE_STAGES>();
    pipeline_input_scales (&input_scales)[globals::PIPELINE_STAGES] = sm_allocator.allocate<pipeline_input_scales, globals::PIPELINE_STAGES>();
    pipeline_outputs &output_tiles = sm_allocator.allocate<pipeline_outputs>();

    // Allocate tensor memory
    tensor_allocator<1, config::CLUSTER_SIZE> tm_allocator;
    auto out_tm  = tm_allocator.allocate<full_tt_fl<globals::COL_BLOCK>>(0);                 // columns 000-255
    auto A_sc_tm = tm_allocator.allocate<full_tt_fp8e8m0<16*globals::PIPELINE_STAGES>>(256); // columns 256-383
    auto B_sc_tm = tm_allocator.allocate<full_tt_fp8e8m0<32*globals::PIPELINE_STAGES>>(384); // columns 384-511

    // Set up mbarriers
    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore scales_sm_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore scales_tm_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore matmul_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore tensor_finished;
    __shared__ semaphore outputs_arrived;
    if (threadIdx.x == 32) {
        #pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1); // even CTA
            init_semaphore(scales_sm_arrived[i], 0, 1); // even CTA
            init_semaphore(scales_tm_arrived[i], 0, 1); // even CTA
            init_semaphore(matmul_finished[i], 0, 1); // even CTA
        }
        init_semaphore(tensor_finished, 0, config::CLUSTER_SIZE);
        init_semaphore(outputs_arrived, 0, 1); // local
    }
    everyone::tma::cluster::sync();

    // Warpgroup configuration
    int lane_id = warp::laneid();
    int warp_id = warpgroup::warpid();
    int warpgroup_id = warpgroup::groupid();
    int cta_id = cluster_ctarank();
    int cluster_id = clusterIdx().x;

    // Pipeline configuration
    const int num_blocks_per_row = G.C.cols() / globals::COL_BLOCK;
    const int num_blocks_per_col = G.C.rows() / globals::ROW_BLOCK;
    const int num_blocks = num_blocks_per_row * num_blocks_per_col;
    const int num_iters_per_block = G.A.cols() / globals::REDUCTION_BLOCK;
    const int num_blocks_per_supergroup = globals::SUPERGROUP_BLOCKS * num_blocks_per_row;

    // Declare stage and phasebits for semaphore waits
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    uint32_t last_stage = globals::PIPELINE_STAGES;

    // Main divergence
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        // Producer group
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / config::CLUSTER_SIZE) {
            // Compute block indices
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(globals::SUPERGROUP_BLOCKS, num_blocks_per_col - supergroup_idx * globals::SUPERGROUP_BLOCKS);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * globals::SUPERGROUP_BLOCKS + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            if (warp_id == 3 && lane_id == 0) {
                // Load input matrices to shared memory
                for (int i = 0; i < num_iters_per_block; ++i) {
                    tma::cluster::wait(matmul_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);

                    if (stage == last_stage) {
                        arrive(outputs_arrived);
                        last_stage = globals::PIPELINE_STAGES;
                    }

                    tma::cluster::load_async(input_tiles[stage].A, G.A, {row_block_idx * 2 + cta_id, i}, inputs_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B, G.B, {col_block_idx * 2 + cta_id, i}, inputs_arrived[stage], (uint16_t)(1 << cta_id), 0);

                    if (i == num_iters_per_block - 1) {
                        last_stage = stage;
                    }

                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            } else if (warp_id == 2 && lane_id == 0) {
                // Load scale matrices to shared memory
                for (int i = 0; i < num_iters_per_block; ++i) {
                    tma::cluster::wait(scales_tm_arrived[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    tma::cluster::load_async(input_scales[stage].A,    G.A_sc, {row_block_idx * 2 + cta_id, i, 0, 0}, scales_sm_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    tma::cluster::load_async(input_scales[stage].B[0], G.B_sc, {col_block_idx * 2 + 0,      i, 0, 0}, scales_sm_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    tma::cluster::load_async(input_scales[stage].B[1], G.B_sc, {col_block_idx * 2 + 1,      i, 0, 0}, scales_sm_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            } else if (cta_id == 0 && warp_id == 1 && lane_id == 0) {
                // Load scale matrices to tensor memory
                #pragma unroll 2
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::cluster::expect_bytes(scales_sm_arrived[stage], sizeof(globals::A_sc_tile) * 2 + sizeof(globals::B_sc_tile) * 4);
                    tma::cluster::wait(scales_sm_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    tma::cluster::wait(matmul_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);

                    auto A_sc_tm_subtile   = A_sc_tm.subtile<full_tt_fp8e8m0<16>>(stage * 16);
                    auto B_sc_tm_subtile_0 = B_sc_tm.subtile<full_tt_fp8e8m0<16>>(stage * 32);
                    auto B_sc_tm_subtile_1 = B_sc_tm.subtile<full_tt_fp8e8m0<16>>(stage * 32 + 16);
                    load_mxnv_scale_async2(A_sc_tm_subtile,   input_scales[stage].A);
                    load_mxnv_scale_async2(B_sc_tm_subtile_0, input_scales[stage].B[0]);
                    load_mxnv_scale_async2(B_sc_tm_subtile_1, input_scales[stage].B[1], scales_tm_arrived[stage]);

                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            } else if (cta_id == 0 && warp_id == 0 && lane_id == 0) {
                // Launch tensor core matrix multiply
                tma::cluster::wait(tensor_finished, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);
                #pragma unroll 8
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::cluster::expect_bytes(inputs_arrived[stage], (sizeof(globals::A_fp8_tile) + sizeof(globals::B_fp8_tile)) * 2);
                    tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    tma::cluster::wait(scales_tm_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    if (i == 0) mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.subtile<full_tt_fp8e8m0<16>>(stage * 16), 
                                        B_sc_tm.subtile<full_tt_fp8e8m0<32>>(stage * 32),
                                        matmul_finished[stage]);
                    else mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                  A_sc_tm.subtile<full_tt_fp8e8m0<16>>(stage * 16), 
                                  B_sc_tm.subtile<full_tt_fp8e8m0<32>>(stage * 32),
                                  matmul_finished[stage]);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        }
        if (warp_id == 3 && lane_id == 0 && last_stage < globals::PIPELINE_STAGES) {
            tma::cluster::wait(matmul_finished[last_stage], get_phasebit<1>(phasebits, last_stage));
            arrive(outputs_arrived);
        }
    } else {
        // Consumer group
        using consumer = group<config::CONSUMER_WARPGROUPS * WARPGROUP_WARPS>;
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / config::CLUSTER_SIZE) {
            // Compute block indices
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(globals::SUPERGROUP_BLOCKS, num_blocks_per_col - supergroup_idx * globals::SUPERGROUP_BLOCKS);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * globals::SUPERGROUP_BLOCKS + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            // Wait for the last matmul to complete
            wait(outputs_arrived, get_phasebit<0>(phasebits, globals::PIPELINE_STAGES));
            update_phasebit<0>(phasebits, globals::PIPELINE_STAGES);

            // Load the output from tensor memory into registers
            rt_bf<globals::ROW_BLOCK / 8, globals::COL_BLOCK / 2> C_reg;
            warpgroup::load_async(C_reg, out_tm.subtile<tt_fl<globals::ROW_BLOCK / 2, globals::COL_BLOCK / 2>>(0, warpgroup::groupid() * globals::COL_BLOCK / 2));
            tensor_load_wait();
            consumer::sync(1);
            if (consumer::laneid() == 0)
                tma::cluster::arrive(tensor_finished, 0, 1); // signal CTA 0

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                if (warpgroup::groupid() == i) {
                    warpgroup::store(output_tiles.C, C_reg);
                    warpgroup::sync(2 + i);
                    if (warpgroup::laneid() == 0) {
                        tma::store_async(G.C, output_tiles.C, {row_block_idx * 2 + cta_id, col_block_idx * 2 + i});
                        tma::store_async_read_wait();
                    }
                }
                consumer::sync(1);
            }
        }
    }
}

void entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &C
) {
    globals G {
        .A = kittens::py::tensor_to_gl<globals::A_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<globals::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<globals::B_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<globals::B_sc_gl>(B_sc),
        .C = kittens::py::tensor_to_gl<globals::C_gl>(C)
    };
    kittens::py::launch_kernel<config, globals, kernel>(G);
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

__host__ void entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &A_fp8,
    at::Tensor &A_sc
) {
    globals G {
        .A_bf16 = kittens::py::tensor_to_gl<globals::A_bf16_gl>(A_bf16),
        .A_fp8 = kittens::py::tensor_to_gl<globals::A_fp8_gl>(A_fp8),
        .A_sc = kittens::py::tensor_to_gl<globals::A_sc_gl>(A_sc)
    };
    kittens::py::launch_kernel<config, globals, kernel>(G);
}

} // namespace mxfp8_quantize

PYBIND11_MODULE(_C, m) {
    m.def("mxfp8_gemm", &mxfp8_gemm::entrypoint);
    m.def("mxfp8_quantize", &mxfp8_quantize::entrypoint);
}
