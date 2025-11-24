/*
    Benchmarks:
        - 4096x4096x4096 : 1236.21 TFLOp/s
        - 8192x8192x8192 : 2558.32 TFLOp/s
        - 16384x16384x16384 : 2700.41 TFLOp/s
        - 204800x2048x1536 : 2307.49 TFLOp/s
*/

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include "torch/csrc/utils/pybind.h"

using namespace kittens;

namespace mxfp8_gemm {

struct config {
    static constexpr int CLUSTER_SIZE = 2;

    static constexpr int SM_COUNT = 148;
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
    static constexpr int SCALE_BLOCK = 512; // 4 K=32-blocks per 128 rows per CTA

    using A_fp8_tile = st_fp8e4m3<ROW_BLOCK / 2, REDUCTION_BLOCK>; // CTA distributed
    using A_sc_vec = sv_fp8e8m0<SCALE_BLOCK>;
    using B_fp8_tile = st_fp8e4m3<COL_BLOCK / 2, REDUCTION_BLOCK>; // CTA distributed
    using B_sc_vec = sv_fp8e8m0<SCALE_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK / 2>;            // CTA/WG distributed

    using A_gl = gl<fp8e4m3, 1, 1, -1, -1, A_fp8_tile>;
    using A_sc_gl = gl<fp8e8m0, 1, -1, -1, SCALE_BLOCK, A_sc_vec>;
    using B_gl = gl<fp8e4m3, 1, 1, -1, -1, B_fp8_tile>;
    using B_sc_gl = gl<fp8e8m0, 1, -1, -1, SCALE_BLOCK, B_sc_vec>;
    using C_gl = gl<bf16, 1, 1, -1, -1, C_tile>;

    A_gl A;       // M x K
    A_sc_gl A_sc; // (M // ROW_BLOCK) x (K // COL_BLOCK) x SCALE_BLOCK
    B_gl B;       // N x K
    B_sc_gl B_sc; // (M // ROW_BLOCK) x (K // COL_BLOCK) x SCALE_BLOCK
    C_gl C;       // M x N

    __host__ inline dim3 grid() const { return dim3(config::SM_COUNT); }
    __host__ inline dim3 block() const { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const { return config::DYNAMIC_SHARED_MEMORY; }
};

struct pipeline_input_tiles {
    globals::A_fp8_tile A;
    globals::B_fp8_tile B;
};

struct pipeline_input_scales {
    globals::A_sc_vec A;
    globals::B_sc_vec B[2];
};

struct pipeline_outputs {
    globals::C_tile C;
};

__device__ inline void kernel(const globals &G) {
    // Warpgroup configuration
    int lane_id = warp::laneid();
    int warp_id = warpgroup::warpid();
    int warpgroup_id = warpgroup::groupid();
    int cta_id = cluster_ctarank();
    int cluster_id = clusterIdx().x;

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    static_assert(sizeof(pipeline_input_tiles) * globals::PIPELINE_STAGES +
                  sizeof(pipeline_input_scales) * globals::PIPELINE_STAGES + 1024 +
                  sizeof(pipeline_outputs) <= config::DYNAMIC_SHARED_MEMORY);
    pipeline_input_tiles (&input_tiles)[globals::PIPELINE_STAGES] = sm_allocator.allocate<pipeline_input_tiles, globals::PIPELINE_STAGES>();
    pipeline_input_scales (&input_scales)[globals::PIPELINE_STAGES] = sm_allocator.allocate<pipeline_input_scales, globals::PIPELINE_STAGES>();
    pipeline_outputs &output_tiles = sm_allocator.allocate<pipeline_outputs>();

    // Allocate tensor memory
    tensor_allocator<1, config::CLUSTER_SIZE> tm_allocator;
    uint32_t out_tm_addr = tm_allocator.get_addr(0);    // columns 000-255
    uint32_t A_sc_tm_addr = tm_allocator.get_addr(256); // columns 256-383
    uint32_t B_sc_tm_addr = tm_allocator.get_addr(384); // columns 384-511

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
            init_semaphore(inputs_arrived[i], 0, config::CLUSTER_SIZE);
            init_semaphore(scales_sm_arrived[i], 0, config::CLUSTER_SIZE);
            init_semaphore(scales_tm_arrived[i], 0, 1); // odd CTA
            init_semaphore(matmul_finished[i], 0, 1); // odd CTA
        }
        init_semaphore(tensor_finished, 0, config::CLUSTER_SIZE);
        init_semaphore(outputs_arrived, 0, 1); // local
    }
    everyone::tma::cluster::sync();

    // Pipeline configuration
    int num_blocks_per_row = G.C.cols() / globals::COL_BLOCK;
    int num_blocks_per_col = G.C.rows() / globals::ROW_BLOCK;
    int num_blocks = num_blocks_per_row * num_blocks_per_col;
    int num_iters_per_block = G.A.cols() / globals::REDUCTION_BLOCK;
    int num_blocks_per_supergroup = globals::SUPERGROUP_BLOCKS * num_blocks_per_row;

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

                    tma::cluster::expect_bytes(inputs_arrived[stage], sizeof(globals::A_fp8_tile) + sizeof(globals::B_fp8_tile), 0);
                    tma::cluster::load_async(input_tiles[stage].A, G.A, {row_block_idx * 2 + cta_id, i}, inputs_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B, G.B, {col_block_idx * 2 + cta_id, i}, inputs_arrived[stage], (uint16_t)(1 << cta_id), 0);

                    if (i == num_iters_per_block - 1) {
                        last_stage = stage;
                    }

                    // Update stage
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            } else if (warp_id == 2 && lane_id == 0) {
                // Load scale matrices to shared memory
                for (int i = 0; i < num_iters_per_block; ++i) {
                    tma::cluster::wait(scales_tm_arrived[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    tma::cluster::expect_bytes(scales_sm_arrived[stage], sizeof(globals::A_sc_vec) + sizeof(globals::B_sc_vec) * 2, 0);
                    tma::cluster::load_async(input_scales[stage].A, G.A_sc, {row_block_idx * 2 + cta_id, i, 0}, scales_sm_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    tma::cluster::load_async(input_scales[stage].B[0], G.B_sc, {col_block_idx * 2 + 0, i, 0}, scales_sm_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    tma::cluster::load_async(input_scales[stage].B[1], G.B_sc, {col_block_idx * 2 + 1, i, 0}, scales_sm_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            } else if (cta_id == 0 && warp_id == 1 && lane_id == 0) {
                // Load scale matrices to tensor memory
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::cluster::wait(scales_sm_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    tma::cluster::wait(matmul_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    uint64_t A_sc_desc = detail::matrix_descriptor_raw(&input_scales[stage].A, 128, 128, 0);
                    uint64_t B_sc_desc[2] = {detail::matrix_descriptor_raw(&input_scales[stage].B[0], 128, 128, 0),
                                             detail::matrix_descriptor_raw(&input_scales[stage].B[1], 128, 128, 0)};

                    asm volatile("{tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;}"
                        :: "r"(A_sc_tm_addr + stage * 16), "l"(A_sc_desc));
                    asm volatile("{tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;}"
                        :: "r"(B_sc_tm_addr + stage * 16 + 4 * 0), "l"(B_sc_desc[0]));
                    asm volatile("{tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;}"
                        :: "r"(B_sc_tm_addr + stage * 16 + 4 * 1), "l"(B_sc_desc[1]));
                    detail::tcgen05::commit<config::CLUSTER_SIZE>(scales_tm_arrived[stage]);
                    asm volatile("{tcgen05.fence::before_thread_sync;}");
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            } else if (cta_id == 0 && warp_id == 0 && lane_id == 0) {
                // Launch tensor core matrix multiply
                tma::cluster::wait(tensor_finished, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    tma::cluster::wait(scales_tm_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);

                    constexpr uint64_t M = globals::ROW_BLOCK;
                    constexpr uint64_t N = globals::COL_BLOCK;
                    constexpr uint64_t K = globals::REDUCTION_BLOCK;
                    constexpr uint32_t I_descs[4] = {
                        detail::tcgen05::instruction_descriptor<bf16, fp8e4m3, fp8e8m0, M, N, 0, 0>(),
                        detail::tcgen05::instruction_descriptor<bf16, fp8e4m3, fp8e8m0, M, N, 0, 1>(),
                        detail::tcgen05::instruction_descriptor<bf16, fp8e4m3, fp8e8m0, M, N, 0, 2>(),
                        detail::tcgen05::instruction_descriptor<bf16, fp8e4m3, fp8e8m0, M, N, 0, 3>()
                    };
                    st_descriptor<globals::A_fp8_tile, 0> A_desc(input_tiles[stage].A);
                    st_descriptor<globals::B_fp8_tile, 0> B_desc(input_tiles[stage].B);

                    asm volatile("{tcgen05.fence::after_thread_sync;}");
                    asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory");
                    asm volatile("{.reg .pred P1; \t\n"
                                    "setp.eq.u32 P1, 1, %6; \t\n"
                                    "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %5, [%3], [%4], P1; \t\n}"
                                    :: "r"(out_tm_addr), 
                                    "l"(A_desc.chunk_descriptor(0)),
                                    "l"(B_desc.chunk_descriptor(0)),
                                    "r"(A_sc_tm_addr + stage * 16),
                                    "r"(B_sc_tm_addr + stage * 16),
                                    "r"(I_descs[0]),
                                    "r"(i == 0 ? 0 : 1));
                    #pragma unroll
                    for (int i = 1; i < K / 32; i++) {
                        asm volatile("{.reg .pred P1; \t\n"
                                        "setp.eq.u32 P1, 1, %6; \t\n"
                                        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %5, [%3], [%4], P1; \t\n}"
                                        :: "r"(out_tm_addr), 
                                        "l"(A_desc.chunk_descriptor(i)),
                                        "l"(B_desc.chunk_descriptor(i)),
                                        "r"(A_sc_tm_addr + stage * 16),
                                        "r"(B_sc_tm_addr + stage * 16),
                                        "r"(I_descs[i]),
                                        "n"(1));
                    }
                    detail::tcgen05::commit<config::CLUSTER_SIZE>(matmul_finished[stage]);
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
            tt<float, globals::ROW_BLOCK / 2, globals::COL_BLOCK> tm(out_tm_addr);
            rt_bf<globals::ROW_BLOCK / 8, globals::COL_BLOCK / 2> C_reg;
            warpgroup::load_async(C_reg, tm.subtile<tt<float, globals::ROW_BLOCK / 2, globals::COL_BLOCK / 2>>(0, warpgroup::groupid() * globals::COL_BLOCK / 2));
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

} // namespace mxfp8_quantize

PYBIND11_MODULE(_C, m) {
    m.def("mxfp8_gemm", &mxfp8_gemm::entrypoint);
}
