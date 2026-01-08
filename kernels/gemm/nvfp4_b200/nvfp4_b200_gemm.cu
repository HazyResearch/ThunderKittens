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
    static constexpr int EPI_PIPE_DEPTH = 4;

    static constexpr int SUPERGROUP_BLOCKS = 2;
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
    __shared__ semaphore scales_arrived;
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 32) {
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(scales_arrived, 0, 1);
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

    // Block swizzling
    int num_row_blocks = g.D.rows() / C::ROW_BLOCK;
    int num_col_blocks = g.D.cols() / C::COL_BLOCK;
    int num_blocks = num_row_blocks * num_col_blocks;
    int num_red_blocks = 2 * g.A.cols() / C::RED_BLOCK;
    int num_blocks_per_supergroup = C::SUPERGROUP_BLOCKS * num_col_blocks;

    // Declare stage and phasebits for semaphore waits
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Main divergence
    if (warpgroup_id == C::NUM_WARPGROUPS - 1) {
        // Producer group
        warpgroup::increase_registers<C::PRODUCER_REGISTERS>();

        if (warp_id == 3 && lane_id == 0) {
            // Load input matrices to shared memory
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / config::CLUSTER_SIZE) {
                // Compute block indices
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
        } else if (cta_id == 0 && warp_id == 0 && lane_id == 0) {
            // Launch tensor core matrix multiplies
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / config::CLUSTER_SIZE) {
                tma::cluster::wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                update_phasebit<1>(phasebits, 0);
                for (int i = 0; i < num_red_blocks; i++) {
                    tma::cluster::expect_bytes(inputs_arrived[stage], 2 * (sizeof(input_tiles_t) + sizeof(input_scales_t)));
                    tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ii++) {
                        auto A_sc_tm_subtile    = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*16 + ii*16 +  0);
                        auto B_sc_tm_subtile_0  = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32 + ii*32 +  0);
                        auto B_sc_tm_subtile_1  = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32 + ii*32 + 16);
                        auto &A_sc_sm_subtile   = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0])    + 16*32*ii);
                        auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16*32*ii);
                        auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0]) + 16*32*ii);
                        load_mxnv_scale_async2(A_sc_tm_subtile,   A_sc_sm_subtile);
                        load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);
                        load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                    }

                    // It appears that tcgen05.cp and tcgen05.mma are implicitly pipelined. In case a race condition is found, try uncommenting these lines.
                    // kittens::detail::tcgen05::commit<2>(scales_arrived, 0b1);
                    // wait(scales_arrived, get_phasebit<0>(phasebits, C::LOAD_PIPE_DEPTH));
                    // update_phasebit<0>(phasebits, C::LOAD_PIPE_DEPTH);

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
        const float2 global_scale = {g.A_sc_global[{0}] * g.B_sc_global[{0}],
                                     g.A_sc_global[{0}] * g.B_sc_global[{0}]};

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / config::CLUSTER_SIZE) {
            // Compute block indices
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

            // Decode with global scale
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                #pragma unroll
                for (int ii = 0; ii < D_reg[i].height; ii++) {
                    #pragma unroll
                    for (int jj = 0; jj < D_reg[i].width; jj++) {
                        #pragma unroll
                        for (int kk = 0; kk < D_reg[i].packed_per_tile; kk++) {
                            D_reg[i].tiles[ii][jj].data[kk] = __float22bfloat162_rn(
                                __fmul2_rd(__bfloat1622float2(D_reg[i].tiles[ii][jj].data[kk]), global_scale));
                        }
                    }
                }
            }

            // Save to HBM
            #pragma unroll
            for(int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
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

#include "ATen/ops/linalg_vector_norm.h" 
#include "ATen/ops/unsqueeze.h"

namespace nvfp4_quantize {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int TILE_M = 128;   // This should not change
    static constexpr int TILE_N = 64;   // This should not change
    static constexpr int K_BLOCK_SIZE = 16; // This should not change

    using A_bf16_tile  = st_bf<TILE_M, TILE_N, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<TILE_M, TILE_N, false>;
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

__device__ inline void kernel(const globals &G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    globals::A_bf16_tile &A_bf16_smem = sm_allocator.allocate<globals::A_bf16_tile>();
    globals::A_fp4x2_tile &A_fp4x2_smem = *reinterpret_cast<globals::A_fp4x2_tile *>(&A_bf16_smem);
    globals::A_sc_tile &A_sc_smem = *reinterpret_cast<globals::A_sc_tile *>(
        reinterpret_cast<uint64_t>(&A_fp4x2_smem) + sizeof(A_fp4x2_smem));

    // TODO: Implement
}

__host__ inline void absmax(const at::Tensor &x, at::Tensor &out) {
    const at::Scalar pos_infty = at::Scalar(std::numeric_limits<double>::infinity());
    auto out_view = at::_ops::squeeze::call(out);
    at::_ops::linalg_vector_norm_out::call(x, pos_infty, c10::nullopt, false, c10::nullopt, out_view);
}

__host__ void entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &A_fp4x2,
    at::Tensor &A_sc,
    at::Tensor &A_sc_global
) {
    absmax(A_bf16, A_sc_global);
    globals G {
        .A_bf16 = kittens::py::tensor_to_gl<globals::A_bf16_gl>(A_bf16),
        .A_fp4x2 = kittens::py::tensor_to_gl<globals::A_fp4x2_gl>(A_fp4x2),
        .A_sc = kittens::py::tensor_to_gl<globals::A_sc_gl>(A_sc),
        .A_sc_global = kittens::py::tensor_to_gl<globals::A_sc_global_gl>(A_sc_global)
    };
    kittens::py::launch_kernel<config, globals, kernel>(G);
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
