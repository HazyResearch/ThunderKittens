#include "kittens.cuh"
#include "../common.cuh"

using namespace kittens;

namespace nvfp4_gemm {

static constexpr int SF_VEC = 16;

template <int _N_PAIR_COUNT, int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE>
struct config {
    static_assert(_N_PAIR_COUNT == 1 || _N_PAIR_COUNT == 2 || _N_PAIR_COUNT == 4, "N_PAIR_COUNT must be 1, 2, or 4");
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 12, "LOAD_PIPE_DEPTH must be greater than 0 and at most 12");
    static_assert(_SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be greater than 0");

    static constexpr int CLUSTER_SIZE = 2;

    static constexpr int N_PAIR_COUNT = _N_PAIR_COUNT;
    static constexpr bool USE_PREFERRED_CLUSTER = N_PAIR_COUNT > 1;
    static constexpr int CONSUMER_WARPGROUPS = N_PAIR_COUNT == 4 ? 1 : 2;
    static constexpr int PRODUCER_WARPS = USE_PREFERRED_CLUSTER ? WARPGROUP_WARPS : 3;
    static constexpr int NUM_WARPS = CONSUMER_WARPGROUPS * WARPGROUP_WARPS + PRODUCER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int M_TILE_COUNT = USE_PREFERRED_CLUSTER ? 2 : 1;
    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int A_PIPE_SLOTS = USE_PREFERRED_CLUSTER ? M_TILE_COUNT * LOAD_PIPE_DEPTH : 0;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr cache_policy A_SC_CACHE_POLICY =
        USE_PREFERRED_CLUSTER ? cache_policy::EVICT_LAST : cache_policy::NORMAL;

    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int EPI_COLS = 32;
    static constexpr int EPI_PIPE_DEPTH = Nb / EPI_COLS;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int FP8_VALUES_PER_TMEM_COL = 4 / sizeof(fp8e4m3);
    static constexpr int ACC_TMEM_COLS = M_TILE_COUNT * Nb;
    static constexpr int SINGLE_PAIR_SCALE_TMEM_COLS = (16 + 32*MMA_PER_TILE) * LOAD_PIPE_DEPTH / FP8_VALUES_PER_TMEM_COL;
    static constexpr int PREFERRED_SCALE_TMEM_COLS = (M_TILE_COUNT*2*16 + 2*64) / FP8_VALUES_PER_TMEM_COL;
    static constexpr int TMEM_COLS = ACC_TMEM_COLS + (USE_PREFERRED_CLUSTER ? PREFERRED_SCALE_TMEM_COLS : SINGLE_PAIR_SCALE_TMEM_COLS);
    static constexpr int TMEM_LIMIT = USE_PREFERRED_CLUSTER ? MAX_TENSOR_COLS_EXCLUSIVE : MAX_TENSOR_COLS;
    static_assert(TMEM_COLS <= TMEM_LIMIT, "Tensor-memory allocation exceeds allocator capacity");
};

template <typename C>
__device__ inline int2 get_job_coords(int block_idx, int num_row_blocks, int num_col_blocks) {
    const int2 tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE, false, C::USE_PREFERRED_CLUSTER>(
        num_row_blocks, num_col_blocks, block_idx / C::N_PAIR_COUNT);
    return {tile_coord.x, tile_coord.y * C::N_PAIR_COUNT + block_idx % C::N_PAIR_COUNT};
}

template <typename C>
struct globals {
    using A_fp4x2_tile  = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tma_tile = st_fp8e4m3<8, 256, false>;
    using B_fp4x2_tile  = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tma_tile = A_sc_tma_tile;
    using A_sc_grp_tile = st_fp8e4m3<128, 16, false>;
    using scale_atom    = st_fp8e4m3<32, 16, false>;
    using Dq_tile       = st_fp4e2m1_2<C::Mb/2, C::Nb/2>;
    using Ds_tile       = st_fp8e4m3<128, 16, false>;

    union scale_group {
        A_sc_tma_tile tma;
        A_sc_grp_tile tile;
        scale_atom atoms[A_sc_grp_tile::rows / scale_atom::rows];
    };
    static_assert(sizeof(scale_group) == sizeof(A_sc_tma_tile));

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<fp8e4m3,    1, -1, -1, 256, A_sc_tma_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<fp8e4m3,    1, -1, -1, 256, B_sc_tma_tile>;
    using B_sc_global_gl = A_sc_global_gl;
    using Dq_gl          = gl<fp4e2m1_2,  1,  1, -1, -1, Dq_tile>;
    using Ds_gl          = gl<fp8e4m3,    1,  1, -1, -1, Ds_tile>;

    A_fp4x2_gl     A;           // M x (K // 2)
    A_sc_gl        A_sc;        // (M // 128) x (K // 32) x 256
    A_sc_global_gl A_sc_global; // (1,)
    B_fp4x2_gl     B;           // N x (K // 2)
    B_sc_gl        B_sc;        // (N // 128) x (K // 32) x 256
    B_sc_global_gl B_sc_global; // (1,)
    A_sc_global_gl D_sc_global; // (1,) FP32 output global scale
    Dq_gl          Dq;          // M x (N // 2)
    Ds_gl          Ds;          // cuBLASLt scale atoms flattened as 128 x 16 byte tiles

    struct ab_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct b_tiles_t {
        B_fp4x2_tile B;
    };
    using input_tiles_t = std::conditional_t<C::USE_PREFERRED_CLUSTER, b_tiles_t, ab_tiles_t>;
    struct input_scales_t {
        scale_group A[C::M_TILE_COUNT];
        scale_group B[C::B_SC_SIZE];
    };
    struct outputs_t {
        Dq_tile Dq[C::M_TILE_COUNT];
        Ds_tile Ds[C::M_TILE_COUNT];
    };

    __host__ inline dim3 grid() const {
        const int num_ctas = (Dq.rows()/(C::M_TILE_COUNT*(C::Mb/2)))*(Dq.cols()/(C::Nb/2));
        return dim3(C::USE_PREFERRED_CLUSTER ? num_ctas : min(num_ctas, num_sms()));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(A_fp4x2_tile)   * C::A_PIPE_SLOTS    +
                                               sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(outputs_t);
        static_assert(_dynamic_shared_memory <= kittens::MAX_SHARED_MEMORY_OVERSIZED - 1024);
        return _dynamic_shared_memory;
    }
};

template <typename TT, typename ST>
__device__ inline void stage_b_scales(TT B_sc_tm, ST &input_scales, int chunk_idx) {
    #pragma unroll
    for (int n = 0; n < std::extent_v<decltype(ST::B)>; n++)
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            auto B_sc_tm_subtile = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>((j*2+n)*16);
            load_mxnv_scale_async2(B_sc_tm_subtile, input_scales.B[n].atoms[chunk_idx*2 + j]);
        }
}


// Byte size of the cuBLAS swizzled output-scale buffer: 128-row by 4-k-block atoms of 512 bytes.
__host__ inline size_t scale_buffer_bytes(size_t M, size_t N) {
    return ((M + 127) / 128 * 128) * ((N + 63) / 64 * 4);
}

// Spec-NVFP4 block scale: the e4m3 RN byte for amax/6. An all-zero block yields 0x00, as cuBLASLt emits.
__device__ inline uint8_t scale_byte(float amax_over_6) {
    return __nv_cvt_float_to_fp8(amax_over_6, __NV_SATFINITE, __NV_E4M3);
}

// fp32 reciprocal of the e4m3 scale; the bf16 round-trip is lossless at three mantissa bits.
__device__ inline float reciprocal_scale(uint8_t scale) {
    uint32_t bf16_hi;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(bf16_hi) : "h"(uint16_t(uint16_t(scale) << 8)));
    // Floored here rather than in scale_byte: a 0x00 byte must still give a finite factor, or the
    // all-zero codes would go through 0 * inf = NaN. A no-op for every representable scale.
    return 1.0f / fmaxf(std::bit_cast<float>(bf16_hi), 0x1p-9f);
}

// Scales and packs 16 accumulator values as eight e2m1x2 bytes
__device__ __forceinline__ uint2 pack_e2m1_x16_scaled(const rv_fl<32*SF_VEC, naive_l> &values,
                                                      float factor) {
    uint2 packed;
    asm volatile(
        "{\n"
        ".reg .b8  byte<8>;\n"
        ".reg .b64 pair<8>;\n"
        ".reg .b64 factor2;\n"
        ".reg .f32 val<16>;\n"
        "mov.b64 factor2, {%18, %18};\n"
        "mov.b64 pair0, {%2, %3};\n"
        "mov.b64 pair1, {%4, %5};\n"
        "mov.b64 pair2, {%6, %7};\n"
        "mov.b64 pair3, {%8, %9};\n"
        "mov.b64 pair4, {%10, %11};\n"
        "mov.b64 pair5, {%12, %13};\n"
        "mov.b64 pair6, {%14, %15};\n"
        "mov.b64 pair7, {%16, %17};\n"
        "mul.rn.ftz.f32x2 pair0, pair0, factor2;\n"
        "mul.rn.ftz.f32x2 pair1, pair1, factor2;\n"
        "mul.rn.ftz.f32x2 pair2, pair2, factor2;\n"
        "mul.rn.ftz.f32x2 pair3, pair3, factor2;\n"
        "mul.rn.ftz.f32x2 pair4, pair4, factor2;\n"
        "mul.rn.ftz.f32x2 pair5, pair5, factor2;\n"
        "mul.rn.ftz.f32x2 pair6, pair6, factor2;\n"
        "mul.rn.ftz.f32x2 pair7, pair7, factor2;\n"
        "mov.b64 {val0, val1}, pair0;\n"
        "mov.b64 {val2, val3}, pair1;\n"
        "mov.b64 {val4, val5}, pair2;\n"
        "mov.b64 {val6, val7}, pair3;\n"
        "mov.b64 {val8, val9}, pair4;\n"
        "mov.b64 {val10, val11}, pair5;\n"
        "mov.b64 {val12, val13}, pair6;\n"
        "mov.b64 {val14, val15}, pair7;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, val1, val0;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, val3, val2;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, val5, val4;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, val7, val6;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte4, val9, val8;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte5, val11, val10;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte6, val13, val12;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte7, val15, val14;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "mov.b32 %1, {byte4, byte5, byte6, byte7};\n"
        "}"
        : "=r"(packed.x), "=r"(packed.y)
        : "f"(values[0][0]),  "f"(values[1][0]),  "f"(values[2][0]),  "f"(values[3][0]),
          "f"(values[4][0]),  "f"(values[5][0]),  "f"(values[6][0]),  "f"(values[7][0]),
          "f"(values[8][0]),  "f"(values[9][0]),  "f"(values[10][0]), "f"(values[11][0]),
          "f"(values[12][0]), "f"(values[13][0]), "f"(values[14][0]), "f"(values[15][0]),
          "f"(factor));
    return packed;
}

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tma_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tma_tile>();
        g.Dq.template prefetch_tma<typename G::Dq_tile>();
        g.Ds.template prefetch_tma<typename G::Ds_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_rank = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int cta_in_pair = C::USE_PREFERRED_CLUSTER ? (cta_rank & 1) : cta_rank;
    const int pair_id = C::USE_PREFERRED_CLUSTER ? (cta_rank >> 1) : 0;
    const int pair_leader = C::USE_PREFERRED_CLUSTER ? (cta_rank & ~1) : 0;
    const int cluster_width = C::USE_PREFERRED_CLUSTER ? cluster_nctarank() : C::CLUSTER_SIZE;
    const int num_pairs = cluster_width >> 1;
    const uint32_t full_mask = (1u << cluster_width) - 1;
    const uint32_t pair_mask = (0x55555555u & full_mask) << cta_in_pair;
    const uint32_t pair_ctas_mask = uint32_t(0b11u << pair_leader);
    const uint32_t self_mask = uint32_t(1u << cta_rank);
    const int num_row_blocks = g.Dq.rows() / (C::M_TILE_COUNT * C::Mb);
    const int num_col_blocks = g.Dq.cols() / (C::N_PAIR_COUNT * (C::Nb/2));
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_red_blocks = 2 * g.A.cols() / C::Kb; // A is fp4x2-packed, so A.cols() == K/2
    const int first_block = C::USE_PREFERRED_CLUSTER ? (blockIdx.x >> 1) : cluster_id;
    uint32_t stage = 0, A_stage = 0;
    uint32_t bitfield = 0xFFFF0000;
    uint32_t A_bitfield = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::A_fp4x2_tile *A_tiles = nullptr;
    if constexpr (C::USE_PREFERRED_CLUSTER) A_tiles = &sm_allocator.allocate<typename G::A_fp4x2_tile, C::A_PIPE_SLOTS>()[0];
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();
    tensor_allocator<1, C::CLUSTER_SIZE, false, C::USE_PREFERRED_CLUSTER> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ clc::handle clc_handle;
    __shared__ semaphore schedule_arrived, schedule_finished;
    __shared__ semaphore tmem_provisioned, tmem_finished;
    __shared__ semaphore A_tiles_arrived[C::USE_PREFERRED_CLUSTER ? C::A_PIPE_SLOTS : 1];
    __shared__ semaphore A_tiles_finished[C::USE_PREFERRED_CLUSTER ? C::A_PIPE_SLOTS : 1];
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished[C::M_TILE_COUNT];
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, cluster_width - 1);
        if constexpr (C::USE_PREFERRED_CLUSTER) {
            init_semaphore(schedule_arrived, 0, 1);
            init_semaphore(schedule_finished, 0, (3+C::CONSUMER_WARPGROUPS)*cluster_width + num_pairs);
            #pragma unroll
            for (int i = 0; i < C::A_PIPE_SLOTS; ++i) {
                init_semaphore(A_tiles_arrived[i], 0, 1);
                init_semaphore(A_tiles_finished[i], 0, num_pairs);
            }
        }
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, num_pairs);
        }
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::M_TILE_COUNT; ++i) init_semaphore(outputs_finished[i], 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    auto next_job = [&]<bool WG_COLLECTIVE>(int it, int &block_idx) -> bool {
        if constexpr (C::USE_PREFERRED_CLUSTER) {
            if (it == 0) return true;
            wait(schedule_arrived, (it-1)%2);
            const auto schedule = clc::query(clc_handle);
            if constexpr (WG_COLLECTIVE) {
                warpgroup::sync(2 + warpgroup_id);
                warpgroup::tma::cluster::arrive(schedule_finished, 0);
            } else tma::cluster::arrive(schedule_finished, 0);
            if (!schedule.success) return false;
            block_idx = schedule.x / C::CLUSTER_SIZE + pair_id;
            return true;
        } else {
            if (it > 0) block_idx += gridDim.x / C::CLUSTER_SIZE;
            return block_idx < num_blocks;
        }
    };

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) { // Producer group
        const int warp_id = warpgroup::warpid();
        if (warp_id == (C::USE_PREFERRED_CLUSTER ? 3 : 1)) {
            pdl::wait();
            everyone::tma::cluster::wait();
            int block_idx = first_block;
            for (int it = 0; ; ++it) {
                if (!next_job.template operator()<false>(it, block_idx)) break;
                const auto [row_block_idx, col_block_idx] = get_job_coords<C>(block_idx, num_row_blocks, num_col_blocks);

                for (int i = 0; i < num_red_blocks; ++i) {
                    if constexpr (C::USE_PREFERRED_CLUSTER) {
                        #pragma unroll
                        for (int m = 0; m < C::M_TILE_COUNT; ++m) {
                            wait(A_tiles_finished[A_stage], get_phasebit<1>(A_bitfield, A_stage));
                            if (pair_id == i % num_pairs)
                                tma::cluster::load_async<dim::ROW, cache_policy::EVICT_LAST>(A_tiles[A_stage], g.A, {row_block_idx*2*C::M_TILE_COUNT + 2*m + cta_in_pair, i}, A_tiles_arrived[A_stage], pair_mask, pair_leader);
                            update_phasebit<1>(A_bitfield, A_stage);
                            A_stage = ring_advance<C::A_PIPE_SLOTS>(A_stage);
                        }
                    }
                    wait(inputs_finished[stage], get_phasebit<1>(bitfield, stage));
                    if constexpr (!C::USE_PREFERRED_CLUSTER) {
                        tma::cluster::load_async(input_tiles[stage].A, g.A, {row_block_idx*2 + cta_in_pair, i}, tiles_arrived[stage], self_mask, pair_leader);
                    }
                    tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx*2 + cta_in_pair, i}, tiles_arrived[stage], self_mask, 0);
                    update_phasebit<1>(bitfield, stage);
                    stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                }
            }
        } else if (warp_id == 2) {
            pdl::wait();
            everyone::tma::cluster::wait();
            int block_idx = first_block;
            for (int it = 0; ; ++it) {
                if (!next_job.template operator()<false>(it, block_idx)) break;
                const auto [row_block_idx, col_block_idx] = get_job_coords<C>(block_idx, num_row_blocks, num_col_blocks);

                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(bitfield, stage));
                    if (!C::USE_PREFERRED_CLUSTER || pair_id == i % num_pairs) {
                        #pragma unroll
                        for (int m = 0; m < C::M_TILE_COUNT; ++m)
                            tma::cluster::load_async<dim::ROW, C::A_SC_CACHE_POLICY>(input_scales[stage].A[m].tma, g.A_sc, {row_block_idx*2*C::M_TILE_COUNT + 2*m + cta_in_pair, i, 0}, scales_arrived[stage], pair_mask, pair_leader);
                    }
                    tma::cluster::load_async(input_scales[stage].B[cta_in_pair].tma, g.B_sc,
                        {col_block_idx*2 + cta_in_pair, i, 0}, scales_arrived[stage], pair_ctas_mask, 0);
                    update_phasebit<1>(bitfield, stage);
                    stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                }
            }
        } else if (C::USE_PREFERRED_CLUSTER && warp_id == 1) {
            pdl::wait();
            everyone::tma::cluster::wait();
            for (int it = 0; ; ++it) {
                if (cta_rank == 0) {
                    wait(schedule_finished, (it+1)%2);
                    clc::schedule(clc_handle, schedule_arrived);
                }
                tma::expect_bytes(schedule_arrived, sizeof(clc_handle));
                wait(schedule_arrived, it%2);
                auto schedule = clc::query(clc_handle);
                tma::cluster::arrive(schedule_finished, 0);
                if (!schedule.success) break;
            }
        } else if (cta_in_pair == 0 && warp_id == 0) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            if constexpr (C::M_TILE_COUNT == 1) {
                auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::LOAD_PIPE_DEPTH>>(C::ACC_TMEM_COLS);
                auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(C::ACC_TMEM_COLS + 16*C::LOAD_PIPE_DEPTH/C::FP8_VALUES_PER_TMEM_COL);
                int block_idx = first_block;
                for (int it = 0; ; ++it) {
                    if (!next_job.template operator()<false>(it, block_idx)) break;
                    wait(outputs_finished[0], (it+1)%2);
                    tensor_after_thread_sync();
                    for (int i = 0; i < num_red_blocks; i++) {
                        tma::expect_bytes(scales_arrived[stage], C::CLUSTER_SIZE*sizeof(G::input_scales_t));
                        wait(scales_arrived[stage], get_phasebit<0>(bitfield, stage));
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*16);
                        load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[stage].A[0].tile);
                        stage_b_scales(B_sc_tm.template subtile<full_tt_fp8e4m3<64>>(stage*C::MMA_PER_TILE*32), input_scales[stage], 0);
                        stage_b_scales(B_sc_tm.template subtile<full_tt_fp8e4m3<64>>(stage*C::MMA_PER_TILE*32 + 64), input_scales[stage], 1);
                        tma::expect_bytes(tiles_arrived[stage], C::CLUSTER_SIZE*sizeof(G::input_tiles_t));
                        wait(tiles_arrived[stage], get_phasebit<0>(bitfield, stage));
                        auto B_sc_tm_subtile = B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32);
                        if (i == 0) warpgroup::mm2_ABt<64, true>(out_tm, input_tiles[stage].A, input_tiles[stage].B, A_sc_tm_subtile, B_sc_tm_subtile, inputs_finished[stage]);
                        else        warpgroup::mma2_ABt<64, true>(out_tm, input_tiles[stage].A, input_tiles[stage].B, A_sc_tm_subtile, B_sc_tm_subtile, inputs_finished[stage]);
                        update_phasebit<0>(bitfield, stage);
                        stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                    }
                    tensor_commit<2>(outputs_arrived);
                }
            } else {
                full_tt_fl<C::Nb> out_tm[C::M_TILE_COUNT];
                full_tt_fp8e4m3<16> A_sc_tm[C::M_TILE_COUNT][2];
                full_tt_fp8e4m3<64> B_sc_tm[2];
                #pragma unroll
                for (int m = 0; m < C::M_TILE_COUNT; m++) {
                    out_tm[m] = tm_allocator.template allocate<full_tt_fl<C::Nb>>(m*C::Nb);
                    #pragma unroll
                    for (int p = 0; p < 2; p++)
                        A_sc_tm[m][p] = tm_allocator.template allocate<full_tt_fp8e4m3<16>>(C::ACC_TMEM_COLS + 8*m + 4*p);
                }
                #pragma unroll
                for (int p = 0; p < 2; p++)
                    B_sc_tm[p] = tm_allocator.template allocate<full_tt_fp8e4m3<64>>(C::ACC_TMEM_COLS + 16 + 16*p);
                int block_idx = first_block;
                for (int it = 0; ; ++it) {
                    if (!next_job.template operator()<false>(it, block_idx)) break;
                    for (int i = 0; i < num_red_blocks; i++) {
                        const int parity = i & 1;
                        tma::expect_bytes(scales_arrived[stage], C::CLUSTER_SIZE*sizeof(G::input_scales_t));
                        wait(scales_arrived[stage], get_phasebit<0>(bitfield, stage));
                        #pragma unroll
                        for (int m = 0; m < C::M_TILE_COUNT; m++)
                            load_mxnv_scale_async2(A_sc_tm[m][parity], input_scales[stage].A[m].tile);
                        stage_b_scales(B_sc_tm[0], input_scales[stage], 0);
                        tma::expect_bytes(tiles_arrived[stage], C::CLUSTER_SIZE*sizeof(G::input_tiles_t));
                        wait(tiles_arrived[stage], get_phasebit<0>(bitfield, stage));
                        const int next_A_stage = ring_advance<C::A_PIPE_SLOTS>(A_stage);
                        tma::expect_bytes(A_tiles_arrived[A_stage], C::CLUSTER_SIZE*sizeof(typename G::A_fp4x2_tile));
                        tma::expect_bytes(A_tiles_arrived[next_A_stage], C::CLUSTER_SIZE*sizeof(typename G::A_fp4x2_tile));
                        wait(A_tiles_arrived[A_stage], get_phasebit<0>(A_bitfield, A_stage));
                        wait(A_tiles_arrived[next_A_stage], get_phasebit<0>(A_bitfield, next_A_stage));
                        st_descriptor<typename G::A_fp4x2_tile, 0> A0_desc(A_tiles[A_stage]);
                        st_descriptor<typename G::A_fp4x2_tile, 0> A1_desc(A_tiles[next_A_stage]);
                        st_descriptor<typename G::B_fp4x2_tile, 0> B_desc(input_tiles[stage].B);
                        #pragma unroll
                        for (int c = 0; c < 2; c++) {
                            const bool init = i == 0 && c == 0;
                            const full_tt_fp8e4m3<16> A0_sc_tm(A_sc_tm[0][parity].addr + 2*c);
                            const full_tt_fp8e4m3<16> A1_sc_tm(A_sc_tm[1][parity].addr + 2*c);
                            if (init) { wait(outputs_finished[0], (it+1)%2); tensor_after_thread_sync(); }
                            mma2_ABt_chunk<64, false, true, collector::DISCARD, collector::FILL>(out_tm[0], A0_desc, B_desc, A0_sc_tm, B_sc_tm[c], c, 0, init);
                            if (c == 1) tensor_aread_commit<2>(A_tiles_finished[A_stage], full_mask);
                            if (init) { wait(outputs_finished[1], (it+1)%2); tensor_after_thread_sync(); }
                            mma2_ABt_chunk<64, false, true, collector::DISCARD, collector::LASTUSE>(out_tm[1], A1_desc, B_desc, A1_sc_tm, B_sc_tm[c], c, 0, init);
                            if (c == 1) tensor_aread_commit<2>(A_tiles_finished[next_A_stage], full_mask);
                            if (c == 0) stage_b_scales(B_sc_tm[1], input_scales[stage], 1);
                        }
                        tensor_commit<2>(inputs_finished[stage], full_mask);
                        update_phasebit<0>(A_bitfield, A_stage);
                        update_phasebit<0>(A_bitfield, next_A_stage);
                        A_stage = ring_advance<C::A_PIPE_SLOTS>(A_stage, C::M_TILE_COUNT);
                        update_phasebit<0>(bitfield, stage);
                        stage = ring_advance<C::LOAD_PIPE_DEPTH>(stage);
                    }
                    tensor_commit<2>(outputs_arrived, pair_ctas_mask);
                }
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) { // Consumer group
        using consumer_group = group<WARPGROUP_WARPS*C::CONSUMER_WARPGROUPS>;
        everyone::tma::cluster::wait_aligned();
        if (warpgroup_id == 0 && warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        pdl::wait();
        const float global_scale = g.A_sc_global[{0}] * g.B_sc_global[{0}] / g.D_sc_global[{0}];
        const float amax_scale = fabsf(global_scale) * (1.0f / 6.0f);
        const int row = warpgroup::warpid() * 32 + laneid();

        int block_idx = first_block;
        for (int it = 0; ; ++it) {
            if (!next_job.template operator()<true>(it, block_idx)) break;
            const auto [row_block_idx, col_block_idx] = get_job_coords<C>(block_idx, num_row_blocks, num_col_blocks);
            wait(outputs_arrived, it % 2);

            if (it > 0) {
                if (warpgroup_id == 0) warpgroup::tma::store_async_read_wait<0>();
                consumer_group::sync(1);
            }

            for (int m = 0; m < C::M_TILE_COUNT; ++m) {
                const auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(m*C::Nb);
                const int tile_row = row_block_idx*2*C::M_TILE_COUNT + 2*m + cta_in_pair;
                #pragma unroll
                for (int ii = 0; ii < C::EPI_PIPE_DEPTH/C::CONSUMER_WARPGROUPS; ++ii) {
                    const int i = warpgroup_id * (C::EPI_PIPE_DEPTH/C::CONSUMER_WARPGROUPS) + ii;
                    #pragma unroll
                    for (int block = 0; block < 2; ++block) {
                        rv_fl<32*SF_VEC, naive_l> row_values;
                        float row_amax;
                        warpgroup::load_async_max_abs(
                            row_values, row_amax, out_tm, i*C::EPI_COLS + block*SF_VEC);
                        tensor_load_wait();
                        const uint8_t scale = scale_byte(row_amax * amax_scale);
                        *reinterpret_cast<uint2 *>(&output_tiles.Dq[m][{row, i*16 + block*8}]) =
                            pack_e2m1_x16_scaled(row_values, global_scale * reciprocal_scale(scale));
                        const int scale_idx = i*2 + block;
                        *reinterpret_cast<uint8_t *>(&output_tiles.Ds[m][{
                            (scale_idx/4)*32 + laneid(), warpgroup::warpid()*4 + scale_idx%4}]) = scale;
                    }
                }
                tensor_before_thread_sync();
                consumer_group::sync(1);
                if (warpgroup_id == 0) {
                    warpgroup::tma::cluster::arrive(outputs_finished[m], pair_leader, 1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                        g.Dq, output_tiles.Dq[m], {tile_row, col_block_idx});
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                        g.Ds, output_tiles.Ds[m], {tile_row*num_col_blocks*C::N_PAIR_COUNT + col_block_idx, 0});
                }
                consumer_group::sync(1);
            }
        }
        if (warpgroup_id == 0) warpgroup::tma::store_async_read_wait<0>();
        consumer_group::sync(1);
        if (warpgroup_id == 0) warpgroup::pdl::arrive();
        if (warpgroup_id == 0 && warpgroup::warpid() == 0) {
            if (warp::elect_leader()) {
                if constexpr (C::USE_PREFERRED_CLUSTER) {
                    #pragma unroll
                    for (int r = 0; r < C::CLUSTER_SIZE*C::N_PAIR_COUNT; ++r)
                        if (r < cluster_width && r != cta_rank) tma::cluster::arrive(tmem_finished, r);
                } else tma::cluster::arrive(tmem_finished, 1 - cta_rank);
            }
            wait(tmem_finished, 0);
            tm_allocator.deprovision();
        }
    }
}

} // namespace nvfp4_gemm

namespace nvfp4_utils {

// Logical (row, k-block) scales to the kernel's TMA layout: 128-row by 16-k-block 2048-byte atoms.
__global__ void swizzle_a_scales(__nv_fp8_e4m3 *tma, const __nv_fp8_e4m3 *logical, int M, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * (K / 16)) return;
    int row = idx / (K / 16), k_block_idx = idx % (K / 16);
    tma[(size_t(row/128) * (K/256) + k_block_idx/16) * 2048 + (row%128) * 16 + k_block_idx%16]
        = logical[scale_swizzle_idx(row, k_block_idx, K / 16)];
}

} // namespace nvfp4_utils

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE) __launch_bounds__(C::NUM_THREADS)
__global__ void kernel_entrypoint(const __grid_constant__ nvfp4_gemm::globals<C> g) {
    nvfp4_gemm::kernel<C>(g);
}

__global__ void reference_fp4_output_kernel(fp4e2m1_2 *Dq, fp8e4m3 *Ds, const float *D,
                                            float d_scale, size_t M, size_t N) {
    const size_t block = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t k_blocks = N / nvfp4_gemm::SF_VEC;
    if (block >= M*k_blocks) return;

    const size_t row = block / k_blocks, k_block = block % k_blocks;
    const float *d = D + row*N + k_block*nvfp4_gemm::SF_VEC;
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < nvfp4_gemm::SF_VEC; ++i) amax = fmaxf(amax, fabsf(d[i]));

    const float inv_d = 1.0f / d_scale;
    const uint8_t scale = nvfp4_gemm::scale_byte(amax * inv_d * (1.0f / 6.0f));
    const float reciprocal = nvfp4_gemm::reciprocal_scale(scale) * inv_d;
    Ds[scale_swizzle_idx(int(row), int(k_block), int(k_blocks))] = std::bit_cast<fp8e4m3>(scale);
    fp4e2m1_2 *dq = Dq + (row*N + k_block*nvfp4_gemm::SF_VEC) / 2;
    #pragma unroll
    for (int i = 0; i < nvfp4_gemm::SF_VEC; i += 2) {
        const float2 pair = {d[i] * reciprocal, d[i + 1] * reciprocal};
        dq[i/2] = std::bit_cast<fp4e2m1_2>(__nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest));
    }
}

// Dequantizes the kernel's output through the same swizzled scale layout it wrote them with, so the
// shared rel-err checker can be reused.
__global__ void dequant_fp4_output_kernel(float *D, const fp4e2m1_2 *Dq, const fp8e4m3 *Ds,
                                          float d_scale, size_t M, size_t N) {
    const size_t pair = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= M*N/2) return;

    const size_t row = pair / (N/2), col = (pair % (N/2)) * 2;
    const int k_blocks = int(N / nvfp4_gemm::SF_VEC);
    const float scale = float(Ds[scale_swizzle_idx(int(row), int(col/nvfp4_gemm::SF_VEC), k_blocks)]) * d_scale;
    const float2 codes = static_cast<float2>(std::bit_cast<__nv_fp4x2_e2m1>(Dq[pair]));
    D[row*N + col]     = codes.x * scale;
    D[row*N + col + 1] = codes.y * scale;
}

static void check_fp4_output(const fp4e2m1_2 *d_out, const fp8e4m3 *d_out_sc,
                             const fp4e2m1_2 *d_ref, const fp8e4m3 *d_ref_sc,
                             const float *d_ref_float, float *d_deq, float d_scale,
                             size_t M, size_t N) {
    const size_t sc_size = nvfp4_gemm::scale_buffer_bytes(M, N);
    std::vector<uint8_t> out(M*N/2), ref(M*N/2), out_sc(sc_size), ref_sc(sc_size);
    CUDACHECK(cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(ref.data(), d_ref, ref.size(), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(out_sc.data(), d_out_sc, out_sc.size(), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(ref_sc.data(), d_ref_sc, ref_sc.size(), cudaMemcpyDeviceToHost));

    size_t packed_mismatches = 0, scale_mismatches = 0, saturated = 0, zero_scales = 0;
    for (size_t i = 0; i < out.size(); ++i) packed_mismatches += out[i] != ref[i];
    for (size_t i = 0; i < out_sc.size(); ++i) {
        scale_mismatches += out_sc[i] != ref_sc[i];
        saturated += out_sc[i] == 0x7e; // e4m3 max, 448
        zero_scales += out_sc[i] == 0;
    }
    const double packed_rate = double(packed_mismatches) / double(out.size());
    const double scale_rate = double(scale_mismatches) / double(out_sc.size());
    std::cout << "mismatch rates: packed " << packed_rate << ", scale " << scale_rate << "\n";
    // saturated gates too: an e4m3-max scale byte means amax/6 overflowed and values were clamped.
    static constexpr double MISMATCH_RATE_TOL = 1e-3;
    if (saturated || packed_rate > MISMATCH_RATE_TOL || scale_rate > MISMATCH_RATE_TOL) {
        std::cout << "correctness: FAIL (saturated " << saturated << ", zero scales "
                  << zero_scales << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    dequant_fp4_output_kernel<<<(M*N/2 + 255)/256, 256>>>(d_deq, d_out, d_out_sc, d_scale, M, N);
    CUDACHECK(cudaDeviceSynchronize());
    check_correctness(d_deq, d_ref_float, M*N, 0.15);
}

template <typename C>
__host__ void run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    using G = nvfp4_gemm::globals<C>;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Mb=" << C::Mb << " Nb=" << C::Nb << " Kb=" << C::Kb
              << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH
              << " EPI_PIPE_DEPTH=" << C::EPI_PIPE_DEPTH << " M_TILE_COUNT=" << C::M_TILE_COUNT
              << " N_PAIR_COUNT=" << C::N_PAIR_COUNT << "\n";

    constexpr size_t M_STEP = C::M_TILE_COUNT * C::Mb;
    constexpr size_t N_STEP = C::N_PAIR_COUNT * C::Nb;
    if (M % M_STEP != 0 || N % N_STEP != 0 || K % C::Kb != 0) {
        std::cout << "unsupported shape: M, N, K must be multiples of " << M_STEP << ", "
                  << N_STEP << ", " << C::Kb << "\n";
        std::exit(EXIT_FAILURE);
    }

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    CUDACHECK(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0));
    const size_t ds_size = nvfp4_gemm::scale_buffer_bytes(M, N);
    const size_t arg_size = size_t(M) * K / 2 + size_t(N) * K / 2 + size_t(M) * N / 2 + ds_size;
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory
    std::vector<__nv_fp4x2_e2m1*> d_A(arg_group_count);
    std::vector<__nv_fp4x2_e2m1*> d_B(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_A_sc(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_A_sc_tma(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_B_sc(arg_group_count);
    std::vector<fp4e2m1_2*> d_Dq(arg_group_count);
    std::vector<fp8e4m3*> d_Ds(arg_group_count);
    fp4e2m1_2 *d_Dq_ref;
    fp8e4m3 *d_Ds_ref;
    float *d_D_ref, *d_deq, *d_scales; // d_scales = {A, B, D} global scales
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M*K*sizeof(__nv_fp4x2_e2m1)/2));
        CUDACHECK(cudaMalloc(&d_B[i], N*K*sizeof(__nv_fp4x2_e2m1)/2));
        CUDACHECK(cudaMalloc(&d_A_sc[i], M*K*sizeof(__nv_fp8_e4m3)/16));
        CUDACHECK(cudaMalloc(&d_A_sc_tma[i], M*K*sizeof(__nv_fp8_e4m3)/16));
        CUDACHECK(cudaMalloc(&d_B_sc[i], N*K*sizeof(__nv_fp8_e4m3)/16));
        CUDACHECK(cudaMalloc(&d_Dq[i], M * N / 2));
        CUDACHECK(cudaMalloc(&d_Ds[i], ds_size));
    }
    CUDACHECK(cudaMalloc(&d_Dq_ref, M * N / 2));
    CUDACHECK(cudaMalloc(&d_Ds_ref, ds_size));
    CUDACHECK(cudaMalloc(&d_D_ref, M * N * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_deq, M * N * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_scales, 3*sizeof(float)));

    // Initialize matrices with random values on device
    static constexpr float D_SCALE = 1.0f; // FP32 output global scale
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_A[i]), M*K/2, seed + i * 100, 0.0f, 255.0f);
        CUDACHECK(cudaMemset(d_A[i], 0, K/2)); // zero row 0: exercises the all-zero output block path
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_B[i]), N*K/2, seed + i * 100 + 1, 0.0f, 255.0f);
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_A_sc[i], M*K/16, seed + i*100 + 2, 0.1f, 10.0f);
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_B_sc[i], N*K/16, seed + i*100 + 3, 0.1f, 10.0f);
        CUDACHECK(cudaMemset(d_Dq[i], 0, M*N/2));
        CUDACHECK(cudaMemset(d_Ds[i], 0, ds_size));
        nvfp4_utils::swizzle_a_scales<<<(M*K/16 + 255)/256, 256>>>(d_A_sc_tma[i], d_A_sc[i], M, K);
    }
    // e4m3 output scales cap at 448; an A*B product of 0.0025 keeps amax/6 in range at K=32768.
    const float h_scales[3] = {0.05f, 0.05f, D_SCALE};
    CUDACHECK(cudaMemcpy(d_scales, h_scales, sizeof(h_scales), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(d_Dq_ref, 0, M*N/2));
    CUDACHECK(cudaMemset(d_Ds_ref, 0, ds_size));

    // Compute reference GEMM on device
    reference_nvfp4_gemm<float>(d_D_ref, d_A[0], d_B[0], d_A_sc[0], d_B_sc[0],
                               d_scales+0, d_scales+1, int(M), int(N), int(K));
    reference_fp4_output_kernel<<<(M*N/16 + 255)/256, 256>>>(d_Dq_ref, d_Ds_ref, d_D_ref, D_SCALE, M, N);
    CUDACHECK(cudaDeviceSynchronize());

    // Prepare kernel inputs
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::A_fp4x2_gl Ag{d_A[i], nullptr, nullptr, M, K/2};
        typename G::A_sc_gl Asg{d_A_sc_tma[i], nullptr, M/128, K/32, nullptr};
        typename G::A_sc_global_gl Asgg{d_scales+0, nullptr, nullptr, nullptr, nullptr};
        typename G::B_fp4x2_gl Bg{d_B[i], nullptr, nullptr, N, K/2};
        typename G::B_sc_gl Bsg{d_B_sc[i], nullptr, N/128, K/32, nullptr};
        typename G::B_sc_global_gl Bsgg{d_scales+1, nullptr, nullptr, nullptr, nullptr};
        typename G::A_sc_global_gl Dsgg{d_scales+2, nullptr, nullptr, nullptr, nullptr};
        typename G::Dq_gl Dqg{d_Dq[i], nullptr, nullptr, M, N/2};
        typename G::Ds_gl Dsg{d_Ds[i], nullptr, nullptr, ds_size/16, 16};
        g.push_back(G{Ag, Asg, Asgg, Bg, Bsg, Bsgg, Dsgg, Dqg, Dsg});
    }

    // Set kernel attributes
    set_oversized_smem(kernel_entrypoint<C>, g[0].dynamic_shared_memory());
    LaunchConfig<true, true> launch_config(g[0].grid(), g[0].block(), g[0].dynamic_shared_memory(), 0,
                                           dim3(C::CLUSTER_SIZE * C::N_PAIR_COUNT, 1, 1),
                                           dim3(C::CLUSTER_SIZE, 1, 1));

    // Number of iterations
    int num_warmups = ncu ? 0 : 5;
    int num_iters = ncu ? 1 : 10;

    // Cooldown between configurations
    sleep_ms(500);

    auto launch = [&](int n) {
        for (int i = 0; i < n; i++)
            CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[i % arg_group_count]));
    };
    launch(num_warmups);

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    launch(num_iters);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Check correctness
    check_fp4_output(d_Dq[0], d_Ds[0], d_Dq_ref, d_Ds_ref, d_D_ref, d_deq, D_SCALE, M, N);

    // Cleanup
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaFree(d_A[i]));
        CUDACHECK(cudaFree(d_A_sc[i]));
        CUDACHECK(cudaFree(d_A_sc_tma[i]));
        CUDACHECK(cudaFree(d_B[i]));
        CUDACHECK(cudaFree(d_B_sc[i]));
        CUDACHECK(cudaFree(d_Dq[i]));
        CUDACHECK(cudaFree(d_Ds[i]));
    }
    CUDACHECK(cudaFree(d_Dq_ref));
    CUDACHECK(cudaFree(d_Ds_ref));
    CUDACHECK(cudaFree(d_D_ref));
    CUDACHECK(cudaFree(d_deq));
    CUDACHECK(cudaFree(d_scales));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
}

int main(int argc, char **) {
    bool ncu = argc > 1;

    // Template parameters: N_PAIR_COUNT, LOAD_PIPE_DEPTH, SUPERGROUP_SIZE
    run_benchmark<nvfp4_gemm::config<1, 6, 8>>( 1024,  1024,  1024, ncu);
    run_benchmark<nvfp4_gemm::config<1, 6, 8>>( 2048,  2048,  2048, ncu);
    run_benchmark<nvfp4_gemm::config<1, 7, 8>>( 4096,  4096,  4096, ncu);
    run_benchmark<nvfp4_gemm::config<2, 5, 8>>( 8192,  8192,  8192, ncu);
    run_benchmark<nvfp4_gemm::config<4, 5, 6>>(16384, 16384, 16384, ncu);
    run_benchmark<nvfp4_gemm::config<4, 5, 4>>(32768, 32768, 32768, ncu);

    return 0;
}
