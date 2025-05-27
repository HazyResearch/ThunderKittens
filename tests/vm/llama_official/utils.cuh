#pragma once

#include "llama.cuh"

namespace kittens::prototype::vm {
template <typename Config, kittens::ducks::sv::all sv_t>
__device__ static inline auto
rms_norm(const sv_t &rms_scale_smem, const sv_t &activations_smem,
         float rms_norm_eps, void *scratch_memory) {
    using rv_t = rv_fl<sv_t::length>;
    rv_t activations_vec, sq_activations_vec, rms_scale_vec;

    warp::load(activations_vec, activations_smem);
    warp::copy(sq_activations_vec, activations_vec);
    warp::mul(sq_activations_vec, sq_activations_vec, sq_activations_vec);
    float partial_sum = warp::sum(sq_activations_vec);

    float *smem_rms_partial_sums = (float *)scratch_memory;
    if (laneid() == 0) {
        smem_rms_partial_sums[warpid()] = partial_sum;
    }
    group<Config::NUM_CONSUMER_WARPS>::sync(0);

    float full_sum = 0;
#pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {
        full_sum += smem_rms_partial_sums[i];
    }

    float variance = full_sum / 2048.0f;
    float rms_scale = rsqrtf(variance + rms_norm_eps);

    warp::mul(activations_vec, activations_vec, rms_scale);
    warp::load(rms_scale_vec, rms_scale_smem);
    warp::mul(activations_vec, activations_vec, rms_scale_vec);

    return activations_vec;
}

#ifdef KITTENS_BLACKWELL
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     rv_fl<st_t::cols> &activations) {
    using rt_t = rt_bf<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename rt_fl<16, 16>::col_vec;
    // using rcv_t = typename rt_t::col_vec;
    using rv_t = rv_fl<st_t::rows>;
    using sv_t = sv_bf<st_t::rows>;

    rrv_t row_activations;
    warp::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    warp::broadcast_col(broadcast_activations, row_activations);
    warp::load(weights, weights_smem);
    rt_fl<16, 16> out_activations;
    warp::zero(out_activations);
    warp::mma_ABt(out_activations, weights, broadcast_activations,
                  out_activations);
    rcv_t sum_col_vec;
    warp::row_max(sum_col_vec, out_activations);

    rv_t sum_vec;
    warp::copy(sum_vec, sum_col_vec);

    if (laneid() < 16) {
        out_smem[laneid()] = sum_vec[0][0];
    }
    warp::sync();
}
#else
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     rv_fl<st_t::cols> &activations) {
    using rt_t = rt_fl<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename rt_t::col_vec;
    using rv_t = rv_fl<st_t::rows>;
    using sv_t = sv_bf<st_t::rows>;

    rrv_t row_activations;
    warp::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    warp::broadcast_col(broadcast_activations, row_activations);
    warp::load(weights, weights_smem);
    warp::mul(broadcast_activations, broadcast_activations, weights);
    rcv_t sum_col_vec;
    warp::row_sum(sum_col_vec, broadcast_activations);

    rv_t sum_vec;
    warp::copy(sum_vec, sum_col_vec);

    if (laneid() < 16) {
        out_smem[laneid()] = sum_vec[0][0];
    }
    warp::sync();
}
#endif

template <typename Config, kittens::ducks::sv::all sv_t, typename rv_t,
          int SCRATCH_BYTES_PER_WARP>
__device__ static inline void matvec_reduce(uint8_t *scratch, rv_t &sum_vec) {
    rv_t part_vec;
    warp::zero(sum_vec);

#pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {

        // TODO: for now, deliberately not using sizeof(sv_t) here because we've
        // had alignment issues before.
        sv_t &part =
            *reinterpret_cast<sv_t *>(scratch + (i * SCRATCH_BYTES_PER_WARP));

        warp::load(part_vec, part);
        warp::add(sum_vec, sum_vec, part_vec);
    }
}

} // namespace kittens::prototype::vm