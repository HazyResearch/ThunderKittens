#pragma once

#include "llama.cuh"

namespace kittens::prototype::vm
{

template<typename Config, kittens::ducks::st::all sv_t> __device__ static inline auto rms_norm(sv_t &rms_scale_smem, sv_t &activations_smem, void* scratch_memory) {
    using rv_t = rv_fl<sv_t::length>;
    rv_t activations_vec, sq_activations_vec, rms_scale_vec;

    warp::load(activations_vec, activations_smem);
    warp::copy(sq_activations_vec, activations_vec);
    warp::mul(sq_activations_vec, sq_activations_vec, sq_activations_vec);
    float partial_sum = warp::sum(sq_activations_vec);

    float *smem_rms_partial_sums = (float*)scratch_memory;
    
    if (laneid() == 0) {
        smem_rms_partial_sums[warpid()] = partial_sum;
    }

    group<Config::NUM_CONSUMER_WARPS>::sync(0);
    float full_sum = 0;
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {
        full_sum += smem_rms_partial_sums[i];
    }
    float variance = full_sum / 2048.0f;
    float rms_scale = rsqrtf(variance + g.rms_norm_eps);

    warp::mul(activations_vec, activations_vec, rms_scale);
    warp::load(rms_scale_vec, rms_scale_smem[warpid()]);
    warp::mul(activations_vec, activations_vec, rms_scale_vec);

    return activations_vec;
}

template<typename Config, kittens::ducks::st::all st_t> __device__ static inline void matvec(sv_fl<st_t::rows> &out_smem, st_t &weights_smem, rv_fl<st_t::rows> &activations) {
    using rt_t = rt_fl<st_t::rows, st_t::cols>;
    using rcv_t = typename rt_t::col_vec;
    using rv_t = rv_fl<st_t::rows>;
    using sv_t = sv_bf<st_t::rows>;

    rcv_t col_activations;
    warp::copy(col_activations, activations);

    rcv_t broadcast_activations, weights;
    warp::broadcast_col(broadcast_activations, col_activations);
    warp::load(weights, weights_smem);
    warp::mul(broadcast_activations, broadcast_activations, weights);
    warp::row_sum(proj_partial_col_format, broadcast_activations);

    rv_t proj_partial;
    warp::copy(proj_partial, proj_partial_col_format);

    if (laneid() < 16)
    {
        // this might be a bad idea but yolo, it's probably an okay start
        // and fortunately this is code where ncu will tell us if it's bad..
        atomicAdd(&out_smem[laneid()], proj_partial[0][0]);
    }
    warp::sync();
}

    template <typename rt_t, int WARPS_PER_PAGE, typename Config, typename Globals, typename rv_t>
    __device__ inline void matvec(Globals &g, state<Config> &s, rv_t &activations_vec, semaphore &weights_arrived, int weight_pid, int scratch_offset)
    {

        rt_t weights, broadcast_activations;
        typename rt_t::col_vec proj_partial_col_format;
        rv<float, rt_t::rows> proj_partial;

        int page_index = warpid() / WARPS_PER_PAGE;
        int index_in_page = warpid() % WARPS_PER_PAGE;

        if (index_in_page == 0 && laneid() == 0)
        {
            s.record(WEIGHT_WAIT_START + page_index);
        }

        wait(weights_arrived, 0);

        // TODO
        if (index_in_page == 0 && laneid() == 0)
        {
            s.record(WEIGHT_WAIT_DONE + page_index);
        }

        using st_slice_t = st_bf<rt_t::rows, rt_t::cols>;

        st_slice_t(&weights_smem)[WARPS_PER_PAGE] = reinterpret_cast<st_slice_t(&)[WARPS_PER_PAGE]>(s.pages[weight_pid]);
        warp::load(weights, weights_smem[index_in_page]);
        warp::sync();

        warp::broadcast_col(broadcast_activations, activations_vec);
        warp::mul(broadcast_activations, broadcast_activations, weights);
        warp::row_sum(proj_partial_col_format, broadcast_activations);
        warp::copy(proj_partial, proj_partial_col_format);

        auto smem_proj_partials = ((float *)s.scratch()) + scratch_offset;

        // now the first 16 threads have the output.

        if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
        {
            s.record(ATOMIC_ADD_START);
        }

        if (laneid() < 16)
        {
            // this might be a bad idea but yolo, it's probably an okay start
            // and fortunately this is code where ncu will tell us if it's bad..
            atomicAdd(&smem_proj_partials[laneid()], proj_partial[0][0]);
        }

        warp::sync();

        if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
        {
            s.record(ATOMIC_ADD_END);
        }
    }


    template <typename Config, typename Globals, typename rv_t>
    __device__ inline void rms_norm(Globals &g, state<Config> &s, rv_t &activations_vec, int rms_scale_activation_page, semaphore &activations_arrived, semaphore &rms_scale_arrived, int scratch_offset, int finish_pages)
    {

        constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        sv_bf<REDUCTION_DIM_PER_WARP> *rms_scale_smem = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP> *>(s.pages[rms_scale_activation_page].ptr());
        sv_bf<REDUCTION_DIM_PER_WARP> *activations_smem = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP> *>(s.pages[rms_scale_activation_page].ptr(sizeof(sv_bf<2048>)));

        // Setup
        rv_t copy_activations_vec;
        rv_t rms_scale_vec;

        wait(activations_arrived, 0);

        if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
        {
            s.record(ACT_WAIT_DONE);
        }

        warp::load(activations_vec, activations_smem[warpid()]);

        // Step 2: Apply RMS normalization
        warp::copy(copy_activations_vec, activations_vec);                           // cast to float
        warp::mul(copy_activations_vec, copy_activations_vec, copy_activations_vec); // square
        float partial_sum = warp::sum(copy_activations_vec);

        auto smem_rms_partial_sums = ((float *)s.scratch()) + scratch_offset;
        // aggregate sums across the consumer warps
        if (laneid() == 0)
        {
            smem_rms_partial_sums[warpid()] = partial_sum;
        }

        group<Config::NUM_CONSUMER_WARPS>::sync(0);

        float full_sum = 0;
        for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++)
        {
            full_sum += smem_rms_partial_sums[i];
        }

        float variance = full_sum / 2048.0f;
        float rms_scale = rsqrtf(variance + g.rms_norm_eps);

        warp::copy(copy_activations_vec, activations_vec); // unsquare
        warp::mul(copy_activations_vec, copy_activations_vec, rms_scale);
        warp::copy(activations_vec, copy_activations_vec);

        if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
        {
            s.record(RMS_SCALE_WAIT_START);
        }

        // multiply by rms scale
        wait(rms_scale_arrived, 0);

        if (group<Config::NUM_CONSUMER_WARPS>::laneid() == 0)
        {
            s.record(RMS_SCALE_WAIT_DONE);
        }

        // TODO
        // if (warpid() == 0 && laneid() == 0)
        // {
        //     s.record(TEVENT_TRIPLES_END);
        // }

        warp::load(rms_scale_vec, rms_scale_smem[warpid()]);

        warp::mul(activations_vec, activations_vec, rms_scale_vec);
    }


}