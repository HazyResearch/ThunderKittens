#pragma once

#include "llama.cuh"

namespace kittens::prototype::vm
{

    template <typename Config, typename Globals, typename rv_t>
    __device__ inline void rms_norm(Globals &g, state<Config> &s, rv_t &activations_vec, int rms_scale_activation_page, semaphore &activations_arrived, semaphore &rms_scale_arrived, int scratch_offset)
    {

        constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

        sv_bf<REDUCTION_DIM_PER_WARP>* rms_scale_smem   = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP>*>(s.pages[rms_scale_activation_page].ptr());
        sv_bf<REDUCTION_DIM_PER_WARP>* activations_smem = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP>*>(s.pages[rms_scale_activation_page].ptr(sizeof(sv_bf<2048>)));

        // Setup
        rv_t copy_activations_vec; // vector of length 512
        rv_t rms_scale_vec;

        wait(activations_arrived, 0);

        warp::load(activations_vec, activations_smem[warpid()]);
        warp::sync();

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

        // multiply by rms scale
        wait(rms_scale_arrived, 0);

        // TODO
        // if (warpid() == 0 && laneid() == 0)
        // {
        //     s.record(TEVENT_TRIPLES_END);
        // }

        warp::load(rms_scale_vec, rms_scale_smem[warpid()]);
        warp::sync();

        warp::mul(activations_vec, activations_vec, rms_scale_vec);
    }

    template <typename rt_t, int WARPS_PER_PAGE, typename Config, typename Globals, typename rv_t>
    __device__ inline void matvec(Globals &g, state<Config> &s, rv_t &activations_vec, semaphore &weights_arrived, int weight_pid, int scratch_offset)
    {

        rt_t weights, broadcast_activations; // 16 x 512 
        typename rt_t::col_vec proj_partial_col_format;
        rv<float, rt_t::rows> proj_partial;

        int page_index = warpid() / WARPS_PER_PAGE;
        int index_in_page = warpid() % WARPS_PER_PAGE;

        wait(weights_arrived, 0);

        // TODO
        // if (warpid() % WARPS_PER_PAGE == 0 && laneid() == 0)
        // {
        //     s.record(TEVENT_TRIPLES_END + 1 + page_index);
        // }

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
        if (laneid() < 16)
        {
            // this might be a bad idea but yolo, it's probably an okay start
            // and fortunately this is code where ncu will tell us if it's bad..
            atomicAdd(&smem_proj_partials[laneid()], proj_partial[0][0]);
        }
    }

}