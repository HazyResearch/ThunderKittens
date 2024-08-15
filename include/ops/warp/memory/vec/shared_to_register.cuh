/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Load data from a shared vector into a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
template<ducks::rv::all RV, ducks::sv::all SV>
__device__ inline static void load(RV &dst, const SV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(src.tiles == dst.outer_dim);
    
    int laneid = ::kittens::laneid();
    
    __syncwarp();
    if constexpr (dst.inner_dim == 2) {
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
            int o_dim = w*4 + (laneid/4) / 2;
            int i_dim = (laneid/4) % 2;
            // this should be a maximally coalesced load.
            if(idx < dst.outer_dim*16)
                dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(*(U2*)&src[idx]);
        }
        __syncwarp();
        // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = 8*(w%4) + (laneid%4); // repeats every 64 columns
            dst[w][0] = packed_shfl_sync(MASK_ALL, dst[w][0], leader);
            dst[w][1] = packed_shfl_sync(MASK_ALL, dst[w][1], leader+4);
        }
    }
    else {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
            int idx = w*32 + (laneid%4)*8 + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < dst.outer_dim*16) {
                T tmp = base_types::convertor<T, U>::convert(src[idx]);
                if(laneid%2==0) dst[o_dim][0].x = tmp;
                else dst[o_dim][0].y = tmp;
            }
        }
        __syncwarp();
        // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = (laneid/4)*4 + 2*(w%2); // repeats every 64 columns
            dst[w][0].x = __shfl_sync(MASK_ALL, dst[w][0].x, leader);
            dst[w][0].y = __shfl_sync(MASK_ALL, dst[w][0].y, leader+1);
        }
    }
}

/**
 * @brief Store data into a shared vector from a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination shared vector.
 * @param src[in]  The source register vector.
 */
template<ducks::sv::all SV, ducks::rv::all RV>
__device__ inline static void store(SV &dst, const RV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(dst.tiles == src.outer_dim);
    
    int laneid = ::kittens::laneid();
    
    __syncwarp();
    if constexpr (src.inner_dim == 2) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
            int o_dim = w*4 + (laneid/4) / 2;
            int i_dim = (laneid/4) % 2;
            // this should be a maximally coalesced store. I hope!
            if(idx < src.outer_dim*16)
                *(U2*)&dst[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
        }
    }
    else {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
            int idx = w*32 + (laneid%4)*8 + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < src.outer_dim*16) {
                U tmp;
                if(laneid%2==0) tmp = base_types::convertor<U, T>::convert(src[o_dim][0].x);
                else tmp = base_types::convertor<U, T>::convert(src[o_dim][0].y);
                dst[idx] = tmp;
            }
        }
    }
}

}