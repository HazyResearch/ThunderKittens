/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in tensor memory.
 */

#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

template<typename D, typename AB, int M, int K, int N, bool trans_a, bool trans_b, bool neg=false>
struct mma_descriptor {
    uint32_t desc;
    __device__ inline mma_descriptor() {
        desc = 0;
        if constexpr (sizeof(AB) == 2) { // kind::f16
            static_assert(std::is_same_v<D, float> || std::is_same_v<AB, half>);
            desc |= 0b00     << 0;  // sparsity bits unneeded
            desc |= 0b0      << 2;  // dense
            desc |= 0b0      << 3;  // no saturate on fp types
            if constexpr (std::is_same_v<D, float>) { desc |= 0b01 << 4; } // D matrix is FP32
            else                                    { desc |= 0b00 << 4; } // D matrix is FP16
            desc |= 0b0      << 6;  // reserved
            desc |= 0b000    << 7;  // 16-bit A input type as FP16
            desc |= 0b000    << 10; // 16-bit B input type as BF16
            desc |= 0b0      << 13; // Don't negate A matrix
            desc |= 0b0      << 14; // Don't negate B matrix
            desc |= 0b0      << 15; // Don't transpose A matrix
            desc |= 0b1      << 16; // Don't transpose B matrix
            desc |= 0b000010 << 17; // B matrix has N=16
            desc |= 0b0      << 23; // reserved
            desc |= 0b00100  << 24; // A matrix has M=64
            desc |= 0b0      << 29; // reserved
            desc |= 0b00     << 30; // no shift for B-matrix reuse
        }
        else {
            static_assert(sizeof(AB) == 999, "Invalid AB type size; not implemented yet.");
        }
    }
};


} // namespace kittens