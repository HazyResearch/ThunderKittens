// Copy from https://github.com/deepseek-ai/DeepGEMM/blob/0c88cd01392c1073c7049a97d6328c7bba9b3947/deep_gemm/include/deep_gemm/scheduler.cuh

#pragma once

#include "utils.cuh"

namespace deep_gemm {

enum class GemmType {
    Normal,
    GroupedContiguous,
    GroupedMasked
};

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNum1DBlocksPerGroup = 16>
struct Scheduler {
    // int current_iter = -1;
    // uint32_t SHAPE_N;
    uint32_t kNumNBlocks;
    uint32_t kNumGroups;

    uint32_t num_aligned_m_blocks;

    // For normal GEMM
    // Maybe not used in the masked grouped GEMM
    uint32_t num_blocks;
    uint32_t num_blocks_in_group;
    bool is_peer_cta_alive = true;

    // For grouped GEMM
    int* grouped_layout;

    // Only used for masked layout
    uint32_t curr_group_idx, curr_cumsum;

    __device__ __forceinline__ explicit Scheduler(uint32_t shape_m, uint32_t shape_n, uint32_t group_count = 1,
                                                  int* grouped_layout = nullptr) {
        kNumNBlocks = ceil_div(shape_n, BLOCK_N);
        kNumGroups = group_count;
        num_aligned_m_blocks = ceil_div(shape_m, BLOCK_M);
        if constexpr (kGemmType == GemmType::Normal) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
            this->grouped_layout = grouped_layout;
        } else if (kGemmType == GemmType::GroupedMasked) {
            curr_group_idx = curr_cumsum = 0;
            this->grouped_layout = grouped_layout;
        }
    }

    // ReSharper disable once CppNotAllPathsReturnValue
    __device__ __forceinline__ bool is_computation_valid(const uint32_t& m_block_idx, const uint32_t& m_offset) const {
        if constexpr (kGemmType == GemmType::Normal) {
            return true;
        } else if constexpr (kGemmType == GemmType::GroupedContiguous) {
            return __ldg(grouped_layout + m_offset + m_block_idx * BLOCK_M) >= 0;
        } else if constexpr (kGemmType == GemmType::GroupedMasked) {
            return m_offset + m_block_idx * BLOCK_M < __ldg(grouped_layout + curr_group_idx);
        }
    }

    __device__ __forceinline__ bool is_tma_multicast_valid(const uint32_t& m_block_idx) const {
        if (num_blocks_in_group == 1)
            return false;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::GroupedMasked) {
            return true;
        } else {
            DG_STATIC_ASSERT(kGemmType == GemmType::GroupedContiguous, "Invalid Gemm type");
            if constexpr (kIsTMAMulticastOnA) {
                return true;
            } else {
                auto group_idx = __ldg(grouped_layout + m_block_idx * BLOCK_M);
                auto peer_group_idx = __ldg(grouped_layout + (m_block_idx ^ 1) * BLOCK_M);
                return group_idx == peer_group_idx;
            }
        }
    }

    __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t& num_m_blocks, const uint32_t& block_idx,
                                                           uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumTMAMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        auto primary_num_blocks = kIsTMAMulticastOnA ? kNumNBlocks : num_m_blocks;
        auto secondary_num_blocks = kIsTMAMulticastOnA ? num_m_blocks : kNumNBlocks;
        auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        auto group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Fix unaligned TMA multicast
        if (kNumTMAMulticast > 1 and num_blocks_in_group % 2 != 0) {
            if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
                num_blocks_in_group = num_blocks_in_group ^ 1;
            } else {
                in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
                first_block_idx += num_blocks_in_group ^ 1;
                num_blocks_in_group = 1;
            }
        }

        // Convert to final M/N block indices
        if constexpr (kIsTMAMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    template <bool kIgnoreGroupedForGroupedContiguous=true>
    __device__ __forceinline__ uint32_t get_global_idx(const uint32_t& shape_dim, const uint32_t& block_size,
                                                       const uint32_t& block_idx, const uint32_t& m_block_idx=0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::GroupedContiguous) {
            auto offset = kIgnoreGroupedForGroupedContiguous ? 0 : max(0, __ldg(grouped_layout + m_block_idx * BLOCK_M));
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::GroupedMasked) {
            return curr_group_idx * shape_dim + block_idx * block_size;
        }
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx, int current_iter) {
        const auto next_block_idx = current_iter * gridDim.x + blockIdx.x;

        if constexpr (kGemmType == GemmType::GroupedMasked) {
            uint32_t num_m_blocks;
            while (true) {
                // End of the task
                if (curr_group_idx == kNumGroups)
                    return false;
                // Within the current group
                num_m_blocks = ceil_div(static_cast<uint32_t>(__ldg(grouped_layout + curr_group_idx)), BLOCK_M);
                auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
                    break;

                // Move to check the next group
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
        } else {
            if (next_block_idx >= num_blocks)
                return false;
            // NOTES: we don't have to set `is_peer_cta_alive` for masked grouped GEMM, as it must be aligned
            is_peer_cta_alive = kNumNBlocks % kNumTMAMulticast == 0 or          // Always aligned on N (constant bypass)
                                num_aligned_m_blocks % kNumTMAMulticast == 0 or // Always aligned on M (constant bypass)
                                (next_block_idx ^ 1) < num_blocks;              // Peer CTA in bound
            get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }
};

#pragma clang diagnostic pop

} // namespace deep_gemm
