#pragma once

#include "../../../common/common.cuh"
#include "../../../types/shared/shared.cuh"

namespace dsmem {

template<int height, int width, ducks::st_layout::all layout>
__device__ static inline void tile_distribute_smem(st<bf16, height, width, layout> &dst_, st<bf16, height, width, layout> &src_, int cluster_size, int dst_idx, uint32_t size_bytes, uint64_t& barrier) 
{
    if (threadIdx.x == 0) {
        void const* const ptr = &barrier;
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        // **************************************************
        // load from src to dst in different threadblocks
        auto src = &src_;
        auto dst = &dst_;
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src)); 
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));  

        uint32_t neighbor_rank = dst_idx;

        // mapa instr = https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa 
        // find dst addr in neighbor's cta
        uint32_t neighbor_addr_dst = dst_ptr;
        asm volatile (
            "mapa.shared::cluster.u32  %0, %1, %2;\n"
            : "=r"(neighbor_addr_dst)
            : "r"(dst_ptr), "r"(neighbor_rank)
        );
        
        uint32_t neighbor_addr_mbar = mbar_ptr;
        asm volatile (
            "mapa.shared::cluster.u32  %0, %1, %2;\n"
            : "=r"(neighbor_addr_mbar)
            : "r"(mbar_ptr), "r"(neighbor_rank)
        );
        
        // cp.async instr = https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk 
        // copy src into dst in neighbor's cta
        asm volatile (
            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :
            : "r"(neighbor_addr_dst), "r"(src_ptr), "r"(size_bytes), "r"(neighbor_addr_mbar)
            : "memory"
        );
    }
}

__device__ static inline void distribution_wait(uint64_t& barrier, int kPhaseBit) {
    void const* const ptr = &barrier;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ static inline void init_barrier(uint64_t& barrier, int tc) {
    if (threadIdx.x == 0) {
        void const* const ptr = &barrier;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(bar_ptr), "r"(tc)
        );
    }
}

__device__ static inline void set_barrier_bytes(uint64_t& barrier, uint32_t bytes) {
    if (threadIdx.x == 0) {
        void const* const ptr = &barrier;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile (
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(bar_ptr), "r"(bytes)
        );

    }
}

}