#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

#include <cuda.h>
#include <iostream>

namespace kittens {
namespace tma {

/* ----------   Prefetch Tensor Map  ---------- */

/**
 * @brief Prefetches data from global memory into a shared memory tile, along with the tensormap.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in] tile_row_idx The row index of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the requested tile. This is in units of complete tiles.
 */
template<ducks::st::all ST>
__device__ static inline void prefetch(ST &dst, void const* const src_tma_map, int tile_row_idx, int tile_col_idx=0) {
    if (::kittens::laneid()) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
        int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (dst.rows);
        int32_t crd2 = tile_col_idx * (dst.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.async.bulk.prefetch.tensor.3d.L2.global.tile"
            " [%0, {%1, %2, %3}];"
            :
            : "l"(tma_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory"
        );
    }
}

/* ----------   Async load and store data from gmem/smem  ---------- */

/**
 * @brief Asynchronously stores data into global memory from a shared memory tile.
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row index of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the tile destination. This is in units of complete tiles.
 */
template<ducks::st::all ST>
__device__ static inline void store_async(void *dst_tma_map, const ST &src, int tile_row_idx, int tile_col_idx=0) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (src.rows);
        int32_t crd2 = tile_col_idx * (src.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory"
        );
    }
}

/* ----------   Async reduction + store data from gmem/smem  ---------- */

/**
 * @brief Asynchronously performs an add reduction and stores the result into global memory from a shared memory tile.
 *
 * This function performs an asynchronous add reduction and copy operation using CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row index of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the tile destination. This is in units of complete tiles.
 */
template<ducks::st::all ST>
__device__ static inline void store_add_async(void *dst_tma_map, const ST &src, int tile_row_idx, int tile_col_idx=0) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
        int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (src.rows);
        int32_t crd2 = tile_col_idx * (src.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"
            " [%0, {%2, %3, %4}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory"
        );
    }
}

/**
 * @brief Asynchronously performs an min reduction and stores the result into global memory from a shared memory tile.
 *
 * This function performs an asynchronous min reduction and copy operation using CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row index of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the tile destination. This is in units of complete tiles.
 */
template<ducks::st::all ST>
__device__ static inline void store_min_async(void *dst_tma_map, const ST &src, int tile_row_idx, int tile_col_idx=0) {
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
        int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (src.rows);
        int32_t crd2 = tile_col_idx * (src.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.reduce.async.bulk.tensor.3d.global.shared::cta.min.tile.bulk_group"
            " [%0, {%2, %3, %4}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory"
        );
    }
}

/**
 * @brief Asynchronously performs an max reduction and stores the result into global memory from a shared memory tile.
 *
 * This function performs an asynchronous max reduction and copy operation using CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row index of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the tile destination. This is in units of complete tiles.
 */
template<ducks::st::all ST>
__device__ static inline void store_max_async(void *dst_tma_map, const ST &src, int tile_row_idx, int tile_col_idx=0) {
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
        int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (src.rows);
        int32_t crd2 = tile_col_idx * (src.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.reduce.async.bulk.tensor.3d.global.shared::cta.max.tile.bulk_group"
            " [%0, {%2, %3, %4}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory"
        );
    }
}

/**
 * @brief Asynchronously loads data from global memory into a shared memory tile.
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in,out] bar The barrier used for synchronization of the asynchronous copy.
 * @param[in] tile_row_idx The row index of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the requested tile. This is in units of complete tiles.
 */
template<ducks::st::all ST>
__device__ static inline void load_async(ST &dst, void const* const src_tma_map, barrier& bar, int tile_row_idx, int tile_col_idx=0) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
        uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));

        int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (dst.rows);
        int32_t crd2 = tile_col_idx * (dst.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5}], [%2];"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory"
        );
    }
}

namespace cluster {

/**
 * @brief Asynchronously loads data from global memory into a shared memory tile, across a threadblock cluster
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in,out] bar The barrier used for synchronization of the asynchronous copy.
 * @param[in] tile_row_idx The row index of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the requested tile. This is in units of complete tiles.
 * @param[in] cluster_mask The mask of the clusters to broadcast to.
 */
template<ducks::st::all ST>
__device__ static inline void load_async(ST &dst, void const* const src_tma_map, barrier& bar, int tile_row_idx, int tile_col_idx, uint16_t cluster_mask) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
        uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));

        int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (dst.rows);
        int32_t crd2 = tile_col_idx * (dst.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1, {%3, %4, %5}], [%2], %6;"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "h"(cluster_mask)
            : "memory"
        );
    }
}

} // namespace cluster
} // namespace tma
} // namespace kittens