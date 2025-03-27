#pragma once

#include <cuda.h>
#include "../util/reduce.cuh"


namespace kittens {

template <int axis, bool assume_aligned, ReduceOp OP, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void ld_reduce_op(ST &dst, const PGL &src, int dev_id, const COORD &idx) {
    using T = typename ST::dtype;
    using U = typename PGL::dtype;
    const int row_stride = src[dev_id].template stride<axis>();

    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy;
    constexpr int dst_num_elem = dst.height*dst.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T>;
    constexpr int total_calls = (dst_num_elem + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy);
    constexpr bool needs_bounds_check = dst_num_elem % (N_THREADS*elem_per_memcpy);

    coord<> coord = idx.template unit_coord<axis, 3>();
    auto index = ((coord.b * src[dev_id].depth() + coord.d) * src[dev_id].rows() + coord.r) * src[dev_id].cols() + coord.c;
    U* mc_ptr = src.mc_vas[dev_id] + index;
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int laneid = threadIdx.x % N_THREADS;

    for (int i = 0; i < total_calls; i++) {
        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % dst.cols;

        if constexpr (needs_bounds_check) {
            if (row >= dst.rows) continue;
        }

        if constexpr (assume_aligned) {
            float4 tmp;
            U* src_ptr = static_cast<U*>(mc_ptr) + row*row_stride + col;
            multimem_ld_reduce_op<T, OP>::apply_vec(&tmp, src_ptr);
            move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
        }
        else {
            if (row + coord.template dim<axis>() < src[dev_id].template shape<axis>()) {
                float4 tmp;
                U* src_ptr = static_cast<U*>(mc_ptr) + row*row_stride + col;
                multimem_ld_reduce_op<T, OP>::apply_vec(&tmp, src_ptr);
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
            }
        }
    }
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_add(ST &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::ADD>(dst, src, dev_id, idx);
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_min(ST &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::MIN>(dst, src, dev_id, idx);
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_max(ST &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::MAX>(dst, src, dev_id, idx);
}

// template <int axis, bool assume_aligned, ReduceOp OP, ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
// __device__ static inline void reduce_op(PGL p_o, const ST &src, const COORD &idx) {
//     using T = typename ST::T;
//     using U = typename PGL::dtype;
//     const int row_stride = p_o.gl.template stride<axis>();

//     constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::T);
//     constexpr int memcpy_per_row = src.cols / elem_per_memcpy;
//     constexpr int dst_num_elem = src.height*src.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T>;
//     constexpr int total_calls = (dst_num_elem + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy);
//     constexpr bool needs_bounds_check = dst_num_elem % (N_THREADS*elem_per_memcpy);

//     coord<> coord = idx.template unit_coord<axis, 3>();
//     auto index = ((coord.b * p_o.gl.depth() + coord.d) * p_o.gl.rows() + coord.r) * p_o.gl.cols() + coord.c;
//     U* mc_ptr = p_o.mc_ptr + index;
//     uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
//     int laneid = threadIdx.x % N_THREADS;

//     for (int i = 0; i < total_calls; i++) {
//         int load_idx = i * N_THREADS + laneid;
//         int row = load_idx / memcpy_per_row;
//         int col = (load_idx*elem_per_memcpy) % src.cols;


//         if constexpr (needs_bounds_check) {
//             if (row >= src.rows) continue;
//         }

//         if constexpr (assume_aligned) {
//             float4 tmp;
//             move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));

//             T* dst = static_cast<T*>(mc_ptr) + row*row_stride + col;
//             multimem_reduce_op<T, OP>::apply_vec(dst, (T*)&tmp);
//         }
//         else {
//             if (row + coord.template dim<axis>() < p_o.gl.template shape<axis>()) {
//                 float4 tmp;
//                 move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));

//                 T* dst = static_cast<T*>(mc_ptr) + row*row_stride + col;
//                 multimem_reduce_op<T, OP>::apply_vec(dst, (T*)&tmp);
//             }
//         }
//     }
// }

// template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
// __device__ static inline void atomic_add(PGL p_o, const ST &src, const COORD &idx) {
//     reduce_op<2, false, ReduceOp::ADD>(p_o, src, idx);
// }

// template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
// __device__ static inline void atomic_min(PGL p_o, const ST &src, const COORD &idx) {
//     reduce_op<2, false, ReduceOp::MIN>(p_o, src, idx);
// }

// template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
// __device__ static inline void atomic_max(PGL p_o, const ST &src, const COORD &idx) {
//     reduce_op<2, false, ReduceOp::MAX>(p_o, src, idx);
// }


template <int axis, bool assume_aligned, ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void broadcast(const PGL &dst, const ST &src, int dev_id, const COORD &idx) {
    using T = typename ST::dtype;
    using U = typename PGL::dtype;
    const int row_stride = dst[dev_id].template stride<axis>();

    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = src.cols / elem_per_memcpy;
    constexpr int dst_num_elem = src.height*src.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T>;
    constexpr int total_calls = (dst_num_elem + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy);
    constexpr bool needs_bounds_check = dst_num_elem % (N_THREADS*elem_per_memcpy);

    coord<> coord = idx.template unit_coord<axis, 3>();
    auto index = ((coord.b * dst[dev_id].depth() + coord.d) * dst[dev_id].rows() + coord.r) * dst[dev_id].cols() + coord.c;
    U* mc_ptr = dst.mc_vas[dev_id] + index;
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    int laneid = threadIdx.x % N_THREADS;

    for (int i = 0; i < total_calls; i++) {
        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % src.cols;


        if constexpr (needs_bounds_check) {
            if (row >= src.rows) continue;
        }

        if constexpr (assume_aligned) {
            float4 tmp;
            move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
            T* ptr = static_cast<T*>(mc_ptr) + row*row_stride + col;
            move<float4>::stg((float4*)ptr, tmp);
        }
        else {
            if (row + coord.template dim<axis>() < dst[dev_id].template shape<axis>()) {
                float4 tmp;
                move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
                T* ptr = static_cast<T*>(mc_ptr) + row*row_stride + col;
                move<float4>::stg((float4*)ptr, tmp);
            }
        }
    }
}

template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void broadcast(const PGL &p_o, const ST &src, int dev_id, const COORD &idx) {
    broadcast<2, false>(p_o, src, dev_id, idx);
}

} // namespace kittens