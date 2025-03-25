#pragma once

#include <cuda.h>
#include "../util/reduce.cuh"


namespace kittens {

template <int axis, bool assume_aligned, ReduceOp OP, ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void ld_reduce_op(PGL p_o, const ST &src, const COORD &idx) {
    using T = typename ST::T;
    using U = typename PGL::dtype;
    const int row_stride = p_o.gl.template stride<axis>();

    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::T);
    constexpr int memcpy_per_row = src.cols / elem_per_memcpy;
    constexpr int dst_num_elem = src.height*src.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T>;
    constexpr int total_calls = (dst_num_elem + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy);
    constexpr bool needs_bounds_check = dst_num_elem % (N_THREADS*elem_per_memcpy);

    coord<> coord = idx.template unit_coord<axis, 3>();
    auto index = ((coord.b * p_o.gl.depth() + coord.d) * p_o.gl.rows() + coord.r) * p_o.gl.cols() + coord.c;
    U* mc_ptr = p_o.mc_ptr + index;
    int laneid = threadIdx.x % N_THREADS;

    for (int i = 0; i < total_calls; i++) {
        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % src.cols;


        if constexpr (needs_bounds_check) {
            if (row >= src.rows) continue;
        }

        if constexpr (assume_aligned) {
            float4 val;
            T* ptr = static_cast<T*>(mc_ptr) + row*row_stride + col;
            multimem_ld_reduce_op<T, OP>::apply_vec(&val, ptr);
            move<float4>::stg((float4*)ptr, val);
        }
        else {
            if (row + coord.template dim<axis>() < p_o.gl.template shape<axis>()) {
                float4 val;
                T* ptr = static_cast<T*>(mc_ptr) + row*row_stride + col;
                multimem_ld_reduce_op<T, OP>::apply_vec(&val, ptr);
                move<float4>::stg((float4*)ptr, val);
            }
        }
    }
}

template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_add(PGL p_o, const ST &src, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::ADD>(p_o, src, idx);
}

template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_min(PGL p_o, const ST &src, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::MIN>(p_o, src, idx);
}

template <ducks::pgl::all PGL, ducks::st::all ST, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_max(PGL p_o, const ST &src, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::MAX>(p_o, src, idx);
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

} // namespace kittens