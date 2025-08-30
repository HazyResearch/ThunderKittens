/**
 * @file
 * @brief Functions for performing operations between PGLs and shared tiles.
 */

template <int axis, bool assume_aligned, ReduceOp OP, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void ld_reduce_op(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    using T = typename ST::dtype;
    using U = typename PGL::dtype;

    static_assert(std::is_same_v<U, kittens::bf16> || std::is_same_v<U, half> || std::is_same_v<U, float>, 
        "Unsupported type for ld_reduce_op");

    const int row_stride = src.template stride<axis>();
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy;
    constexpr int dst_num_elem = dst.height*dst.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T>;
    constexpr int total_calls = (dst_num_elem + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy);
    constexpr bool needs_bounds_check = dst_num_elem % (GROUP_THREADS*elem_per_memcpy);

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    U *src_mc_ptr = src.mc_ptr_at(unit_coord, dev_idx);
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int laneid = threadIdx.x % GROUP_THREADS;

    for (int i = 0; i < total_calls; i++) {
        int load_idx = i * GROUP_THREADS + laneid;
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % dst.cols;

        if constexpr (needs_bounds_check) {
            if (row >= dst.rows) continue;
        }

        if constexpr (assume_aligned) {
            float4 tmp;
            U* src_ptr = static_cast<U*>(src_mc_ptr) + row*row_stride + col;
            multimem_ld_reduce_op<T, OP>::apply_vec(&tmp, src_ptr);
            move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
        }
        else {
            if (row + unit_coord.template dim<axis>() < src.template shape<axis>()) {
                float4 tmp;
                U* src_ptr = static_cast<U*>(src_mc_ptr) + row*row_stride + col;
                multimem_ld_reduce_op<T, OP>::apply_vec(&tmp, src_ptr);
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
            }
        }
    }
}

/**
 * @brief Add data together across all devices in a PGL and store the result in a shared tile.
 * 
 * @tparam ST The shared tile type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination shared tile to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_add(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    ld_reduce_op<axis, assume_aligned, ReduceOp::ADD>(dst, src, dev_idx, idx);
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_add(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::ADD>(dst, src, dev_idx, idx);
}

/**
 * @brief Store the minimum value across all devices in a PGL into a shared tile.
 * 
 * @tparam ST The shared tile type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination shared tile to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_min(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    ld_reduce_op<axis, assume_aligned, ReduceOp::MIN>(dst, src, dev_idx, idx);
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_min(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::MIN>(dst, src, dev_idx, idx);
}

/**
 * @brief Store the maximum value across all devices in a PGL into a shared tile.
 * 
 * @tparam ST The shared tile type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination shared tile to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_max(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    ld_reduce_op<axis, assume_aligned, ReduceOp::MAX>(dst, src, dev_idx, idx);
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void all_reduce_max(ST &dst, const PGL &src, int dev_idx, const COORD &idx) {
    ld_reduce_op<2, false, ReduceOp::MAX>(dst, src, dev_idx, idx);
}

template <int axis, bool assume_aligned, ReduceOp OP, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void reduce_op(const PGL &dst, const ST &src, int dev_idx, const COORD &idx) {
    using T = typename ST::dtype;
    using U = typename PGL::dtype;
    const int row_stride = dst.template stride<axis>();

    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = src.cols / elem_per_memcpy;
    constexpr int dst_num_elem = src.height*src.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T>;
    constexpr int total_calls = (dst_num_elem + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy);
    constexpr bool needs_bounds_check = dst_num_elem % (GROUP_THREADS*elem_per_memcpy);

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    U *dst_mc_ptr = dst.mc_ptr_at(unit_coord, dev_idx);
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    int laneid = threadIdx.x % GROUP_THREADS;

    for (int i = 0; i < total_calls; i++) {
        int load_idx = i * GROUP_THREADS + laneid;
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % src.cols;

        if constexpr (needs_bounds_check) {
            if (row >= src.rows) continue;
        }

        if constexpr (assume_aligned) {
            float4 tmp;
            move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
            U* dst_ptr = static_cast<U*>(dst_mc_ptr) + row*row_stride + col;
            multimem_reduce_op<U, OP>::apply_vec(dst_ptr, (U*)&tmp);
        }
        else {
            if (row + unit_coord.template dim<axis>() < dst.template shape<axis>()) {
                float4 tmp;
                move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
                U* dst_ptr = static_cast<U*>(dst_mc_ptr) + row*row_stride + col;
                multimem_reduce_op<U, OP>::apply_vec(dst_ptr, (U*)&tmp);
            }
        }
    }
}

/**
 * @brief Add data from a shared tile to global memory for all devices in a PGL.
 * 
 * @tparam ST The shared tile type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination PGL to store the result.
 * @param[in] src The source shared tile to load data from
 */
template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void atomic_add(const PGL &dst, const ST &src, int dev_idx, const COORD &idx) {
    reduce_op<axis, assume_aligned, ReduceOp::ADD>(dst, src, dev_idx, idx);
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void atomic_add(const PGL &dst, const ST &src, int dev_idx, const COORD &idx) {
    reduce_op<2, false, ReduceOp::ADD>(dst, src, dev_idx, idx);
}

/**
 * @brief Store data from a shared tile to global memory for all devices in a PGL.
 * 
 * @tparam ST The shared tile type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination PGL to store the result.
 * @param[in] src The source shared tile to load data from
 */
template <int axis, bool assume_aligned, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void broadcast(const PGL &dst, const ST &src, int dev_idx, const COORD &idx) {
    using T = typename ST::dtype;
    using U = typename PGL::dtype;
    const int row_stride = dst.template stride<axis>();

    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = src.cols / elem_per_memcpy;
    constexpr int dst_num_elem = src.height*src.width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T>;
    constexpr int total_calls = (dst_num_elem + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy);
    constexpr bool needs_bounds_check = dst_num_elem % (GROUP_THREADS*elem_per_memcpy);

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    U *dst_mc_ptr = dst.mc_ptr_at(unit_coord, dev_idx);
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    int laneid = threadIdx.x % GROUP_THREADS;

    for (int i = 0; i < total_calls; i++) {
        int load_idx = i * GROUP_THREADS + laneid;
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % src.cols;

        if constexpr (needs_bounds_check) {
            if (row >= src.rows) continue;
        }

        if constexpr (assume_aligned) {
            float4 tmp;
            move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
            U* dst_ptr = static_cast<U*>(dst_mc_ptr) + row*row_stride + col;
            move<float4>::stg((float4*)dst_ptr, tmp);
        }
        else {
            if (row + unit_coord.template dim<axis>() < dst.template shape<axis>()) {
                float4 tmp;
                move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
                U* dst_ptr = static_cast<U*>(dst_mc_ptr) + row*row_stride + col;
                move<float4>::stg((float4*)dst_ptr, tmp);
            }
        }
    }
}

template <ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void broadcast(const PGL &dst, const ST &src, int dev_idx, const COORD &idx) {
    broadcast<2, false>(dst, src, dev_idx, idx);
}
