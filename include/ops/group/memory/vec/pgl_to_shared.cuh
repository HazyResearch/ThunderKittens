/**
 * @file
 * @brief Functions for a kittens::group to collaboratively perform operations between pgls and shared vectors.
 */

template<ReduceOp OP, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void ld_reduce_op(SV &dst, const PGL &src, int dev_id, const COORD &idx) {
    using U = typename PGL::dtype;
    using T = typename SV::dtype;
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = dst.length / elem_per_transfer; // guaranteed to divide

    static_assert(std::is_same_v<U, kittens::bf16> || std::is_same_v<U, half> || std::is_same_v<U, float>, 
        "Unsupported type for ld_reduce_op");

    U *src_mc_ptr = src.mc_ptr_at(idx.template unit_coord<-1, 3>(), dev_id);
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    #pragma unroll
    for (uint32_t i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            float4 tmp;
            U *src_ptr = static_cast<U*>(src_mc_ptr) + i*elem_per_transfer;
            multimem_ld_reduce_op<T, OP>::apply_vec(&tmp, src_ptr);
            move<float4>::sts(dst_ptr + sizeof(typename SV::dtype)*i*elem_per_transfer, tmp);
        }
    }
}

/**
 * @brief Collaboratively adds data together across all devices and stores the result in a shared vector.
 *
 * @tparam SV The shared vector type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination shared vector to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void all_reduce_add(SV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<ReduceOp::ADD>(dst, src, dev_id, idx);
}

/**
 * @brief Collaboratively store the minimum value across all devices in a PGL into a shared vector.
 *
 * @tparam SV The shared vector type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination shared vector to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void all_reduce_min(SV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<ReduceOp::MIN>(dst, src, dev_id, idx);
}

/**
 * @brief Collaboratively store the maximum value across all devices in a PGL into a shared vector.
 *
 * @tparam SV The shared vector type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination shared vector to store the result.
 * @param[in] src The source PGL to load data across devices from
 */
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void all_reduce_max(SV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<ReduceOp::MAX>(dst, src, dev_id, idx);
}

template<ReduceOp OP, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void reduce_op(const PGL &dst, const SV &src, int dev_id, const COORD &idx) {
    using U = typename PGL::dtype;
    static_assert(std::is_same_v<U, kittens::bf16> || std::is_same_v<U, half> || std::is_same_v<U, float>, 
        "Unsupported type for ld_reduce_op");

    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = src.length / elem_per_transfer; // guaranteed to divide
    U *dst_mc_ptr = dst.mc_ptr_at(idx.template unit_coord<-1, 3>(), dev_id);
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    #pragma unroll
    for (uint32_t i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < src.length) {
            float4 tmp;
            move<float4>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*i*elem_per_transfer);
            U* dst_ptr = static_cast<U*>(dst_mc_ptr) + i*elem_per_transfer;
            multimem_reduce_op<U, OP>::apply_vec(dst_ptr, (U*)&tmp);
        }
    }
}

/**
 * @brief Collaboratively adds data from a shared vector to global memory for all devices in a PGL
 *
 * @tparam SV The shared vector type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination PGL to store the result.
 * @param[in] src The source shared vector to load data from.
 */
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void atomic_add(const PGL &dst, const SV &src, int dev_id, const COORD &idx) {
    reduce_op<ReduceOp::ADD>(dst, src, dev_id, idx);
}

/**
 * @brief Collaboratively stores data from a shared vector to global memory for all devices in a PGL
 *
 * @tparam SV The shared vector type.
 * @tparam PGL The parallel global layout type.
 * @param[out] dst The destination PGL to store the result.
 * @param[in] src The source shared vector to load data from.
 */
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void broadcast(const PGL &dst, const SV &src, int dev_id, const COORD &idx) {
    using U = typename PGL::dtype;
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = src.length / elem_per_transfer; // guaranteed to divide
    U *dst_mc_ptr = dst.mc_ptr_at(idx.template unit_coord<-1, 3>(), dev_id);
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    #pragma unroll
    for (uint32_t i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < src.length) {
            float4 tmp;
            move<float4>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*i*elem_per_transfer);
            move<float4>::stg((float4*)&dst_mc_ptr[i*elem_per_transfer], tmp);
        }
    }
}