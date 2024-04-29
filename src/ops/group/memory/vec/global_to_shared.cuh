/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared vectors from and storing to global memory. 
 */

/**
 * @brief Loads data from global memory into shared memory vector.
 * 
 * This function loads data from a global memory location pointed to by `src` into a shared memory vector `dst`.
 * It calculates the number of elements that can be transferred in one operation based on the size ratio of `float4` to the data type of `SV`.
 * The function ensures coalesced memory access and efficient use of bandwidth by dividing the work among threads in a warp.
 * 
 * @tparam SV Shared vector type, must satisfy ducks::sv::all concept.
 * @param dst Reference to the shared vector where the data will be loaded.
 * @param src Pointer to the global memory location from where the data will be loaded.
 */
template<ducks::sv::all SV>
__device__ static inline void load(SV &dst, const typename SV::dtype *src) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = dst.length / elem_per_transfer; // guaranteed to divide
    __syncwarp();
    #pragma unroll
    for(int i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < dst.length)
            *(float4*)&dst[i*elem_per_transfer] = *(float4*)&src[i*elem_per_transfer];
    }
}

template<ducks::sv::all SV>
__device__ static inline void load_async(SV &dst, const typename SV::dtype *src, cuda::barrier<cuda::thread_scope_block> &barrier) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = dst.length / elem_per_transfer; // guaranteed to divide
    __syncwarp();
    #pragma unroll
    for(int i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            cuda::memcpy_async(
                (void*)&dst[i*elem_per_transfer], 
                (void*)&src[i*elem_per_transfer], 
                cuda::aligned_size_t<16>(sizeof(float4)), 
                barrier
            );
        }
    }
}

/**
 * @brief Stores data from a shared memory vector to global memory.
 * 
 * This function stores data from a shared memory vector `src` to a global memory location pointed to by `dst`.
 * Similar to the load function, it calculates the number of elements that can be transferred in one operation based on the size ratio of `float4` to the data type of `SV`.
 * The function ensures coalesced memory access and efficient use of bandwidth by dividing the work among threads in a warp.
 * 
 * @tparam SV Shared vector type, must satisfy ducks::sv::all concept.
 * @param dst Pointer to the global memory location where the data will be stored.
 * @param src Reference to the shared vector from where the data will be stored.
 */
template<ducks::sv::all SV>
__device__ static inline void store(typename SV::dtype *dst, const SV &src) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = src.length / elem_per_transfer; // guaranteed to divide
    __syncwarp();
    #pragma unroll
    for(int i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < src.length)
            *(float4*)&dst[i*elem_per_transfer] = *(float4*)&src[i*elem_per_transfer]; // lmao it's identical
    }
}