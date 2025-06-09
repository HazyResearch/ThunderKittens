#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared vectors from and
 * storing to global memory.
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
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
static inline void load(SV &dst, const GL &src, const COORD &idx) {
    constexpr uint32_t elem_per_transfer =
        sizeof(sycl::float4) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = dst.length / elem_per_transfer; // guaranteed to divide
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    #pragma unroll
    for (uint32_t i =
             sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
                 2) %
             GROUP_THREADS;
         i < total_calls; i += GROUP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            sycl::float4 tmp;
            move<sycl::float4>::ldg(
                tmp, (sycl::float4 *)&src_ptr[i * elem_per_transfer]);
            move<sycl::float4>::sts(dst_ptr + sizeof(typename SV::dtype) * i *
                                                  elem_per_transfer,
                                    tmp);
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
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
static inline void store(GL &dst, const SV &src, const COORD &idx) {
    constexpr uint32_t elem_per_transfer =
        sizeof(sycl::float4) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = src.length / elem_per_transfer; // guaranteed to divide
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[(idx.template unit_coord<-1, 3>())];
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    #pragma unroll
    for (uint32_t i =
             sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
                 2) %
             GROUP_THREADS;
         i < total_calls; i += GROUP_THREADS) {
        if(i * elem_per_transfer < src.length) {
            sycl::float4 tmp;
            move<sycl::float4>::lds(tmp, src_ptr + sizeof(typename SV::dtype) *
                                                       i * elem_per_transfer);
            move<sycl::float4>::stg(
                (sycl::float4 *)&dst_ptr[i * elem_per_transfer], tmp);
        }
    }
}

template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
static inline void load_async(SV &dst, const GL &src, const COORD &idx) {
    constexpr uint32_t elem_per_transfer =
        sizeof(sycl::float4) / sizeof(typename SV::dtype);
    constexpr uint32_t total_calls = dst.length / elem_per_transfer; // guaranteed to divide
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    #pragma unroll
    for (uint32_t i =
             sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
                 2) %
             GROUP_THREADS;
         i < total_calls; i += GROUP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            /*
            DPCT1053:284: Migration of device assembly code is not supported.
            */
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                    dst_ptr + (uint32_t)sizeof(typename SV::dtype) * i *
                                  elem_per_transfer),
                "l"((uint64_t)&src_ptr[i * elem_per_transfer])
                : "memory");
        }
    }
    /*
    DPCT1026:283: The call to "cp.async.commit_group;
" was removed because current "cp.async" is migrated to synchronous copy
operation. You may need to adjust the code to tune the performance.
    */
    
}