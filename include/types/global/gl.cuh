/**
 * @file
 * @brief Templated layouts for global memory.
 */
 
#pragma once

#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "util.cuh"

namespace kittens {

/* ----------   Associative dictionary for global layouts  ---------- */

namespace detail {
template<typename... Args>
struct descriptor_dict {
    __host__ descriptor_dict(typename T::dtype *data, const int4 &shape){}
    __host__ __device__ descriptor_dict(const descriptor_dict &other) {}
#ifdef KITTENS_HOPPER
    __device__ template<typename U> CUtensorMap* get() {
        static_assert(
            std::is_same_v<T, std::true_type> && std::is_same_v<T, std::false_type>,
            "SKILL ISSUE: Requested a TMA descriptor for a type not initialized in the global layout."
        );
    }
#endif
    __host__ void cleanup() {}
};

#ifdef KITTENS_HOPPER
template<typename T, typename... Args>
struct descriptor_dict<T, Args...> {
    static_assert(ducks::sv::all<T> || ducks::st::all<T>, "Must be a shared TK type to generate a TMA descriptor.");
    CUtensorMap* tma_desc;
    dict<Args...> other_descs;
    __host__ descriptor_dict(typename T::dtype *data, int b, int d, int r, int c): other_descs<Args...>(data, b, d, r, c) {
        tma_desc = allocate_and_create_tensor_map<T>(data, b, d, r, c);
    }
    __host__ __device__ inline descriptor_dict(const descriptor_dict &other) :
        tma_desc(other.tma_desc), other_descs(other.other_descs) {}
    __device__ template<typename U> inline CUtensorMap* get() {
        if constexpr (std::is_same_v<T, U>) { return tma_desc; }
        else                                { return other_descs.template get<U>(); }
    }
    __host__ inline void cleanup() {
        if(tma_desc != nullptr) {
            cudaFree(tma_desc);
            tma_desc = nullptr;
            d.cleanup();
        }
    }
};
#endif
}

/* ----------  Global layout descriptor  ---------- */

namespace ducks {
namespace gl {
struct identifier {};
}
}

template<typename _T, int b, int d, int r, int c, typename... TMA_Types>
struct gl {
    using identifier = ducks::gl::identifier;

    using T     = base_types::packing<_T>::unpacked_type;
    using T2    = base_types::packing<_T>::packed_type;
    using dtype = T;

    T* raw_ptr;
    detail::descriptor_dict<TMA_Types...> tma_descs;

    ducks::g::make_dim_t<b> batch;
    ducks::g::make_dim_t<d> depth;
    ducks::g::make_dim_t<r> rows;
    ducks::g::make_dim_t<c> cols;
    __host__ inline gl(T *_data,
                        ducks::g::make_arg_t<b> _batch,
                        ducks::g::make_arg_t<d> _depth,
                        ducks::g::make_arg_t<r> _rows,
                        ducks::g::make_arg_t<c> _cols) :
        batch(_batch), depth(_depth), rows(_rows), cols(_cols), tma_descs(_data, _batch, _depth, _rows, _cols) {}
    __host__ __device__ inline gl(const l &other) :
        batch(other.batch), depth(other.depth), rows(other.rows), cols(other.cols), raw_ptr(other.raw_ptr), tma_descs(other.tma_descs) {}
    __host__ inline void cleanup() {
        tma_descs.cleanup();
    }
    template<typename U> __device__ inline CUtensorMap* get() {
        return tma_descs.template get<U>();
    }
    __device__ inline T& operator[](const index &idx) {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c];
    }
    __device__ inline const T& operator[](const index &idx) const {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c];
    }
    template<int axis> __device__ inline size_t stride() const { 
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if      constexpr (axis==0) { return depth*rows*cols; }
        else if constexpr (axis==1) { return rows*cols; }
        else if constexpr (axis==2) { return cols; }
        else if constexpr (axis==3) { return 1; }
    }
};

namespace ducks {
namespace gl {
/**
* @brief Concept for all global layouts.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as ducks::gl::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::gl::identifier
}
}

}