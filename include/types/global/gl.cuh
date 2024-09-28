/**
 * @file
 * @brief Templated layouts for global memory.
 */
 
#pragma once

#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "util.cuh"
#ifdef KITTENS_HOPPER
#include "tma.cuh"
#endif

namespace kittens {

/* ----------   Associative dictionary for global layouts  ---------- */

namespace detail {
template<typename... Args>
struct descriptor_dict {
    __host__ descriptor_dict() {}
    template<typename T> __host__ descriptor_dict(T _, int b, int d, int r, int c) {}
    __host__ __device__ descriptor_dict(const descriptor_dict &other) {}
#ifdef KITTENS_HOPPER
    template<typename T> __device__ const CUtensorMap* get() const {
        static_assert(
            std::is_same_v<T, std::true_type> && std::is_same_v<T, std::false_type>,
            "SKILL ISSUE: Requested a TMA descriptor for a type not initialized in the global layout."
        );
    }
#endif
};

#ifdef KITTENS_HOPPER
template<typename T, typename... Args>
struct descriptor_dict<T, Args...> {
    static_assert(ducks::sv::all<T> || ducks::st::all<T>, "Must be a shared TK type to generate a TMA descriptor.");
    CUtensorMap tma_desc;
    descriptor_dict<Args...> other_descs;
    __host__ descriptor_dict() {}
    __host__ descriptor_dict(typename T::dtype *data, int b, int d, int r, int c): other_descs(data, b, d, r, c) {
        tma::detail::create_tensor_map<T>(&tma_desc, data, b, d, r, c);
    }
    __host__ __device__ inline descriptor_dict(const descriptor_dict &other) :
        tma_desc(other.tma_desc), other_descs(other.other_descs) {}
    template<typename U> __device__ inline const CUtensorMap* get() const {
        if constexpr (std::is_same_v<T, U>) { return &tma_desc; }
        else                                { return other_descs.template get<U>(); }
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

namespace detail {
template<typename T> concept tile = ducks::st::all<T> || ducks::rt::all<T> || ducks::cst::all<T> || ducks::crt::all<T>;
template<typename T> concept vec  = ducks::sv::all<T> || ducks::rv::all<T> || ducks::csv::all<T> || ducks::crv::all<T>;
}

template<typename _T, int b, int d, int r, int c, typename... TMA_Types>
struct gl {
    using identifier = ducks::gl::identifier;

    using T     = base_types::packing<_T>::unpacked_type;
    using T2    = base_types::packing<_T>::packed_type;
    using dtype = T;

    T* raw_ptr;

    ducks::g::make_dim_t<b> batch;
    ducks::g::make_dim_t<d> depth;
    ducks::g::make_dim_t<r> rows;
    ducks::g::make_dim_t<c> cols;

    detail::descriptor_dict<TMA_Types...> tma_descs;

    __host__ inline gl(T *_data,
                        ducks::g::make_arg_t<b> _batch,
                        ducks::g::make_arg_t<d> _depth,
                        ducks::g::make_arg_t<r> _rows,
                        ducks::g::make_arg_t<c> _cols) :
            raw_ptr(_data), batch(_batch), depth(_depth), rows(_rows), cols(_cols) {
        tma_descs = detail::descriptor_dict<TMA_Types...>(raw_ptr, batch, depth, rows, cols);
    }
    __host__ __device__ inline gl(const gl &other) :
            raw_ptr(other.raw_ptr), batch(other.batch), depth(other.depth), rows(other.rows), cols(other.cols), tma_descs(other.tma_descs) {}
    template<typename U> __device__ inline const CUtensorMap* get_tma() const {
        return tma_descs.template get<U>();
    }
    __device__ inline T& operator[](const coord &idx) {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c];
    }
    __device__ inline const T& operator[](const coord &idx) const {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c];
    }
    template<detail::tile TILE>__device__ inline T& get(const coord &idx) {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r*TILE::rows)*cols + idx.c*TILE::cols];
    }
    template<detail::tile TILE> __device__ inline const T& get(const coord &idx) const {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r*TILE::rows)*cols + idx.c*TILE::cols];
    }
    template<detail::vec VEC>__device__ inline T& get(const coord &idx) {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c*VEC::length];
    }
    template<detail::vec VEC>__device__ inline const T& get(const coord &idx) const {
        return raw_ptr[((idx.b*depth + idx.d)*rows + idx.r)*cols + idx.c*VEC::length];
    }
    __device__ inline size_t row_stride() const { return cols; }
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