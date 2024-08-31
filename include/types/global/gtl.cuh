/**
 * @file
 * @brief Templated tile layouts for global memory.
 */
 
#pragma once

#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "util.cuh"

namespace kittens {

/* ----------  Global tile descriptor  ---------- */

namespace ducks {
namespace gt {
namespace l {
struct identifier {};
}
}
}

template<typename _T, int _height, int _width, bool _use_raw=true, bool _use_tma=false>
struct gt {
    template<int _b=-1, int _d=-1, int _r=-1, int _c=-1>
    struct l {
        using identifier = ducks::gt::l::identifier;

        using T     = base_types::packing<_T>::unpacked_type;
        using T2    = base_types::packing<_T>::packed_type;
        using dtype = T;
        using ST    = st<T, _height, _width>;
        static constexpr int base_height = _height;
        static constexpr int base_width  = _width;
        static constexpr int base_rows   = _height * kittens::TILE_DIM;
        static constexpr int base_cols   = _width  * kittens::TILE_DIM;
        static constexpr bool raw = _use_raw;
        static constexpr bool tma = _use_tma;

        typename std::conditional_t<raw, T*, std::nullptr_t> raw_ptr = nullptr;
        typename std::conditional_t<tma, CUtensorMap*, std::nullptr_t> tma_ptr = nullptr;
        ducks::g::make_dim_t<_b> batch;
        ducks::g::make_dim_t<_d> depth;
        ducks::g::make_dim_t<_r> rows;
        ducks::g::make_dim_t<_c> cols;
        __host__ inline l(T *_data,
                           ducks::g::make_arg_t<_b> _batch,
                           ducks::g::make_arg_t<_d> _depth,
                           ducks::g::make_arg_t<_r> _rows,
                           ducks::g::make_arg_t<_c> _cols) : batch(_batch), depth(_depth), rows(_rows), cols(_cols) {
            if constexpr (raw) {
                raw_ptr = _data;
            }
            if constexpr (tma) {
                tma_ptr = tma::detail::allocate_and_create_tensor_map<ST>(_data, batch, depth, rows, cols);
            }
        }
        __host__ __device__ inline ~l() {
#ifndef __CUDA_ARCH__
            if constexpr (tma) {
                cudaFree(tma_ptr);
            }
#endif
        }
    };
};
template<ducks::st::all ST> using gt_st = gt<typename ST::dtype, ST::height, ST::width>;

namespace ducks {
namespace gt {
namespace l {
/**
* @brief Concept for all global tile layouts.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as ducks::gt::l::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::gt::l::identifier
}
}
}

}