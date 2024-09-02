/**
 * @file
 * @brief Templated vector layouts for global memory.
 */
 
#pragma once

#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "util.cuh"

namespace kittens {

/* ----------  Global vector descriptor  ---------- */

namespace ducks {
namespace gv {
namespace l {
struct identifier {};
}
}
}

template<typename _T, int _tiles, bool _use_raw=true, bool _use_tma=false>
struct gv {
    template<int _b=-1, int _d=-1, int _r=-1, int _c=-1>
    struct l {
        using identifier = ducks::gv::l::identifier;

        using T     = base_types::packing<_T>::unpacked_type;
        using T2    = base_types::packing<_T>::packed_type;
        using dtype = T;
        using SV    = sv<T, _tiles>;
        static constexpr int base_tiles = _tiles;
        static constexpr int base_length = base_tiles * kittens::TILE_DIM;
        static constexpr bool raw = _use_raw;
        static constexpr bool tma = _use_tma;

        typename std::conditional_t<raw, T*, std::nullptr_t> raw_ptr = nullptr;
        typename std::conditional_t<tma, CUtensorMap*, std::nullptr_t> tma_ptr = nullptr;
        ducks::g::make_dim_t<_b> batch;
        ducks::g::make_dim_t<_d> depth;
        ducks::g::make_dim_t<_r> height;
        ducks::g::make_dim_t<_c> width;
        __host__ inline l(T *_data,
                           ducks::g::make_arg_t<_b> _batch,
                           ducks::g::make_arg_t<_d> _depth,
                           ducks::g::make_arg_t<_r> _height,
                           ducks::g::make_arg_t<_c> _width) : batch(_batch), depth(_depth), height(_height), width(_width) {
            if constexpr (raw) {
                raw_ptr = _data;
            }
            if constexpr (tma) {
                tma_ptr = tma::detail::allocate_and_create_tensor_map<SV>(_data, batch, depth, height, width);
            }
        }
        __host__ inline ~l() {
            if constexpr (tma) {
                cudaFree(tma_ptr);
            }
        }
    };
};
template<ducks::sv::all SV> using gv_sv = gv<typename SV::dtype, SV::tiles>;

namespace ducks {
namespace gv {
namespace l {
/**
* @brief Concept for all global vector layouts.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as ducks::gv::l::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::gv::l::identifier
}
}
}

}