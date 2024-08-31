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

template<ducks::st::all base, bool _use_tma=false>
struct gt {
    using T = base::T;
    static constexpr bool use_tma = _use_tma;
    template<int _b=-1, int _d=-1, int _r=-1, int _c=-1>
    struct l {
        using identifier = ducks::gt::l::identifier;
        using T = base::T;
        static constexpr bool use_tma = _use_tma;
        std::conditional_t<use_tma, CUtensorMap*, T*> data;
        ducks::g::make_dim_t<_b> batch;
        ducks::g::make_dim_t<_d> depth;
        ducks::g::make_dim_t<_r> rows;
        ducks::g::make_dim_t<_c> cols;
        __host__ inline l(T *_data,
                           ducks::g::make_arg_t<_b> _batch,
                           ducks::g::make_arg_t<_d> _depth,
                           ducks::g::make_arg_t<_r> _rows,
                           ducks::g::make_arg_t<_c> _cols) : batch(_batch), depth(_depth), rows(_rows), cols(_cols) {
            if constexpr (use_tma) {
                data = tma::detail::allocate_and_create_tensor_map<base>(_data, batch, depth, rows, cols);
            } else {
                data = _data;
            }
        }
        __host__ __device__ inline ~l() {
#ifdef __CUDA_ARCH__
            if constexpr (use_tma) {
                cudaFree(data);
            }
#endif
        }
    };
};

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