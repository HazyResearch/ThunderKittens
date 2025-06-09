/**
 * @file
 * @brief Templated layouts for complex global memory.
 */
 
#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../../common/common.dp.hpp"
#include "../shared/cst.dp.hpp"
#include "gl.dp.hpp"
#include "util.dp.hpp"
#ifdef KITTENS_HOPPER
#include "tma.dp.hpp"
#endif

namespace kittens {

/* ----------  Global layout descriptor  ---------- */

namespace ducks {
namespace cgl {
struct identifier {};
}
}

// namespace detail {
// template<typename T> concept tile = ducks::cst::all<T> || ducks::crt::all<T>;
// template<typename T> concept vec  = ducks::csv::all<T> || ducks::crv::all<T>;
// }

template <kittens::ducks::gl::all _GL>
/*
DPCT1128:324: The type "filter_layout" is not device copyable for copy
constructor, move constructor, non trivially copyable field "real" and non
trivially copyable field "imag" breaking the device copyable requirement. It is
used in the SYCL kernel, please rewrite the code.
*/
/*
DPCT1128:326: The type "fft_layout" is not device copyable for copy constructor,
move constructor, non trivially copyable field "real" and non trivially copyable
field "imag" breaking the device copyable requirement. It is used in the SYCL
kernel, please rewrite the code.
*/
/*
DPCT1128:421: The type "complex_input_layout" is not device copyable for copy
constructor, move constructor, non trivially copyable field "real" and non
trivially copyable field "imag" breaking the device copyable requirement. It is
used in the SYCL kernel, please rewrite the code.
*/
/*
DPCT1128:423: The type "complex_filter_layout" is not device copyable for copy
constructor, move constructor, non trivially copyable field "real" and non
trivially copyable field "imag" breaking the device copyable requirement. It is
used in the SYCL kernel, please rewrite the code.
*/
/*
DPCT1128:424: The type "complex_fft_layout" is not device copyable for copy
constructor, move constructor, non trivially copyable field "real" and non
trivially copyable field "imag" breaking the device copyable requirement. It is
used in the SYCL kernel, please rewrite the code.
*/
struct cgl {
    using identifier = ducks::cgl::identifier;
    using component  = _GL;
    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype;
    component real, imag;
};

namespace ducks {
namespace cgl {
/**
* @brief Concept for all complex global layouts.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as ducks::cgl::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::cgl::identifier
}
}

}