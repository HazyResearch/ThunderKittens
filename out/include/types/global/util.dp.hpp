#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <type_traits>
#include <cstddef>
#include "../register/register.dp.hpp"

namespace kittens {
namespace ducks {
namespace gl {

template<int d> concept cdim = (d > 0); // represents a compile-time dimension
template<int d> concept rdim = (d == -1); // represents a runtime dimension
template<int _v> struct compiled_dim {
    static_assert(cdim<_v>, "Invalid compile-time dimension value");
    static constexpr size_t v = _v;
    inline compiled_dim(const std::nullptr_t &_) {}
    inline constexpr operator size_t() const { return v; }
};
struct runtime_dim {
    size_t v;
    inline runtime_dim(const size_t &_v) : v(_v) {}
    inline operator size_t() const { return v; }
};
template<int d> using make_dim_t = std::conditional_t<rdim<d>, runtime_dim, compiled_dim<d>>;
template<int d> using make_arg_t = std::conditional_t<rdim<d>, size_t, std::nullptr_t>; // we pass runtime dims as size_t, comptime dims as nullptr_t
}
}

namespace detail {
template<typename T> concept tile = ducks::st::all<T> || ducks::rt::all<T> || ducks::cst::all<T> || ducks::crt::all<T>;
template<typename T> concept vec  = ducks::sv::all<T> || ducks::rv::all<T> || ducks::csv::all<T> || ducks::crv::all<T>;
}

namespace ducks {
namespace coord {
struct identifier {};
}
}
template<typename _T=ducks::default_type> struct coord { // essentially a named int4 for tensor coordinates.
    using identifier = ducks::coord::identifier;
    using BASE = _T; // in units of what type?
    static_assert(std::is_same_v<BASE, ducks::default_type> || detail::tile<BASE> || detail::vec<BASE>); // ensure BASE is a valid type
    int b, d, r, c;
    inline coord(int _b, int _d, int _r, int _c) : b(_b), d(_d), r(_r), c(_c) {}
    inline coord(        int _d, int _r, int _c) : b( 0), d(_d), r(_r), c(_c) {}
    inline coord(                int _r, int _c) : b( 0), d( 0), r(_r), c(_c) {}
    inline coord(                        int _c) : b( 0), d( 0), r( 0), c(_c) {}
    inline coord(                              ) : b( 0), d( 0), r( 0), c( 0) {}
    template<typename U> inline coord(const coord<U> &other) : b(other.b), d(other.d), r(other.r), c(other.c) {}
    inline coord(const sycl::int4 &other)
        : b(other.x()), d(other.y()), r(other.z()), c(other.w()) {}
    inline operator sycl::int4() const { return sycl::int4(b, d, r, c); }
    template<int row_axis, int col_axis> inline coord<ducks::default_type> unit_coord() const {
        if constexpr (detail::tile<BASE>) {
            static_assert(row_axis != col_axis, "row and column axes must be different");
            static_assert(row_axis >= 0 && row_axis <= 3, "row axis must be between 0 and 3");
            static_assert(col_axis >= 0 && col_axis <= 3, "column axis must be between 0 and 3");
            static_assert(col_axis == 3, "for now, column axis must be 3");
            return coord<ducks::default_type>(
                row_axis == 0 ? b*BASE::rows : b,
                row_axis == 1 ? d*BASE::rows : d,
                row_axis == 2 ? r*BASE::rows : r,
                c*BASE::cols
            );
        }
        else if constexpr (detail::vec<BASE>) {
            static_assert(row_axis == -1, "row axis must be be -1 for a vector coordinate to be converted to a unit coordinate");
            static_assert(col_axis >= 0 && col_axis <= 3, "column axis must be between 0 and 3");
            static_assert(col_axis == 3, "for now, column axis must be 3");
            return coord<ducks::default_type>(b, d, r, c*BASE::length);
        }
        else {
            return coord<ducks::default_type>(*this);
        }
    }
    template<int axis> inline int dim() const {
        static_assert(axis >= 0 && axis <= 3, "axis must be between 0 and 3");
        if constexpr      (axis == 0) { return b; }
        else if constexpr (axis == 1) { return d; }
        else if constexpr (axis == 2) { return r; }
        else                          { return c; }
    }
};
namespace ducks {
namespace coord {
/**
* @brief Concept for all coordinate types.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as ducks::coord::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::coord::identifier
template<typename T> concept tile = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || detail::tile<typename T::BASE>);
template<typename T> concept vec  = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || detail::vec<typename T::BASE>);
}
}
}