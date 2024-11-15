#pragma once

#include <type_traits>
#include <cstddef>

namespace kittens {
namespace ducks {
namespace g {

template<int d> concept cdim = (d > 0); // represents a compile-time dimension
template<int d> concept rdim = (d == -1); // represents a runtime dimension
template<int _v> struct compiled_dim {
    static_assert(cdim<_v>, "Invalid compile-time dimension value");
    static constexpr size_t v = _v;
    __host__ __device__ inline compiled_dim(const std::nullptr_t &_) {}
    __host__ __device__ inline constexpr operator size_t() const { return v; }
};
struct runtime_dim {
    size_t v;
    __host__ __device__ inline runtime_dim(const size_t &_v) : v(_v) {}
    __host__ __device__ inline operator size_t() const { return v; }
};
template<int d> using make_dim_t = std::conditional_t<rdim<d>, runtime_dim, compiled_dim<d>>;
template<int d> using make_arg_t = std::conditional_t<rdim<d>, size_t, std::nullptr_t>; // we pass runtime dims as size_t, comptime dims as nullptr_t
}
}

namespace ducks {
namespace coord {
struct identifier {};
}
}
template<typename _T=ducks::default_type> struct coord { // essentially a named int4 for tensor coordinates.
    using identifier = ducks::coord::identifier;
    using BASE = _T; // in units of what type?
    int b, d, r, c;
    __device__ inline coord(int _b, int _d, int _r, int _c) : b(_b), d(_d), r(_r), c(_c) {}
    __device__ inline coord(        int _d, int _r, int _c) : b( 0), d(_d), r(_r), c(_c) {}
    __device__ inline coord(                int _r, int _c) : b( 0), d( 0), r(_r), c(_c) {}
    __device__ inline coord(                        int _c) : b( 0), d( 0), r( 0), c(_c) {}
    __device__ inline coord(                              ) : b( 0), d( 0), r( 0), c( 0) {}
    template<typename U> __device__ inline coord(const coord<U> &other) : b(other.b), d(other.d), r(other.r), c(other.c) {}
    __device__ inline coord(const int4 &other)  : b(other.x), d(other.y), r(other.z), c(other.w) {}
    __device__ inline operator int4() const { return int4(b, d, r, c); }
    template<int axis> __device__ inline int dim() const { return (axis == 0) ? b : (axis == 1) ? d : (axis == 2) ? r : c; }
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
template<typename T> concept st  = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || ducks::st::all <typename T::BASE>);
template<typename T> concept rt  = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || ducks::rt::all <typename T::BASE>);
template<typename T> concept cst = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || ducks::cst::all<typename T::BASE>);
template<typename T> concept crt = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || ducks::crt::all<typename T::BASE>);
template<typename T> concept sv  = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || ducks::sv::all <typename T::BASE>);
template<typename T> concept rv  = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || ducks::rv::all <typename T::BASE>);
template<typename T> concept csv = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || ducks::csv::all<typename T::BASE>);
template<typename T> concept crv = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || ducks::crv::all<typename T::BASE>);
}
}
}