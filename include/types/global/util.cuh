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

using index = int4;
}