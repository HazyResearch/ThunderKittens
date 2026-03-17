#pragma once

namespace kittens {
namespace ducks {
namespace gl {

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

namespace detail {
template<typename T> concept tile = ducks::st::all<T> || ducks::rt::all<T> || ducks::cst::all<T> || ducks::crt::all<T>;
template<typename T> concept vec  = ducks::sv::all<T> || ducks::rv::all<T> || ducks::csv::all<T> || ducks::crv::all<T>;

#if (defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)) && !defined(KITTENS_NO_HOST)
namespace tma {

__host__ static inline std::string format_tma_error(
    const char* error_type,
    const char* error_string,
    int batch, int depth, int rows, int cols,
    CUtensorMap* tma_map,
    CUtensorMapDataType tma_format,
    uint32_t tma_dim,
    void* global_addr,
    const uint64_t* gmem_shape,
    const uint64_t* gmem_stride,
    const uint32_t* smem_shape,
    const uint32_t* smem_stride,
    size_t gmem_shape_size,
    size_t gmem_stride_size,
    size_t smem_shape_size,
    size_t smem_stride_size,
    CUtensorMapInterleave tma_interleave,
    CUtensorMapSwizzle tma_swizzle,
    CUtensorMapL2promotion tma_l2Promotion,
    CUtensorMapFloatOOBfill tma_oobFill,
    const std::string& extra_info = ""
) {
    std::string msg;
    msg += std::string("Error in TMA descriptor creation (") +
               (error_type ? error_type : "Unknown") + "): " +
               (error_string ? error_string : "Unknown CUDA error") + "\n";
    msg += "Parameters:\n";
    msg += "    batch: " + std::to_string(batch) + "\n";
    msg += "    depth: " + std::to_string(depth) + "\n";
    msg += "    rows: " + std::to_string(rows) + "\n";
    msg += "    cols: " + std::to_string(cols) + "\n";
    if (!extra_info.empty()) msg += "    " + extra_info + "\n";
    msg += "cuTensorMapEncodeTiled arguments:\n";
    msg += "    tma_map: " + std::to_string(reinterpret_cast<uintptr_t>(tma_map)) + "\n";
    msg += "    tma_format: " + std::to_string(tma_format) + "\n";
    msg += "    tma_dim: " + std::to_string(tma_dim) + "\n";
    msg += "    global_addr: " + std::to_string(reinterpret_cast<uintptr_t>(global_addr)) + "\n";
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, global_addr);
    msg += "    global_addr memory type: ";
    if (err == cudaSuccess) {
        if (attributes.type == cudaMemoryTypeDevice)       msg += "valid device memory\n";
        else if (attributes.type == cudaMemoryTypeHost)    msg += "host memory (invalid for TMA)\n";
        else if (attributes.type == cudaMemoryTypeManaged) msg += "managed memory\n";
        else msg += "unknown memory type\n";
    } else {
        msg += "unable to determine (error: " + std::string(cudaGetErrorString(err)) + ")\n";
    }
    msg += "    gmem_shape: " + std::to_string(reinterpret_cast<uintptr_t>(gmem_shape)) + " [";
    for (size_t i = 0; i < gmem_shape_size; ++i) {
        msg += std::to_string(gmem_shape[i]);
        if (i < gmem_shape_size - 1) msg += ", ";
    }
    msg += "]\n";
    msg += "    gmem_stride: " + std::to_string(reinterpret_cast<uintptr_t>(gmem_stride)) + " [";
    for (size_t i = 0; i < gmem_stride_size; ++i) {
        msg += std::to_string(gmem_stride[i]);
        if (i < gmem_stride_size - 1) msg += ", ";
    }
    msg += "]\n";
    msg += "    smem_shape: " + std::to_string(reinterpret_cast<uintptr_t>(smem_shape)) + " [";
    for (size_t i = 0; i < smem_shape_size; ++i) {
        msg += std::to_string(smem_shape[i]);
        if (i < smem_shape_size - 1) msg += ", ";
    }
    msg += "]\n";
    msg += "    smem_stride: " + std::to_string(reinterpret_cast<uintptr_t>(smem_stride)) + " [";
    for (size_t i = 0; i < smem_stride_size; ++i) {
        msg += std::to_string(smem_stride[i]);
        if (i < smem_stride_size - 1) msg += ", ";
    }
    msg += "]\n";
    msg += "    tma_interleave: " + std::to_string(tma_interleave) + "\n";
    msg += "    tma_swizzle: " + std::to_string(tma_swizzle) + "\n";
    msg += "    tma_l2Promotion: " + std::to_string(tma_l2Promotion) + "\n";
    msg += "    tma_oobFill: " + std::to_string(tma_oobFill) + "\n";
    return msg;
}

} // namespace tma
#endif // (defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)) && !defined(KITTENS_NO_HOST)
} // namespace detail

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
    __device__ inline coord(int _b, int _d, int _r, int _c) : b(_b), d(_d), r(_r), c(_c) {}
    __device__ inline coord(        int _d, int _r, int _c) : b( 0), d(_d), r(_r), c(_c) {}
    __device__ inline coord(                int _r, int _c) : b( 0), d( 0), r(_r), c(_c) {}
    __device__ inline coord(                        int _c) : b( 0), d( 0), r( 0), c(_c) {}
    __device__ inline coord(                              ) : b( 0), d( 0), r( 0), c( 0) {}
    template<typename U> __device__ inline coord(const coord<U> &other) : b(other.b), d(other.d), r(other.r), c(other.c) {}
    __device__ inline coord(const int4 &other)  : b(other.x), d(other.y), r(other.z), c(other.w) {}
    __device__ inline operator int4() const { return int4(b, d, r, c); }
    template<int row_axis, int col_axis> __device__ inline coord<ducks::default_type> unit_coord() const {
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
    template<int axis> __device__ inline int dim() const {
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