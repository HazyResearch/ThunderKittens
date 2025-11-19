/**
 * @file
 * @brief Register vectors for computations on axes.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"
#include "rv_layout.cuh"

namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
/**
 * @namespace rt
 * 
 * @brief The namespace where concepts and abstract types for register vectors live.
 */
namespace rv {
/**
 * @brief A dummy type used to identify register vectors.
 * 
 * For a type to quack like an rv, it should define its identifier as ducks::rv::identifier.
 * If a type quacks like ducks::rv::identifier, it will be treated as an rv by compiler checks.
 */
struct identifier {};
/**
* @brief Concept for all register vectors.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as rv::identifier.
*/
template<typename T>
concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rv::identifier.

template<typename T> concept naive_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::naive>;
template<typename T> concept align_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::align>;
template<typename T> concept ortho_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::ortho>;
template<typename T> concept tile_layout  = align_layout<T> || ortho_layout<T>; // vector layouts for interacting with tiles.
}
}
/**
 * @brief Register vector structure.
 *
 * @tparam _T The packed data type used for the vector elements.
 * @tparam _outer_dim The size of the tile, in units of TILE_DIM (16).
 * @tparam _inner_dim This controls the layout of the tile in terms of which axis it maps on the register tile layout.
 *
 * Register vectors are used to accumulate and map values across tiles. You can do computation
 * on them directly if you want, but they're not designed to be maximally efficient vectors
 * as they have substantial duplication and strange layouts to help them work efficiently with
 * the register layouts used by the tensor cores. ThunderKittens wants you working with tiles
 * where possible!
 */
template<typename _T, size_t _length, ducks::rv_layout::all _layout=ducks::rv_layout::naive>
struct rv {
    using identifier = ducks::rv::identifier; ///< Type identifier for the rv structure.
    static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
    using layout = _layout;
    static constexpr bool is_naive = std::is_same_v<layout, ducks::rv_layout::naive>;
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = std::conditional_t<is_naive, T, T2>; ///< Data type of the vector elements

    static constexpr int length = _length; ///< Length in elements.
    static_assert(length % kittens::TILE_ROW_DIM<T> == 0, "Length must be divisible by the tile dimension");
    static constexpr int tiles  = _length / kittens::TILE_ROW_DIM<T>; ///< Length in subtiles, aliased for consistency with sv type
    static constexpr int inner_dim = layout::inner_dim; ///< Internal layout within a subtile. Either 1 or 2.
    static constexpr int outer_dim = is_naive ? (tiles+1)/2 : tiles; ///< Outer dim (also length in tiles)
    #if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for fp8");
    #endif

    dtype data[outer_dim][inner_dim]; ///< The actual register vector data.

    __device__ inline       dtype* operator[](size_t idx)       { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    __device__ inline const dtype* operator[](size_t idx) const { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    __device__ inline       dtype& operator[](int2 outin)       { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
    __device__ inline const dtype& operator[](int2 outin) const { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.

    __device__ inline void operator=(const T &value) {
        dtype value2;
        if constexpr(is_naive) {
            value2 = value;
        } else {
            value2 = base_types::packing<T>::pack(value);
        }
        #pragma unroll
        for(int i = 0; i < outer_dim; i++) {
            #pragma unroll
            for(int j = 0; j < inner_dim; j++) {
                data[i][j] = value2;
            }
        }
    }
    template<typename U>
    __device__ inline void operator=(const rv<U, length, layout> &other) {
        using U2 = base_types::packing<U>::packed_type;
        #pragma unroll
        for(int i = 0; i < outer_dim; i++) {
            #pragma unroll
            for(int j = 0; j < inner_dim; j++) {
                data[i][j] = base_types::convertor<T2, U2>::convert(other.data[i][j]);
            }
        }
    }
};

template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_fl = rv<float, _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_bf = rv<bf16,  _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_hf = rv<half,  _l, layout>;

/* ----------  PRINT FUNCTION  ---------- */

/**
 * @brief Print the contents of a register vector as a formatted output.
 * 
 * This function prints register vectors with information about their dimensions
 * and data contents, handling both packed and unpacked data types.
 * 
 * @param vec The register vector to print
 */
template<ducks::rv::all RV>
__device__ void print(const RV &vec) {
    if (laneid() == 0) { // Only first thread in warp prints
        printf("Block %d, Warp %d: Register Vector %d (Type: %s, Layout: %s) - Distributed View:\n", 
               blockIdx.x, threadIdx.x / WARP_THREADS, RV::length,
               std::is_same_v<typename RV::T, float> ? "float" :
               std::is_same_v<typename RV::T, bf16> ? "bf16" :
               std::is_same_v<typename RV::T, half> ? "half" :
               std::is_same_v<typename RV::T, fp8e8m0> ? "fp8e8m0" :
               std::is_same_v<typename RV::T, fp8e4m3> ? "fp8e4m3" :
               std::is_same_v<typename RV::T, fp8e5m2> ? "fp8e5m2" : "unknown",
               RV::is_naive ? "naive" : "tile");
        printf("Each thread holds %dx%d elements\n", RV::outer_dim, RV::inner_dim);
        printf("\n");
    }
    __syncwarp();
    
    // Each thread prints its own data
    for (int tid = 0; tid < WARP_THREADS; tid++) {
        if (laneid() == tid) {
            printf("Thread %2d: ", tid);
            
            // Print the vector data this thread holds
            for (int i = 0; i < RV::outer_dim; i++) {
                printf("Outer[%d]: ", i);
                for (int j = 0; j < RV::inner_dim; j++) {
                    auto value = vec.data[i][j];
                    
                    if constexpr (std::is_same_v<typename RV::dtype, typename RV::T>) {
                        // Unpacked type, print directly
                        if constexpr (std::is_same_v<typename RV::T, float>) {
                            printf("%.3f ", value);
                        } else if constexpr (std::is_same_v<typename RV::T, half>) {
                            printf("%.3f ", __half2float(value));
                        } else if constexpr (std::is_same_v<typename RV::T, bf16>) {
                            printf("%.3f ", __bfloat162float(value));
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e8m0>) {
                            printf("%.3f ", (float)value);
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e4m3>) {
                            printf("%.3f ", (float)value);
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e5m2>) {
                            printf("%.3f ", (float)value);
                        } else {
                            printf("%.3f ", (float)value);
                        }
                    } else {
                        // Packed type - check what type we're dealing with
                        if constexpr (std::is_same_v<typename RV::T, float>) {
                            printf("[%.3f, %.3f] ", value.x, value.y);
                        } else if constexpr (std::is_same_v<typename RV::T, bf16>) {
                            // Handle packed bf16_2 type
                            printf("[%.3f, %.3f] ", __bfloat162float(value.x), __bfloat162float(value.y));
                        } else if constexpr (std::is_same_v<typename RV::T, half>) {
                            // Handle packed half2 type
                            printf("[%.3f, %.3f] ", __half2float(value.x), __half2float(value.y));
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e8m0>) {
                            // Handle packed fp8e8m0_4 types
                            __nv_fp8_e8m0 *vals = reinterpret_cast<__nv_fp8_e8m0*>(const_cast<fp8e8m0_4*>(&value));
                            printf("[%.3f,%.3f,%.3f,%.3f] ", 
                                   (float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e4m3>) {
                            // Handle packed fp8e4m3_4 types  
                            __nv_fp8_e4m3 *vals = reinterpret_cast<__nv_fp8_e4m3*>(const_cast<fp8e4m3_4*>(&value));
                            printf("[%.3f,%.3f,%.3f,%.3f] ", 
                                   (float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e5m2>) {
                            // Handle packed fp8e5m2_4 types
                            __nv_fp8_e5m2 *vals = reinterpret_cast<__nv_fp8_e5m2*>(const_cast<fp8e5m2_4*>(&value));
                            printf("[%.3f,%.3f,%.3f,%.3f] ", 
                                   (float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
                        } else {
                            // Other packed types - print the raw packed value
                            printf("0x%x ", *(uint32_t*)&value);
                        }
                    }
                }
                printf(" ");
            }
            printf("\n");
        }
        __syncwarp(); // Ensure threads print in order
    }
    
    if (laneid() == 0) {
        printf("\n");
    }
}

} // namespace kittens
