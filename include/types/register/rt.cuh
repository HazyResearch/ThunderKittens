/**
 * @file
 * @brief The main ThunderKittens register tile struct, where most computation happens.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"

#include "rt_layout.cuh"
#include "rt_base.cuh"
#include "rv.cuh"

namespace kittens {

/* ----------  MAIN TILE STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
/**
 * @namespace rt
 * 
 * @brief The namespace where concepts and abstract types for register tiles live.
 */
namespace rt {
/**
 * @brief A dummy type used to identify register tiles.
 * 
 * For a type to quack like an rt, it should define its identifier as ducks::rt::identifier.
 * If a type quacks like ducks::rt::identifier, it will be treated as an rt by compiler checks.
 */
struct identifier {};
/**
* @brief Concept for all register tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as rt::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rt::identifier
/**
* @brief Concept for register tiles with row layout.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a register tile.
* - T has an internal type layout that is ducks::rt_layout::row.
*/
template<typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;
/**
* @brief Concept for register tiles with col layout.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a register tile.
* - T has an internal type layout that is ducks::rt_layout::col.
*/
template<typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;
} // namespace rt
} // namespace ducks

/**
 * @brief Main tile structure for manipulating data in registers.
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _height The height of the tile in terms of the number of subtiles.
 * @tparam _width The width of the tile in terms of the number of subtiles.
 * @tparam _layout The layout of the internal base tiles, either row-major or column-major.
 *
 * This structure is designed to handle matrix tiles in a flexible manner, allowing
 * for operations on tiles that are composed of smaller subtiles. It supports both
 * row-major and column-major layouts and includes helper structs for type inference
 * in vector maps.
 * 
 * In general, you probably want a row-major tile, unless you specifically want to call mma
 */
template<typename _T, int _rows, int _cols, ducks::rt_layout::all _layout=ducks::rt_layout::row>
struct rt {
    using identifier = ducks::rt::identifier; ///< Type identifier for the rt structure.
    using layout = _layout; ///< Layout of the matrix tile.
    static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2; ///< Data type of the matrix elements

    static constexpr int rows                = _rows; ///< Total number of rows.
    static_assert(rows % rt_base<T, layout>::tile_size_row == 0, "Rows must be divisible by the tile size");
    static constexpr int cols                = _cols; ///< Total number of columns.
    static_assert(cols % rt_base<T, layout>::tile_size_col == 0, "Columns must be divisible by the tile size");
    static constexpr int height              = rows / rt_base<T, layout>::tile_size_row; ///< Height in subtiles.
    static constexpr int width               = cols / rt_base<T, layout>::tile_size_col; ///< Width in subtiles.
    static constexpr int tile_size_row        = rt_base<T, layout>::tile_size_row;        ///< Size of the base tile.
    static constexpr int tile_size_col        = rt_base<T, layout>::tile_size_col;        ///< Size of the base tile.
    static constexpr int num_elements        = rt_base<T, layout>::num_elements        * width * height; ///< Total number of elements.
    static constexpr int elements_per_thread = rt_base<T, layout>::elements_per_thread * width * height; ///< Elements handled per thread.
    static constexpr int packed_per_thread   = rt_base<T, layout>::packed_per_thread   * width * height; ///< Packed elements per thread.
    static constexpr int packed_per_tile     = rt_base<T, layout>::packed_per_thread; ///< Packed elements per tile.

    rt_base<T, layout> tiles[height][width]; ///< The actual storage for the matrix tile, organized in subtiles.

    using row_vec = rv<T, cols, typename rt_base<T, layout>::row_vec_layout>; ///< A type representing a column vector for this tile.
    using col_vec = rv<T, rows, typename rt_base<T, layout>::col_vec_layout>; ///< A type representing a column vector for this tile.

    __device__ inline void operator=(const T &value) {
        T2 value2 = base_types::packing<T>::pack(value);
        #pragma unroll
        for(int i = 0; i < height; i++) {
            #pragma unroll
            for(int j = 0; j < width; j++) {
                #pragma unroll
                for(int k = 0; k < packed_per_tile; k++) {
                    tiles[i][j].data[k] = value2;
                }
            }
        }
    }
    template<typename U>
    __device__ inline void operator=(const rt<U, rows, cols, layout> &other) {
        using U2 = base_types::packing<U>::packed_type;
        #pragma unroll
        for(int i = 0; i < height; i++) {
            #pragma unroll
            for(int j = 0; j < width; j++) {
                #pragma unroll
                for(int k = 0; k < packed_per_tile; k++) {
                    tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(other.tiles[i][j].data[k]);
                }
            }
        }
    }
};


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// layout and type wrappers

template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl = rt<float, _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf = rt<bf16,  _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_hf = rt<half,  _r, _c, layout>;
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp8e4m3 = rt<fp8e4m3,  _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp8e5m2 = rt<fp8e5m2,  _r, _c, layout>;
#endif
#if defined(KITTENS_BLACKWELL)
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp8e8m0 = rt<fp8e8m0,  _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp4e2m1_2 = rt<fp4e2m1_2,  _r, _c, layout>;
#endif

/* ----------  PRINTOUTS  ---------- */

/**
 * @brief Get a readable type name for register tiles
 */
template<typename T, int rows, int cols>
__device__ constexpr const char* get_rt_type_name() {
    if constexpr (std::is_same_v<T, float>) {
        return "rt_fl";
    } else if constexpr (std::is_same_v<T, half>) {
        return "rt_hf";
    } else if constexpr (std::is_same_v<T, bf16>) {
        return "rt_bf";
#if defined(KITTENS_BLACKWELL)
    } else if constexpr (std::is_same_v<T, fp4e2m1_2>) {
        return "rt_fp4_e2m1_2";
    } else if constexpr (std::is_same_v<T, fp8e8m0>) {
        return "rt_fp8_e8m0";
#endif
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    } else if constexpr (std::is_same_v<T, fp8e4m3>) {
        return "rt_fp8_e4m3";
    } else if constexpr (std::is_same_v<T, fp8e5m2>) {
        return "rt_fp8_e5m2";
#endif
    } else {
        return "rt_unknown";
    }
}

/**
 * @brief Print the contents of a register tile as a formatted table.
 * 
 * This function should be called by all threads in the warp, but only
 * the first thread (laneid() == 0) will coordinate the printing.
 * It shows what each thread holds in its portion of the distributed tile.
 * 
 * @param tile The register tile to print
 */
template<ducks::rt::all RT>
__device__ inline void print(const RT& tile) {
    if (laneid() == 0) { // Only first thread in warp prints
        printf("Block %d, Warp %d: Register Tile %dx%d (Type: %s<%d,%d>) - Distributed View:\n", 
               blockIdx.x, threadIdx.x / WARP_THREADS, RT::rows, RT::cols, 
               get_rt_type_name<typename RT::T, RT::rows, RT::cols>(), RT::rows, RT::cols);
        printf("Each thread holds %d elements (%d packed)\n", 
               RT::elements_per_thread, RT::packed_per_thread);
        printf("\n");
    }
    __syncwarp();
    
    // Each thread prints its own data
    for (int tid = 0; tid < WARP_THREADS; tid++) {
        if (laneid() == tid) {
            printf("Thread %2d: ", tid);
            
            // Print the packed data this thread holds
            for (int i = 0; i < RT::height; i++) {
                for (int j = 0; j < RT::width; j++) {
                    printf("Subtile[%d][%d]: ", i, j);
                    for (int k = 0; k < RT::packed_per_tile && k < 4; k++) { // Limit to first 4 elements to avoid too much output
                        auto packed_val = tile.tiles[i][j].data[k];
                        
                        if constexpr (std::is_same_v<typename RT::dtype, typename RT::T>) {
                            // Unpacked type, print directly
                            if constexpr (std::is_same_v<typename RT::T, float>) {
                                printf("%.3f ", packed_val);
                            } else if constexpr (std::is_same_v<typename RT::T, half>) {
                                printf("%.3f ", __half2float(packed_val));
                            } else if constexpr (std::is_same_v<typename RT::T, bf16>) {
                                printf("%.3f ", __bfloat162float(packed_val));
#if defined(KITTENS_BLACKWELL)
                            } else if constexpr (std::is_same_v<typename RT::T, fp4e2m1>) {
                                printf("%.3f ", (float)packed_val);
#endif
                            } else {
                                printf("%.3f ", (float)packed_val);
                            }
                        } else {
                            // Packed type - check what type we're dealing with
                            if constexpr (std::is_same_v<typename RT::T, float>) {
                                printf("[%.3f, %.3f] ", packed_val.x, packed_val.y);
                            } else if constexpr (std::is_same_v<typename RT::T, bf16>) {
                                // Handle packed bf16_2 type
                                printf("[%.3f, %.3f] ", __bfloat162float(packed_val.x), __bfloat162float(packed_val.y));
#if defined(KITTENS_BLACKWELL)
                            } else if constexpr (std::is_same_v<typename RT::T, fp8e8m0>) {
                                // Extract the 4 individual fp8e8m0 values from the packed fp8e8m0_4
                                __nv_fp8_e8m0 *vals = reinterpret_cast<__nv_fp8_e8m0*>(const_cast<fp8e8m0_4*>(&packed_val));
                                printf("[%.3f,%.3f,%.3f,%.3f] ", 
                                       (float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
                            } else if constexpr (std::is_same_v<typename RT::T, fp4e2m1>) {
                                // Handle packed fp4e2m1_4 types (4 fp4 values packed together)
                                uint8_t *vals = reinterpret_cast<uint8_t*>(const_cast<fp4e2m1_4*>(&packed_val));
                                printf("[%.3f,%.3f,%.3f,%.3f] ", (float)fp4e2m1(vals[0] & 0xF), (float)fp4e2m1((vals[0] >> 4) & 0xF), (float)fp4e2m1(vals[1] & 0xF), (float)fp4e2m1((vals[1] >> 4) & 0xF));
#endif
                            } else {
                                // Other packed types - print the raw packed value
                                printf("0x%x ", *(uint32_t*)&packed_val);
                            }
                        }
                    }
                    if (RT::packed_per_tile > 4) printf("... ");
                }
            }
            printf("\n");
        }
        __syncwarp(); // Ensure threads print in order
    }
    
    if (laneid() == 0) {
        printf("\n");
    }
    __syncwarp();
}

} // namespace kittens
