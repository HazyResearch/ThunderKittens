/**
 * @file
 * @brief Conversions between data layouts and types for register tiles.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/* ----------  LAYOUT SWAPS  ---------- */

/**
 * @brief Perform a matrix transpose on a block of 8 bf16_2 elements using inline assembly.
 *
 * This low-level operation is utilized by higher-level layout swap functions to transpose
 * the layout of bf16_2 elements within a register tile. The function leverages inline PTX
 * assembly to efficiently swap the layout of the given block.
 *
 * @param[out] dst A reference to the destination bf16_2 element where the transposed result is stored.
 * @param[in] src A reference to the source bf16_2 element to be transposed.
 */
__device__ inline void swap_layout_8(bf16_2 &dst, const bf16_2 &src) {
    asm volatile (
        "movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
    :   "+r"(*(uint32_t*)(&dst))
    :   "r"(*(uint32_t*)(&src))
    );
}
/**
 * @brief Swaps the layout of a register base tile.
 *
 * This function swaps the layout of a register base tile by performing a series of layout swaps
 * on its constituent bf16_2 elements. It is used to change the data layout within a register tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register base tile where the result will be stored.
 * @param src[in] Reference to the source register base tile to be swapped.
 */
template<typename T2, ducks::rt_layout::all layout>
__device__ inline void swap_layout(rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt_base<T2, layout> &src) {
    swap_layout_8(dst.data[0], src.data[0]);
    // technically this swap can be eliminated if we simply reinterpret the layout of the registers
    // everywhere else in the code, but that feels... very likely to cause bugs and not worth it. 
    T2 data1_cache = src.data[1]; // important for swap!
    swap_layout_8(dst.data[1], src.data[2]);
    swap_layout_8(dst.data[2], data1_cache);
    swap_layout_8(dst.data[3], src.data[3]);
}
/**
 * @brief Swaps the layout of a register tile.
 *
 * This function swaps the layout of a register tile by iterating over its height and width
 * and performing layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register tile where the result will be stored.
 * @param src[in] Reference to the source register tile to be swapped.
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt<T2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            swap_layout(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/**
 * @brief Swaps the layout of a register base tile in place.
 *
 * This function swaps the layout of a register base tile in place by casting it to the
 * transposed layout type and then performing the layout swap.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register base tile to be swapped in place.
 * @return A reference to the swapped register base tile.
 */
template<typename T2, ducks::rt_layout::all layout>
__device__ inline rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(const rt_base<T2, layout> &src) {
    rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst = *(rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>*)(&src);
    swap_layout(dst, src);
    return dst;
}
/**
 * @brief Swaps the layout of a register tile in place.
 *
 * This function swaps the layout of a register tile in place by iterating over its height and width
 * and performing in-place layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be swapped in place.
 * @return A reference to the swapped register tile.
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(rt<T2, _height, _width, layout> &tile) {
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < tile.width; j++) {
            swap_layout_inplace(tile.tiles[i][j]);
        }
    }
    return *(rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type>*)(&tile);
}

/* ----------  TRANSPOSE  ---------- */

/**
 * @brief Transposes a register base tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register base tile to be transposed.
 */
template<typename T2, ducks::rt_layout::all layout>
__device__ inline void transpose(rt_base<T2, layout> &dst, const rt_base<T2, layout> &src) {
    swap_layout_8(dst.data[0], src.data[0]);
    // technically this swap can be eliminated if we simply reinterpret the layout of the registers
    // everywhere else in the code, but that feels... very likely to cause bugs and not worth it. 
    T2 data1_cache = src.data[1]; // important for swap!
    swap_layout_8(dst.data[1], src.data[2]);
    swap_layout_8(dst.data[2], data1_cache);
    swap_layout_8(dst.data[3], src.data[3]);
}
/**
 * @brief Transposes a register tile.
 * 
 * This function is marked "sep", which means that the registers underlying dst MUST be separate
 * from the registers underlying src.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the src register tile, and the width of the dst tile.
 * @tparam _width The width of the src register tile, and the height of the dst tile.
 * @tparam layout The layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register tile to be transposed.
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void transpose_sep(rt<T2, _width, _height, layout> &dst, const rt<T2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < _height; i++) {
        #pragma unroll
        for(int j = 0; j < _width; j++) {
            transpose(dst.tiles[j][i], src.tiles[i][j]);
        }
    }
}

/**
 * @brief Transposes a register base tile in-place.
 *
 * @tparam T2 The data type of the register base tile elements.
 * @tparam layout The current layout of the register base tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register base tile.
 */
template<typename T2, ducks::rt_layout::all layout>
__device__ inline rt_base<T2, layout>& transpose_inplace(rt_base<T2, layout> &src) {
    transpose(src, src);
    return src;
}
/**
 * @brief Transposes a square register tile in-place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height (in units of 16) of the src register tile, and the width of the dst tile. (Must be the same as _width.)
 * @tparam _width The width (in units of 16) of the src register tile, and the height of the dst tile. (Must be the same as _height.)
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register tile.
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _height, _width, layout>& transpose_inplace(rt<T2, _height, _width, layout> &tile) {
    static_assert(_width == _height, "in-place register tile transpose is only allowed for square tiles.");
    #pragma unroll
    for(int i = 0; i < _height; i++) {
        #pragma unroll
        for(int j = 0; j < i; j++) {
            rt_base<T2, layout> tmp;
            copy(tmp, tile.tiles[i][j]);
            transpose(tile.tiles[i][j], tile.tiles[j][i]);
            transpose(tile.tiles[j][i], tmp);
        }
        transpose_inplace(tile.tiles[i][i]);
    }
    return tile;
}

/* ----------  TYPE SWAPS  ---------- */

/**
 * @brief Copies a register base tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam layout The current layout of the register base tile.
 * @param[out] dst A reference to the destination register base tile.
 * @param[in] src A reference to the source register base tile.
 */
template<typename T, typename U, ducks::rt_layout::all layout>
__device__ static inline void copy(rt_base<T, layout> &dst, const rt_base<U, layout> &src) {
    using T2 = typename base_types::packing<T>::packed_type;
    using U2 = typename base_types::packing<U>::packed_type;
    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_types::convertor<T2, U2>::convert(src.data[k]);
    }
}
/**
 * @brief Copies a register tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam _height The height (in units of 16) of the register tiles.
 * @tparam _width The width (in units of 16) of the register tiles.
 * @tparam layout The current layout of the register tile.
 * @param[out] dst A reference to the destination register tile.
 * @param[in] src A reference to the source register tile.
 */
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            copy(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/* ----------  CAUSAL  ---------- */

/**
 * @brief Makes a square register tile causal by zeroing elements above the main diagonal.
 *
 * This function modifies a square register tile in-place to make it causal. All elements
 * above the main diagonal are set to zero, while elements on or below the main diagonal
 * are left unchanged.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be made causal.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j < i) { // below the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j > i) { // above the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint32_t MASK_X = 0xFF773311, MASK_Y = 0xF7733110; // magic numbers for on-diagonal core matrices
                dst.tiles[i][j].data[1] = src.tiles[i][j].data[1]; // below diagonal, copy
                dst.tiles[i][j].data[2] = packed_val; // above diagonal, zero
                if((MASK_X >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }
                else {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                }
                if((MASK_Y >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
                else {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                }
            }
        }
    }
}


/* ----------  SUBTILE  ---------- */

/**
* @brief Returns a reference to a subtile of the given tile.
*
* @tparam subtile_height The height of the subtile.
* @tparam RT The type of the input tile, which must satisfy the ducks::rt::all concept.
* @param src The input tile.
* @param idx The index of the subtile.
* @return A reference to the subtile.
*
* @note The subtile height must evenly divide the tile height.
*/
template<int subtile_height, ducks::rt::all RT>
__device__ inline rt<typename RT::T, subtile_height, RT::width, typename RT::layout> &subtile_inplace(RT & src, int idx) {
    static_assert(RT::height % subtile_height == 0, "subtile height should evenly divide tile height.");
    return reinterpret_cast<rt<typename RT::T, subtile_height, RT::width, typename RT::layout>&>(
        src.tiles[idx*subtile_height]
    );
}

}