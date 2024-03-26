#include "testing_flags.cuh"

#ifdef TEST_WARP_REGISTER_CONVERSIONS

#include "testing_commons.cuh"

namespace warp {
namespace reg {
namespace conversions {
struct swap_layout {
    template<int H, int W, int NW, ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_swaplayout";
    template<int H, int W, int NW, ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, ducks::rt_layout::all L> __device__ static void device_func(const bf16 *input, bf16 *output) {
        rt_bf<H, W, L> reg_tile;
        load(reg_tile, input, W*16);
        auto reg_tile_other_layout = swap_layout_inplace(reg_tile);
        store(output, reg_tile_other_layout, W*16);
    }
};
struct transpose {
    template<int H, int W, int NW, ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_transpose";
    template<int H, int W, int NW, ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i+j*H*16] = i_ref[i*W*16+j];
    }
    template<int H, int W, int NW, ducks::rt_layout::all L> __device__ static void device_func(const bf16 *input, bf16 *output) {
        rt_bf<H, W, L> reg_tile;
        rt_bf<W, H, L> reg_tile_transpose;
        load(reg_tile, input, W*16);
        transpose_sep(reg_tile_transpose, reg_tile);
        store(output, reg_tile_transpose, H*16);
    }
};
struct type_convert {
    template<int H, int W, int NW, typename T2, typename U2> using valid = std::bool_constant<NW == 1 && W*H<=32>; // this is warp-level
        static inline const std::string test_identifier = "reg_typeconvert";
        template<int H, int W, int NW, typename T2, typename U2> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, typename T2, typename U2> __device__ static void device_func(const bf16 *input, bf16 *output) {
        rt<U2, H, W> reg_tile_U2;
        rt<T2, H, W> reg_tile_T2;
        load(reg_tile_U2, input, W*16);
        copy(reg_tile_T2, reg_tile_U2);
        store(output, reg_tile_T2, W*16);
    }
};
struct subtile {
    template<int H, int W, int NW, typename ST_H> using valid = std::bool_constant<NW == 1 && (H%(ST_H::value))==0 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_subtile";
    template<int H, int W, int NW, typename ST_H> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i*W*16 + j] = i_ref[i*W*16 + j] + float(i/(ST_H::value*16));
    }
    template<int H, int W, int NW, typename _ST_H> __device__ static void device_func(const bf16 *input, bf16 *output) {
        constexpr int ST_H = _ST_H::value;
        rt_fl<H, W> reg_tile;
        load(reg_tile, input, W*16);
        #pragma unroll
        for(int i = 0; i < H/ST_H; i++) {
            auto &ref = subtile_inplace<ST_H>(reg_tile, i);
            add(ref, ref, float(i));
        }
        store(output, reg_tile, W*16);
    }
};

void tests(test_data &results);
}
}
}

#endif