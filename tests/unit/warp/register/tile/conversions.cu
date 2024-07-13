#include "conversions.cuh"

#ifdef TEST_WARP_REGISTER_TILE_CONVERSIONS

struct test_swap_layout {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_swaplayout";
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        kittens::rt_bf<H, W, L> reg_tile;
        kittens::load(reg_tile, input, W*16);
        auto reg_tile_other_layout = kittens::swap_layout_inplace(reg_tile);
        kittens::store(output, reg_tile_other_layout, W*16);
    }
};
struct test_transpose {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_transpose";
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i+j*H*16] = i_ref[i*W*16+j];
    }
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        kittens::rt_bf<H, W, L> reg_tile;
        kittens::rt_bf<W, H, L> reg_tile_transpose;
        kittens::load(reg_tile, input, W*16);
        kittens::transpose_sep(reg_tile_transpose, reg_tile);
        kittens::store(output, reg_tile_transpose, H*16);
    }
};
struct test_type_convert {
    template<int H, int W, int NW, typename T2, typename U2> using valid = std::bool_constant<NW == 1 && W*H<=32>; // this is warp-level
        static inline const std::string test_identifier = "reg_typeconvert";
        template<int H, int W, int NW, typename T2, typename U2> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, typename T2, typename U2> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        kittens::rt<U2, H, W> reg_tile_U2;
        kittens::rt<T2, H, W> reg_tile_T2;
        kittens::load(reg_tile_U2, input, W*16);
        kittens::copy(reg_tile_T2, reg_tile_U2);
        kittens::store(output, reg_tile_T2, W*16);
    }
};
struct test_subtile {
    template<int H, int W, int NW, typename ST_H> using valid = std::bool_constant<NW == 1 && (H%(ST_H::value))==0 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_subtile";
    template<int H, int W, int NW, typename ST_H> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i*W*16 + j] = i_ref[i*W*16 + j] + float(i/(ST_H::value*16));
    }
    template<int H, int W, int NW, typename _ST_H> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        constexpr int ST_H = _ST_H::value;
        kittens::rt_fl<H, W> reg_tile;
        kittens::load(reg_tile, input, W*16);
        #pragma unroll
        for(int i = 0; i < H/ST_H; i++) {
            auto &ref = kittens::subtile_inplace<ST_H>(reg_tile, i);
            kittens::add(ref, ref, float(i));
        }
        kittens::store(output, reg_tile, W*16);
    }
};
struct test_make_causal {
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_make_causal";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i*W*16 + j] = j<=i ? i_ref[i*W*16 + j] : 0;
    }
    template<int H, int W, int NW> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        kittens::rt_fl<H, W> reg_tile;
        kittens::load(reg_tile, input, W*16);
        kittens::make_causal(reg_tile, reg_tile);
        kittens::store(output, reg_tile, W*16);
    }
};

void warp::reg::tile::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_size_2d_warp<test_swap_layout, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_swap_layout, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);

    sweep_size_2d_warp<test_transpose, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_transpose, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);

    sweep_size_2d_warp<test_type_convert, SIZE, SIZE, float, kittens::bf16>::run(results);
    sweep_size_2d_warp<test_type_convert, SIZE, SIZE, kittens::bf16, float>::run(results);
    sweep_size_2d_warp<test_type_convert, SIZE, SIZE, float, kittens::half>::run(results);
    sweep_size_2d_warp<test_type_convert, SIZE, SIZE, kittens::half, float>::run(results);
    sweep_size_2d_warp<test_type_convert, SIZE, SIZE, kittens::half, kittens::bf16>::run(results);
    sweep_size_2d_warp<test_type_convert, SIZE, SIZE, kittens::bf16, kittens::half>::run(results);

    sweep_size_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    sweep_size_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    sweep_size_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    sweep_size_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);

    sweep_size_2d_warp<test_make_causal, SIZE, SIZE>::run(results);
}

#endif