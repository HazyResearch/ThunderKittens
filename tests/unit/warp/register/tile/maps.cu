#include "maps.cuh"

#ifdef TEST_WARP_REGISTER_TILE_MAPS

struct test_exp {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_exp";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) o_ref[i] = ::expf(i_ref[i]);
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt_bf<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        kittens::exp(reg_tile, reg_tile);
        kittens::warp::store(output, reg_tile, {});
    }
};
struct test_tril {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_tril";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // triangular lower, with diagonal starting at row_idx 4H
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i*W*16 + j] = j<=i+(4*H) ? i_ref[i*W*16 + j] : 0;
    }
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        auto masked_tile = kittens::apply(reg_tile,
            []__device__(int row, int col, float val) -> float {
                return col<=row+(4*H) ? val : 0.f;
            }
        );
        kittens::warp::store(output, masked_tile, {});
    }
};

void warp::reg::tile::maps::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_size_2d_warp<test_exp, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_exp, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<test_tril, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_tril, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
}

#endif