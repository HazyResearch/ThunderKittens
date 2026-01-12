#include "reductions.cuh"

#ifdef TEST_GROUP_REG_TILE_REDUCTIONS

struct normalize_row {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_row";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_sum         += i_ref[i*W*16+j];
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] /= row_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::warp::row_sum(accum, reg_tile);
        kittens::warp::div_row(reg_tile, reg_tile, accum);
        kittens::warp::store(output, reg_tile, {});
    }
};
struct normalize_col {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_col";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < W*16; i++) {
            float col_sum = 0;
            for(int j = 0; j < H*16; j++) {
                o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                col_sum         += i_ref[i+j*W*16];
            }
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] /= col_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::row_vec accum;
        kittens::warp::col_sum(accum, reg_tile);
        kittens::warp::div_col(reg_tile, reg_tile, accum);
        kittens::warp::store(output, reg_tile, {});
    }
};
struct broadcast_row {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_broadcast_row";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_sum         += i_ref[i*W*16+j];
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] = row_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::warp::row_sum(accum, reg_tile);
        kittens::warp::broadcast_row(reg_tile, accum);
        kittens::warp::store(output, reg_tile, {});
    }
};
struct broadcast_col {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_broadcast_col";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < W*16; i++) {
            float col_sum = 0;
            for(int j = 0; j < H*16; j++) {
                o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                col_sum         += i_ref[i+j*W*16];
            }
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] = col_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::row_vec accum;
        kittens::warp::col_sum(accum, reg_tile);
        kittens::warp::broadcast_col(reg_tile, accum);
        kittens::warp::store(output, reg_tile, {});
    }
};

void group::reg::tile::reductions::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/register/tile/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_size_2d_warp<normalize_row, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<normalize_row, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<normalize_col, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<normalize_col, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<broadcast_row, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<broadcast_row, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<broadcast_col, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<broadcast_col, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    std::cout << std::endl;
}

#endif