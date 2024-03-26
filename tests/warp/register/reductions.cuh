#include "testing_flags.cuh"

#ifdef TEST_WARP_REGISTER_REDUCTIONS

#include "testing_commons.cuh"

namespace warp {
namespace reg {
namespace reductions {
struct normalize_row {
    template<int H, int W, int NW, ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_row";
    template<int H, int W, int NW, ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_sum         += i_ref[i*W*16+j];
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] /= row_sum;
        }
    }
    template<int H, int W, int NW, ducks::rt_layout::all L> __device__ static void device_func(const bf16 *input, bf16 *output) {
        rt_fl<H, W, L> reg_tile;
        load(reg_tile, input, W*16);
        typename rt_fl<H, W, L>::col_vec accum;
        row_sum(accum, reg_tile);
        div_row(reg_tile, reg_tile, accum);
        store(output, reg_tile, W*16);
    }
};
struct normalize_col {
    template<int H, int W, int NW, ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_col";
    template<int H, int W, int NW, ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < W*16; i++) {
            float col_sum = 0;
            for(int j = 0; j < H*16; j++) {
                o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                col_sum         += i_ref[i+j*W*16];
            }
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] /= col_sum;
        }
    }
    template<int H, int W, int NW, ducks::rt_layout::all L> __device__ static void device_func(const bf16 *input, bf16 *output) {
        rt_fl<H, W, L> reg_tile;
        load(reg_tile, input, W*16);
        typename rt_fl<H, W, L>::row_vec accum;
        col_sum(accum, reg_tile);
        div_col(reg_tile, reg_tile, accum);
        store(output, reg_tile, W*16);
    }
};
struct broadcast_row {
    template<int H, int W, int NW, ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_broadcast_row";
    template<int H, int W, int NW, ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_sum         += i_ref[i*W*16+j];
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] = row_sum;
        }
    }
    template<int H, int W, int NW, ducks::rt_layout::all L> __device__ static void device_func(const bf16 *input, bf16 *output) {
        rt_fl<H, W, L> reg_tile;
        load(reg_tile, input, W*16);
        typename rt_fl<H, W, L>::col_vec accum;
        row_sum(accum, reg_tile);
        kittens::broadcast_row(reg_tile, accum);
        store(output, reg_tile, W*16);
    }
};
struct broadcast_col {
    template<int H, int W, int NW, ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_broadcast_col";
    template<int H, int W, int NW, ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < W*16; i++) {
            float col_sum = 0;
            for(int j = 0; j < H*16; j++) {
                o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                col_sum         += i_ref[i+j*W*16];
            }
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] = col_sum;
        }
    }
    template<int H, int W, int NW, ducks::rt_layout::all L> __device__ static void device_func(const bf16 *input, bf16 *output) {
        rt_fl<H, W, L> reg_tile;
        load(reg_tile, input, W*16);
        typename rt_fl<H, W, L>::row_vec accum;
        col_sum(accum, reg_tile);
        kittens::broadcast_col(reg_tile, accum);
        store(output, reg_tile, W*16);
    }
};

void tests(test_data &results);
}
}
}

#endif