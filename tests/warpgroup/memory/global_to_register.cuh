#include "testing_flags.cuh"

#ifdef TEST_WARPGROUP_MEMORY_GLOBAL_TO_REGISTER

#include "testing_commons.cuh"

namespace warpgroup {
namespace memory {
namespace global_to_register {

struct load_store {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<H%4 == 0 && NW == 4 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "wg_reg_loadstore";
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        kittens::rt_bf<H/4, W, L> reg_tile;
        kittens::warpgroup::load(reg_tile, input, W*16);
        kittens::warpgroup::store(output, reg_tile, W*16);
    }
};

struct vec_load_store {
    template<int S, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<S%4 == 0 && NW == 4 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "wg_reg_vec_loadstore";
    template<int S, int NW, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, kittens::ducks::rt_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        kittens::col_vec<kittens::rt_bf<S/4, S/4, L>> reg_vec;
        kittens::warpgroup::load(reg_vec, input);
        kittens::warpgroup::store(output, reg_vec);
    }
};

void tests(test_data &results);

}
}
}

#endif