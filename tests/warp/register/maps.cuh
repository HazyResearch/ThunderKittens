#include "testing_flags.cuh"

#ifdef TEST_WARP_REGISTER_MAPS

#include "testing_commons.cuh"

namespace warp {
namespace reg {
namespace maps {
struct test_exp {
    template<int H, int W, int NW, ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_exp";
    template<int H, int W, int NW, ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) o_ref[i] = ::expf(i_ref[i]); // overwrite the whole thing
    }
    template<int H, int W, int NW, ducks::rt_layout::all L> __device__ static void device_func(const bf16 *input, bf16 *output) {
        rt_bf<H, W, L> reg_tile;
        load(reg_tile, input, W*16);
        kittens::exp(reg_tile, reg_tile);
        store(output, reg_tile, W*16);
    }
};

void tests(test_data &results);
}
}
}

#endif