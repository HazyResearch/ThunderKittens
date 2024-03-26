#include "testing_flags.cuh"

#ifdef TEST_WARP_MEMORY_GLOBAL_TO_REGISTER

#include "testing_commons.cuh"

namespace warp {
namespace memory {
namespace global_to_register {

struct load_store {
    template<int H, int W, int NW, ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_loadstore";
    template<int H, int W, int NW, ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, ducks::rt_layout::all L> __device__ static void device_func(const bf16 *input, bf16 *output) {
        rt_bf<H, W, L> reg_tile;
        load(reg_tile, input, W*16);
        store(output, reg_tile, W*16);
    }
};

struct vec_load_store {
    template<int S, int NW, ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_vec_loadstore";
    template<int S, int NW, ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, ducks::rt_layout::all L> __device__ static void device_func(const bf16 *input, bf16 *output) {
        col_vec<rt_bf<S, S, L>> reg_vec;
        load(reg_vec, input);
        store(output, reg_vec);
    }
};

void tests(test_data &results);

}
}
}

#endif