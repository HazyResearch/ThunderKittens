#include "global_to_register.cuh"

#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER

struct load_store {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_loadstore";
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        kittens::rt_bf<H, W, L> reg_tile;
        kittens::load(reg_tile, input, W*16);
        kittens::store(output, reg_tile, W*16);
    }
};

void warp::memory::tile::global_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/global_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_2d_warp<load_store, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<load_store, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
}

#endif