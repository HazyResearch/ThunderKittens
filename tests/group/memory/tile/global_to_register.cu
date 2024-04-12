#include "global_to_register.cuh"

#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_REGISTER

struct load_store {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "reg_loadstore";
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        using G = kittens::group<NW>;
        kittens::rt_bf<H/NW, W, L> reg_tile;
        G::load(reg_tile, input, W*16);
        G::store(output, reg_tile, W*16);
    }
};

void group::memory::tile::global_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/tile/global_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_2d<load_store, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<load_store, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<load_store, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<load_store, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<load_store, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<load_store, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
}

#endif