#include "global_to_register.cuh"

#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_REGISTER

template<typename T>
struct group_load_store {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_reg_loadstore_gmem=half" :
                                                                                         "group_reg_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __device__ static void device_func(const dtype *input, dtype *output) {
        using G = kittens::group<NW>;
        kittens::rt<dtype, H/NW, W, L> reg_tile;
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
                         
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
}

#endif