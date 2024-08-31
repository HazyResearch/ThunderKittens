#include "global_to_register.cuh"

#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER

template<typename T>
struct load_store {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "reg_loadstore_gmem=half" :
                                                                                         "reg_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::gt::l::all GTL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GTL input, GTL output) {
        kittens::rt_bf<H, W, L> reg_tile;
        kittens::load(reg_tile, input, 0, 0, 0, 0);
        kittens::store(output, reg_tile, 0, 0, 0, 0);
    }
};

void warp::memory::tile::global_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/global_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_2d_warp<load_store<float>, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<load_store<float>, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<load_store<kittens::bf16>, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<load_store<kittens::bf16>, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<load_store<kittens::half>, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<load_store<kittens::half>, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
}

#endif