#include "shared_to_register.cuh"

#ifdef TEST_GROUP_MEMORY_TILE_SHARED_TO_REGISTER

struct load_store {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L, kittens::ducks::rt_layout::all RL> using valid = std::bool_constant<H%NW==0 &&
        (!std::is_same_v<L, kittens::ducks::st_layout::swizzle> || W == 1 || W == 2 || W == 4 || W == 8 || W == 16) && W*H<=64>;
    static inline const std::string test_identifier = "shared_reg_loadstore";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L, kittens::ducks::rt_layout::all RL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L, kittens::ducks::rt_layout::all RL> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::st_bf<H, W, L> &shared_tile = al.allocate<kittens::st_bf<H, W, L>>();
        G::load(shared_tile, input, W*16);
        __syncthreads();
        kittens::rt_bf<H/NW, W, RL> reg_tile;
        G::load(reg_tile, shared_tile);
        __syncthreads();
        G::store(shared_tile, reg_tile);
        __syncthreads();
        G::store(output, shared_tile, W*16);
    }
};

void group::memory::tile::shared_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/tile/shared_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_st_layout_size_2d<load_store, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_st_layout_size_2d<load_store, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_st_layout_size_2d<load_store, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_st_layout_size_2d<load_store, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_st_layout_size_2d<load_store, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_st_layout_size_2d<load_store, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
}

#endif