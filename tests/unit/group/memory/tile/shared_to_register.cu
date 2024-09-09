#include "shared_to_register.cuh"

#ifdef TEST_GROUP_MEMORY_TILE_SHARED_TO_REGISTER

template<typename T>
struct group_shared_reg_load_store {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all RL> using valid = std::bool_constant<H%NW==0 &&
        W*H<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_shared_reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_shared_reg_loadstore_gmem=half" :
                                                                                         "group_shared_reg_loadstore_gmem=float";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all RL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all RL> __device__ static void device_func(const GL &input, GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::st<dtype, H, W> &shared_tile = al.allocate<kittens::st<dtype, H, W>>();
        G::load(shared_tile, input, {});
        __syncthreads();
        kittens::rt<dtype, H/NW, W, RL> reg_tile;
        G::load(reg_tile, shared_tile);
        __syncthreads();
        G::store(shared_tile, reg_tile);
        __syncthreads();
        G::store(output, shared_tile, {});
    }
};

void group::memory::tile::shared_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/tile/shared_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_2d<group_shared_reg_load_store<float>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<float>, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<float>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<float>, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<float>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<float>, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::bf16>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::bf16>, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::bf16>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::bf16>, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::bf16>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::bf16>, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::half>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::half>, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::half>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::half>, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::half>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_shared_reg_load_store<kittens::half>, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
}

#endif