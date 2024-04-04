#include "testing_flags.cuh"

#ifdef TEST_WARP_MEMORY_SHARED_TO_REGISTER

#include "testing_commons.cuh"

namespace warp {
namespace memory {
namespace shared_to_register {

struct load_store {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L, kittens::ducks::rt_layout::all RL> using valid = std::bool_constant<NW == 1 &&
        (!std::is_same_v<L, kittens::ducks::st_layout::tma_swizzle> || W == 1 || W == 2 || W == 4) && W*H<=64>;
    static inline const std::string test_identifier = "shared_reg_loadstore";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L, kittens::ducks::rt_layout::all RL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L, kittens::ducks::rt_layout::all RL> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::st_bf<H, W, L> &shared_tile = al.allocate<kittens::st_bf<H, W, L>>();
        kittens::load(shared_tile, input, W*16);
        __syncthreads();
        kittens::rt_bf<H, W, RL> reg_tile;
        kittens::load(reg_tile, shared_tile);
        __syncthreads();
        kittens::store(shared_tile, reg_tile);
        __syncthreads();
        kittens::store(output, shared_tile, W*16);
    }
};

struct vec_load_store {
    template<int S, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "shared_reg_vec_loadstore";
    template<int S, int NW, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) o_ref[i] = i_ref[i] + 1.f; // just a dummy op to prevent optimization away
    }
    template<int S, int NW, kittens::ducks::rt_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::col_vec<kittens::st_bf<S, S>> &shared_vec = al.allocate<kittens::col_vec<kittens::st_bf<S, S>>>();
        kittens::col_vec<kittens::rt_bf<S, S, L>> reg_vec;
        kittens::load(shared_vec, input);
        __syncthreads();
        kittens::load(reg_vec, shared_vec);
        kittens::add(reg_vec, reg_vec, float(1.)); // TODO: CHANGE HOST TOO
        kittens::store(shared_vec, reg_vec);
        __syncthreads();
        kittens::store(output, shared_vec);
    }
};

void tests(test_data &results);

}
}
}

#endif