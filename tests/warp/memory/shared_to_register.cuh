#include "testing_flags.cuh"

#ifdef TEST_WARP_MEMORY_SHARED_TO_REGISTER

#include "testing_commons.cuh"

namespace warp {
namespace memory {
namespace shared_to_register {

struct load_store {
    template<int H, int W, int NW, ducks::st_layout::all L, ducks::rt_layout::all RL> using valid = std::bool_constant<NW == 1 &&
        (!std::is_same_v<L, ducks::st_layout::tma_swizzle> || W == 1 || W == 2 || W == 4) && W*H<=64>;
    static inline const std::string test_identifier = "shared_reg_loadstore";
    template<int H, int W, int NW, ducks::st_layout::all L, ducks::rt_layout::all RL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, ducks::st_layout::all L, ducks::rt_layout::all RL> __device__ static void device_func(const bf16 *input, bf16 *output) {
        extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
        shared_allocator<16> al((int*)&__shm[0]); 
        st_bf<H, W, L> &shared_tile = al.allocate<st_bf<H, W, L>>();
        load(shared_tile, input, W*16);
        __syncthreads();
        rt_bf<H, W, RL> reg_tile;
        load(reg_tile, shared_tile);
        __syncthreads();
        store(shared_tile, reg_tile);
        __syncthreads();
        store(output, shared_tile, W*16);
    }
};

struct vec_load_store {
    template<int S, int NW, ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "shared_reg_vec_loadstore";
    template<int S, int NW, ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) o_ref[i] = i_ref[i] + 1.f; // just a dummy op to prevent optimization away
    }
    template<int S, int NW, ducks::rt_layout::all L> __device__ static void device_func(const bf16 *input, bf16 *output) {
        extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
        shared_allocator al((int*)&__shm[0]); 
        col_vec<st_bf<S, S>> &shared_vec = al.allocate<col_vec<st_bf<S, S>>>();
        col_vec<rt_bf<S, S, L>> reg_vec;
        load(shared_vec, input);
        __syncthreads();
        load(reg_vec, shared_vec);
        add(reg_vec, reg_vec, float(1.)); // TODO: CHANGE HOST TOO
        store(shared_vec, reg_vec);
        __syncthreads();
        store(output, shared_vec);
    }
};

void tests(test_data &results);

}
}
}

#endif