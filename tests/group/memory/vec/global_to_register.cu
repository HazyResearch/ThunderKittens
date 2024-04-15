#include "global_to_register.cuh"

#ifdef TEST_GROUP_MEMORY_VEC_GLOBAL_TO_REGISTER

struct vec_load_store {
    template<int S, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<S%NW==0 && S<=64>; // this is group-level
    static inline const std::string test_identifier = "reg_vec_loadstore";
    template<int S, int NW, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, kittens::ducks::rt_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        using G = kittens::group<NW>;
        kittens::col_vec<kittens::rt_bf<S, S, L>> reg_vec;
        G::load(reg_vec, input);
        G::store(output, reg_vec);
    }
};

void group::memory::vec::global_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/vec/global_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d<vec_load_store, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_1d<vec_load_store, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_1d<vec_load_store, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
}

#endif