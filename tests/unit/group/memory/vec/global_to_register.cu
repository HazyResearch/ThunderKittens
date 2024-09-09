#include "global_to_register.cuh"

#ifdef TEST_GROUP_MEMORY_VEC_GLOBAL_TO_REGISTER

template<typename T>
struct vec_load_store {
    using dtype = T;
    template<int S, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<S%NW==0 && S<=64>; // this is group-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_reg_vec_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_reg_vec_loadstore_gmem=half" :
                                                                                         "group_reg_vec_loadstore_gmem=float";
    template<int S, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL &input, GL &output) {
        using G = kittens::group<NW>;
        kittens::col_vec<kittens::rt<dtype, S, S, L>> reg_vec;
        G::load(reg_vec, input, {});
        G::store(output, reg_vec, {});
    }
};

void group::memory::vec::global_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/vec/global_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d<vec_load_store<float>, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store<float>, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_1d<vec_load_store<float>, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store<float>, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_1d<vec_load_store<float>, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store<float>, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
}

#endif