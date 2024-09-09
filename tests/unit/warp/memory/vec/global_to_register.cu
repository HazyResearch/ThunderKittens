#include "global_to_register.cuh"

#ifdef TEST_WARP_MEMORY_VEC_GLOBAL_TO_REGISTER

template<typename T>
struct reg_vec_load_store {
    using dtype = T;
    template<int S, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<dtype, kittens::bf16> ? "reg_vec_loadstore_gmem=bf16" :
                                                      std::is_same_v<dtype, kittens::half> ? "reg_vec_loadstore_gmem=half" :
                                                                                             "reg_vec_loadstore_gmem=float";
    template<int S, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL &input, GL &output) {
        kittens::col_vec<kittens::rt_bf<S, S, L>> reg_vec;
        kittens::load(reg_vec, input, {});
        kittens::store(output, reg_vec, {});
    }
};

void warp::memory::vec::global_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/global_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_gmem_type_1d_warp<reg_vec_load_store, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, SIZE, kittens::ducks::rt_layout::col>::run(results);
}


#endif