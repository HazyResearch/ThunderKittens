#include "conversions.cuh"

#ifdef TEST_WARP_REGISTER_VEC_CONVERSIONS

struct vec_copy_convert {
    template<int S, int NW, kittens::ducks::rt_layout::all L1, kittens::ducks::rt_layout::all L2>
    using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_vec_convert";
    template<int S, int NW, kittens::ducks::rt_layout::all L1, kittens::ducks::rt_layout::all L2>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, kittens::ducks::rt_layout::all L1, kittens::ducks::rt_layout::all L2>
    __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        kittens::col_vec<kittens::rt_bf<S, S, L1>> vec1;
        kittens::col_vec<kittens::rt_bf<S, S, L2>> vec2;
        kittens::load(vec1, input);
        kittens::copy(vec2, vec1);
        kittens::store(output, vec2);
    }
};

void warp::reg::vec::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/vec/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::ducks::rt_layout::row, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::ducks::rt_layout::row, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::ducks::rt_layout::col, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d_warp<vec_copy_convert, SIZE, kittens::ducks::rt_layout::col, kittens::ducks::rt_layout::col>::run(results);
}

#endif