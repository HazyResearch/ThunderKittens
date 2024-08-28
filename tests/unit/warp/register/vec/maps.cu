#include "maps.cuh"

#ifdef TEST_WARP_REGISTER_VEC_MAPS

struct vec_add1 {
    template<int S, int NW, kittens::ducks::rt_layout::all L>
    using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_vec_add1";
    template<int S, int NW, kittens::ducks::rt_layout::all L>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < o_ref.size(); i++) o_ref[i] = i_ref[i]+1.; // overwrite the whole thing
    }
    template<int S, int NW, kittens::ducks::rt_layout::all L>
    __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        kittens::col_vec<kittens::rt_fl<S, S, L>> vec;
        kittens::load(vec, input);
        kittens::add(vec, vec, __float2bfloat16(1.));
        kittens::store(output, vec);
    }
};

void warp::reg::vec::maps::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/vec/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d_warp<vec_add1, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d_warp<vec_add1, SIZE, kittens::ducks::rt_layout::col>::run(results);
}

#endif