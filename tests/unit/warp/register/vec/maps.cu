#include "maps.cuh"

#ifdef TEST_WARP_REGISTER_VEC_MAPS

struct vec_add1 {
    template<int S, int NW, kittens::ducks::rv_layout::all L>
    using valid = std::bool_constant<NW == 1 && S<=64 && sizeof(dtype) != 1>; // this is warp-level
    static inline const std::string test_identifier = "reg_vec_add1";
    template<int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < o_ref.size(); i++) o_ref[i] = i_ref[i]+1.; // overwrite the whole thing
    }
    template<int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L>
    __device__ static void device_func(const GL &input, const GL &output) {
        kittens::rv_fl<16*S, L> vec;
        kittens::load(vec, input, {});
        kittens::add(vec, vec, 1.f);
        kittens::store(output, vec, {});
    }
};

void warp::reg::vec::maps::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/vec/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d_warp<vec_add1, SIZE, kittens::ducks::rv_layout::naive>::run(results);
    sweep_size_1d_warp<vec_add1, SIZE, kittens::ducks::rv_layout::align>::run(results);
    sweep_size_1d_warp<vec_add1, SIZE, kittens::ducks::rv_layout::ortho>::run(results);
}

#endif