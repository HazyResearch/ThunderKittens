#include "reductions.cuh"

#ifdef TEST_WARP_REGISTER_VEC_REDUCTIONS

struct vec_norm {
    template<int S, int NW, kittens::ducks::rt_layout::all L>
    using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_vec_norm";
    template<int S, int NW, gvl_t GVL, kittens::ducks::rt_layout::all L>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        float f = 1.f;
        for(int i = 0; i < o_ref.size(); i++) f += i_ref[i];
        for(int i = 0; i < o_ref.size(); i++) o_ref[i] = i_ref[i] / f;
    }
    template<int S, int NW, gvl_t GVL, kittens::ducks::rt_layout::all L>
    __device__ static void device_func(const GVL &input, GVL &output) {
        kittens::col_vec<kittens::rt_fl<S, S, L>> vec;
        kittens::load(vec, input, {});
        float f = 1.f;
        kittens::sum(f, vec, f);
        kittens::div(vec, vec, f);
        kittens::store(output, vec, {});
    }
};

void warp::reg::vec::reductions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/vec/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d_warp<vec_norm, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d_warp<vec_norm, SIZE, kittens::ducks::rt_layout::col>::run(results);
}

#endif