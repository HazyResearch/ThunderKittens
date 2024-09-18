#include "maps.cuh"

#ifdef TEST_GROUP_SHARED_VEC_MAPS

struct vec_add1 {
    template<int S, int NW>
    using valid = std::bool_constant<S%NW==0 && S<=64>; // this is group-level
    static inline const std::string test_identifier = "shared_vec_add1";
    template<int S, int NW, gl_t GL>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < o_ref.size(); i++) o_ref[i] = i_ref[i]+1.; // overwrite the whole thing
    }
    template<int S, int NW, gl_t GL>
    __device__ static void device_func(const GL &input, GL &output) {
        using G = kittens::group<NW>;
        __shared__ kittens::col_vec<kittens::st_bf<16*S, 16*S>> vec;
        G::load(vec, input, {});
        G::sync();
        G::add(vec, vec, __float2bfloat16(1.));
        G::sync();
        G::store(output, vec, {});
    }
};

void group::shared::vec::maps::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/shared/vec/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d<vec_add1, SIZE, 2>::run(results);
    sweep_size_1d<vec_add1, SIZE, 4>::run(results);
    sweep_size_1d<vec_add1, SIZE, 12>::run(results);
}

#endif