#include "conversions.cuh"

#ifdef TEST_GROUP_SHARED_VEC_CONVERSIONS

struct vec_copy {
    template<int S, int NW> using valid = std::bool_constant<S%NW==0 && S<=64>; // this is group-level
    static inline const std::string test_identifier = "shared_vec_convert";
    template<int S, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        __shared__ kittens::col_vec<kittens::st_bf<16*S, 16*S>> vec1;
        __shared__ kittens::col_vec<kittens::st_bf<16*S, 16*S>> vec2;
        G::load(vec1, input, {});
        G::sync(0);
        G::copy(vec2, vec1);
        G::sync(0);
        G::store(output, vec2, {});
    }
};

void group::shared::vec::conversions::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/shared/vec/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d<vec_copy, SIZE, 2>::run(results);
    sweep_size_1d<vec_copy, SIZE, 4>::run(results);
    sweep_size_1d<vec_copy, SIZE, 12>::run(results);
    std::cout << std::endl;
}

#endif