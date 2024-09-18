#include "conversions.cuh"

#ifdef TEST_WARP_SHARED_VEC_CONVERSIONS

template<typename T>
struct shared_vec_convert {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_vec_convert_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_vec_convert_gmem=half" :
                                                                                         "shared_vec_convert_gmem=float";
    template<int S, int NW, gl_t GL>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, gl_t GL>
    __device__ static void device_func(const GL &input, GL &output) {
        __shared__ kittens::col_vec<kittens::st<dtype, 16*S, 16*S>> vec1;
        __shared__ kittens::col_vec<kittens::st<dtype, 16*S, 16*S>> vec2;
        kittens::load(vec1, input, {});
        kittens::copy(vec2, vec1);
        kittens::store(output, vec2, {});
    }
};

void warp::shared::vec::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/vec/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_gmem_type_1d_warp<shared_vec_convert, SIZE>::run(results);
}

#endif